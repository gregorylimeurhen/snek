import random
import time
import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm.auto
import src.environment as env
import src.models as models
import src.policies as policies
import src.utils as u


def _paths(cfg):
	root = u.ensure_dir(cfg["data_dir"])
	run = u.ensure_dir(root / cfg["run_name"])
	return root, run


def _blank():
	return {
		"starts_snake": [],
		"starts_food": [],
		"tx0": [],
		"ta": [],
		"tx1": [],
		"rx0": [],
		"ra": [],
		"rs": [],
		"rc": [],
	}


def _stack(xs, shape, dtype):
	if xs:
		return torch.stack(xs)
	shape = tuple(shape)
	return torch.empty(shape, dtype=dtype)


def _finalize(part, size, depth):
	out = {}
	out["starts_snake"] = _stack(part["starts_snake"], (0, 2, 2), torch.long)
	out["starts_food"] = _stack(part["starts_food"], (0, 2), torch.long)
	out["tx0"] = _stack(part["tx0"], (0, 3, size, size), torch.uint8)
	out["ta"] = _stack(part["ta"], (0,), torch.long)
	out["tx1"] = _stack(part["tx1"], (0, 3, size, size), torch.uint8)
	out["rx0"] = _stack(part["rx0"], (0, 3, size, size), torch.uint8)
	out["ra"] = _stack(part["ra"], (0, depth), torch.long)
	out["rs"] = _stack(part["rs"], (0,), torch.float32)
	out["rc"] = _stack(part["rc"], (0, depth), torch.float32)
	return out


def _branch_rollout(sim, policy, depth, first):
	acts = [u.act_id(first)]
	cons = [float(sim.last_consumed)]
	survival = 1.0
	if not sim.state.alive and not sim.state.won:
		survival = 0.0
	while len(acts) < depth:
		if not sim.state.alive:
			acts.append(0)
			cons.append(0.0)
			continue
		act = policy.action(sim.state)
		sim.step(act)
		acts.append(u.act_id(act))
		cons.append(float(sim.last_consumed))
		if not sim.state.alive and not sim.state.won:
			survival = 0.0
	acts = torch.tensor(acts, dtype=torch.long)
	survival = torch.tensor(survival, dtype=torch.float32)
	cons = torch.tensor(cons, dtype=torch.float32)
	return acts, survival, cons


def preprocess(cfg):
	u.seed_all(cfg["seed"])
	_, run = _paths(cfg)
	size = cfg["L"] * cfg["U"]
	depth = cfg["D"]
	counts = u.split_counts(cfg["dataset_size"], cfg["split"])
	names = ["train"] * counts[0] + ["val"] * counts[1] + ["test"] * counts[2]
	parts = {"train": _blank(), "val": _blank(), "test": _blank()}
	rng = random.Random(cfg["seed"])
	bar = tqdm.auto.tqdm(range(cfg["dataset_size"]), desc="preprocess")
	for i in bar:
		split = names[i]
		sim_seed = rng.randrange(1 << 63)
		pol_seed = rng.randrange(1 << 63)
		sim = env.Simulator(cfg["L"], cfg["U"], random.Random(sim_seed))
		pol = policies.RandomBackbonePerturbedHamiltonianPolicy(cfg["L"], random.Random(pol_seed))
		state = sim.reset()
		part = parts[split]
		part["starts_snake"].append(torch.tensor(state.snake, dtype=torch.long))
		part["starts_food"].append(torch.tensor(state.food, dtype=torch.long))
		while sim.state.alive:
			x0 = u.image_u8(sim.display())
			pol_act = pol.action(sim.state)
			pol_state = pol.rng.getstate()
			for act in env.ACTIONS:
				snap = sim.snapshot()
				pol.rng.setstate(pol_state)
				sim.step(act)
				x1 = u.image_u8(sim.display())
				part["tx0"].append(x0)
				part["ta"].append(torch.tensor(u.act_id(act), dtype=torch.long))
				part["tx1"].append(x1)
				acts, survival, cons = _branch_rollout(sim, pol, depth, act)
				part["rx0"].append(x0)
				part["ra"].append(acts)
				part["rs"].append(survival)
				part["rc"].append(cons)
				sim.restore(snap)
			pol.rng.setstate(pol_state)
			sim.step(pol_act)
			bar.set_postfix(split=split, steps=sim.state.time)
	for name in ("train", "val", "test"):
		out = _finalize(parts[name], size, depth)
		torch.save(out, run / (name + ".pt"))
	meta = {"counts": counts, "depth": depth, "size": size}
	torch.save(meta, run / "meta.pt")
	return meta


def _transition_loader(data, batch_size, pin_memory, shuffle):
	if data["tx0"].shape[0] == 0:
		return None
	ds = torch.utils.data.TensorDataset(data["tx0"], data["ta"], data["tx1"])
	cls = torch.utils.data.DataLoader
	return cls(ds, batch_size=batch_size, pin_memory=pin_memory, shuffle=shuffle)


def _rollout_loader(data, batch_size, pin_memory, shuffle):
	if data["rx0"].shape[0] == 0:
		return None
	ds = torch.utils.data.TensorDataset(data["rx0"], data["ra"], data["rs"], data["rc"])
	cls = torch.utils.data.DataLoader
	return cls(ds, batch_size=batch_size, pin_memory=pin_memory, shuffle=shuffle)


def _transition_pass(model, loader, opt, dev, cfg, train):
	if loader is None:
		return {"loss": 0.0, "wm": 0.0, "enc": 0.0}
	total = {"loss": 0.0, "wm": 0.0, "enc": 0.0}
	n = 0
	model.train(train)
	bar = tqdm.auto.tqdm(loader, leave=False, disable=not train, desc="transition")
	for x0, a, x1 in bar:
		x0 = u.image_f32(x0.to(dev, non_blocking=True))
		x1 = u.image_f32(x1.to(dev, non_blocking=True))
		a = a.to(dev, non_blocking=True)
		with torch.set_grad_enabled(train):
			h0 = model.enc(x0)
			h1 = model.enc(x1)
			pred = model.wm(h0, a)
			wm = F.mse_loss(pred, h1.detach())
			enc = 0.5 * (u.sigreg(h0) + u.sigreg(h1))
			loss = cfg["wm_w"] * wm + cfg["enc_w"] * enc
			if train:
				opt.zero_grad()
				loss.backward()
				opt.step()
		b = x0.shape[0]
		n += b
		total["loss"] += loss.item() * b
		total["wm"] += wm.item() * b
		total["enc"] += enc.item() * b
	for key in total:
		total[key] /= max(1, n)
	return total


def _rollout_pass(model, loader, opt, dev, cfg, train):
	if loader is None:
		return {"loss": 0.0, "survival": 0.0, "consumption": 0.0}
	total = {"loss": 0.0, "survival": 0.0, "consumption": 0.0}
	n = 0
	model.train(train)
	bar = tqdm.auto.tqdm(loader, leave=False, disable=not train, desc="rollout")
	for x0, acts, survival, cons in bar:
		x0 = u.image_f32(x0.to(dev, non_blocking=True))
		acts = acts.to(dev, non_blocking=True)
		survival = survival.to(dev, non_blocking=True)
		cons = cons.to(dev, non_blocking=True)
		with torch.set_grad_enabled(train):
			_, pred_survival, pred_cons = model.rollout(x0, acts)
			s_loss = F.binary_cross_entropy(pred_survival, survival)
			c_loss = F.binary_cross_entropy(pred_cons, cons)
			loss = cfg["eval_w"] * (s_loss + c_loss)
			if train:
				opt.zero_grad()
				loss.backward()
				opt.step()
		b = x0.shape[0]
		n += b
		total["loss"] += loss.item() * b
		total["survival"] += s_loss.item() * b
		total["consumption"] += c_loss.item() * b
	for key in total:
		total[key] /= max(1, n)
	return total


def _save(path, model, opt, cfg, hist, epoch):
	state = {
		"model": model.state_dict(),
		"opt": opt.state_dict(),
		"cfg": cfg,
		"hist": hist,
		"epoch": epoch,
	}
	torch.save(state, path)


def train(cfg):
	u.seed_all(cfg["seed"])
	_, run = _paths(cfg)
	train_data = torch.load(run / "train.pt", map_location="cpu")
	val_data = torch.load(run / "val.pt", map_location="cpu")
	dev = u.device()
	pin = dev.type == "cuda"
	size = cfg["L"] * cfg["U"]
	model = models.System(size, cfg["H"]).to(dev)
	opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
	train_t = _transition_loader(train_data, cfg["batch_size"], pin, True)
	train_r = _rollout_loader(train_data, cfg["batch_size"], pin, True)
	val_t = _transition_loader(val_data, cfg["batch_size"], pin, False)
	val_r = _rollout_loader(val_data, cfg["batch_size"], pin, False)
	hist = []
	best = None
	wait = 0
	has_val = val_data["rx0"].shape[0] > 0 or val_data["tx0"].shape[0] > 0
	patience = cfg["epochs"] // 10 if has_val else None
	for epoch in range(1, cfg["epochs"] + 1):
		u.say("epoch " + str(epoch))
		train_tr = _transition_pass(model, train_t, opt, dev, cfg, True)
		train_ro = _rollout_pass(model, train_r, opt, dev, cfg, True)
		with torch.no_grad():
			val_tr = _transition_pass(model, val_t, opt, dev, cfg, False)
			val_ro = _rollout_pass(model, val_r, opt, dev, cfg, False)
		row = {
			"epoch": epoch,
			"train_transition": train_tr["loss"],
			"train_rollout": train_ro["loss"],
			"val_transition": val_tr["loss"],
			"val_rollout": val_ro["loss"],
		}
		row["train_loss"] = row["train_transition"] + row["train_rollout"]
		row["val_loss"] = row["val_transition"] + row["val_rollout"]
		hist.append(row)
		_save(run / "latest.pt", model, opt, cfg, hist, epoch)
		score = row["val_loss"] if has_val else row["train_loss"]
		if best is None or score < best:
			best = score
			wait = 0
			_save(run / "best.pt", model, opt, cfg, hist, epoch)
		else:
			wait += 1
			if has_val and wait > patience:
				break
	torch.save(hist, run / "history.pt")
	return hist


def load_model(cfg, name="best.pt"):
	_, run = _paths(cfg)
	dev = u.device()
	size = cfg["L"] * cfg["U"]
	model = models.System(size, cfg["H"]).to(dev)
	state = torch.load(run / name, map_location=dev)
	model.load_state_dict(state["model"])
	model.eval()
	return model, dev


def _candidate(sim, depth, first, rng):
	snap = sim.snapshot()
	acts = []
	sim.step(first)
	acts.append(u.act_id(first))
	while len(acts) < depth:
		if not sim.state.alive:
			acts.append(0)
			continue
		act = rng.choice(env.ACTIONS)
		sim.step(act)
		acts.append(u.act_id(act))
	sim.restore(snap)
	return torch.tensor(acts, dtype=torch.long)


def _planner_batch_size(dev):
	if dev.type == "cuda":
		return 256
	if dev.type == "mps":
		return 128
	return 64


def _score_actions(model, dev, x0, acts):
	x0 = u.image_f32(x0.unsqueeze(0).to(dev, non_blocking=True))
	with torch.no_grad():
		h0 = model.enc(x0)
		batch = _planner_batch_size(dev)
		scores = []
		for lo in range(0, acts.shape[0], batch):
			hi = min(lo + batch, acts.shape[0])
			chunk = acts[lo:hi].to(dev, non_blocking=True)
			h = h0.expand(chunk.shape[0], -1)
			_, survival, cons = model.rollout_h(h, chunk)
			score = survival + cons.sum(1)
			scores.append(score.cpu())
	return torch.cat(scores)


def plan_action(model, sim, cfg, rng, dev):
	x0 = u.image_u8(sim.display())
	total = max(len(env.ACTIONS), cfg["planner_samples"])
	acts = []
	firsts = []
	for i in range(total):
		first = env.ACTIONS[i] if i < len(env.ACTIONS) else rng.choice(env.ACTIONS)
		firsts.append(first)
		acts.append(_candidate(sim, cfg["D"], first, rng))
	acts = torch.stack(acts)
	scores = _score_actions(model, dev, x0, acts)
	best_i = int(scores.argmax().item())
	return firsts[best_i], float(scores[best_i].item()), total


def test(cfg):
	u.seed_all(cfg["seed"])
	_, run = _paths(cfg)
	data = torch.load(run / "test.pt", map_location="cpu")
	model, dev = load_model(cfg)
	n = data["starts_snake"].shape[0]
	limit = n if cfg["test_limit"] < 1 else min(n, cfg["test_limit"])
	rng = random.Random(cfg["seed"] + 1)
	scores = []
	msg = "test device=" + str(dev) + " episodes=" + str(limit)
	msg += " planner_samples=" + str(cfg["planner_samples"])
	u.say(msg)
	bar = tqdm.auto.tqdm(range(limit), desc="test")
	for i in bar:
		sim_seed = rng.randrange(1 << 63)
		sim = env.Simulator(cfg["L"], cfg["U"], random.Random(sim_seed))
		snake = tuple(tuple(int(x) for x in seg.tolist()) for seg in data["starts_snake"][i])
		food = tuple(int(x) for x in data["starts_food"][i].tolist())
		sim.reset(snake, food)
		cap = 4 * cfg["L"] * cfg["L"] * max(1, cfg["L"] * cfg["L"] - 2)
		steps = 0
		t_ep = time.perf_counter()
		msg = "episode " + str(i + 1) + "/" + str(limit)
		msg += " start food=" + str(food) + " cap=" + str(cap)
		u.say(msg)
		while sim.state.alive and steps < cap:
			t0 = time.perf_counter()
			act, best, count = plan_action(model, sim, cfg, rng, dev)
			t1 = time.perf_counter()
			sim.step(act)
			steps += 1
			if steps <= 3 or steps % 25 == 0:
				msg = "episode " + str(i + 1) + " step " + str(steps)
				msg += " action=" + str(act) + " score=" + str(round(best, 4))
				msg += " plans=" + str(count) + " plan_s=" + str(round(t1 - t0, 3))
				msg += " len=" + str(len(sim.state.snake)) + " alive=" + str(sim.state.alive)
				msg += " won=" + str(sim.state.won)
				u.say(msg)
		score = u.completion(sim.state, cfg["L"])
		scores.append(score)
		t_done = round(time.perf_counter() - t_ep, 3)
		msg = "episode " + str(i + 1) + " done steps=" + str(steps)
		msg += " completion=" + str(round(score, 4)) + " time_s=" + str(t_done)
		u.say(msg)
		bar.set_postfix(score=sum(scores) / len(scores))
	mean = 0.0 if not scores else sum(scores) / len(scores)
	out = {"mean_completion": mean, "n": len(scores), "scores": scores}
	torch.save(out, run / "test_results.pt")
	return out


def preview(cfg, steps=8):
	rng = random.Random(cfg["seed"])
	sim = env.Simulator(cfg["L"], cfg["U"], random.Random(rng.randrange(1 << 63)))
	pol_rng = random.Random(rng.randrange(1 << 63))
	pol = policies.RandomBackbonePerturbedHamiltonianPolicy(cfg["L"], pol_rng)
	sim.reset()
	frames = [sim.display().clone()]
	while sim.state.alive and len(frames) < steps:
		act = pol.action(sim.state)
		sim.step(act)
		frames.append(sim.display().clone())
	return frames
