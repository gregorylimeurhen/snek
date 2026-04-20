import pathlib
import random
import time
import torch
import src.environment as env


def seed_all(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def device():
	if torch.cuda.is_available():
		return torch.device("cuda")
	mps = getattr(torch.backends, "mps", None)
	if mps is not None and mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def ensure_dir(path):
	path = pathlib.Path(path)
	path.mkdir(parents=True, exist_ok=True)
	return path


def split_counts(n, ratios):
	a = int(n * ratios[0])
	b = int(n * ratios[1])
	c = int(n) - a - b
	return a, b, c


def image_u8(img):
	return img.permute(2, 0, 1).contiguous()


def image_f32(img):
	return img.float() / 255.0


def act_id(act):
	return env.AI[act]


def action(i):
	return env.ACTIONS[int(i)]


def sigreg(z, m=32, k=17):
	if z.shape[0] < 2:
		return z.new_zeros(())
	ds = torch.randn((m, z.shape[1]), device=z.device)
	ds = ds / ds.norm(dim=1, keepdim=True).clamp_min(1e-12)
	u = z @ ds.t()
	t = torch.linspace(-3, 3, k, device=z.device)
	w = torch.exp(-(t * t))
	g = torch.exp(-0.5 * t * t)
	a = u.t().unsqueeze(-1) * t
	r = torch.cos(a).mean(1)
	i = torch.sin(a).mean(1)
	v = (r - g).square() + i.square()
	return z.shape[0] * (v * w).mean()


def stamp():
	return time.strftime("%Y-%m-%d %H:%M:%S")


def say(msg):
	print("[" + stamp() + "] " + str(msg), flush=True)


def completion(state, extent):
	max_food = extent * extent - 2
	if max_food < 1:
		return 1.0
	return max(0.0, min(1.0, (len(state.snake) - 2) / max_food))
