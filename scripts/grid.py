import itertools
import pathlib
import sys
import tomllib
import tqdm.auto


def _here():
	return pathlib.Path(__file__).resolve().parent


def _root():
	return _here().parent


sys.path.insert(0, str(_root()))
import src.pipeline as pipe
import src.utils as u


def _grid(path):
	out = {}
	for raw in path.read_text().splitlines():
		line = raw.strip()
		if not line:
			continue
		sep = ":" if ":" in line else "="
		key, vals = line.split(sep, 1)
		key = key.strip()
		vals = vals.strip()
		text = "x = " + vals
		if "," in vals:
			text = "x = [" + vals + "]"
		out[key] = tomllib.loads(text)["x"]
		if not isinstance(out[key], list):
			out[key] = [out[key]]
	return out


def _jobs(grid):
	keys = list(grid)
	total = 1
	for key in keys:
		total *= len(grid[key])
	sets = itertools.product(*(grid[key] for key in keys))
	return keys, total, sets


def _text(items):
	return ", ".join(key + "=" + str(val) for key, val in items.items())


def _best_row(hist):
	if not hist:
		return None
	return min(hist, key=lambda row: row["val_loss"])


def _better(cur, best):
	if best is None:
		return True
	if cur["score"] > best["score"]:
		return True
	return cur["score"] == best["score"] and cur["val_loss"] < best["val_loss"]


def _line(path, msg):
	text = "[" + u.stamp() + "] " + msg
	tqdm.auto.tqdm.write(text)
	with path.open("a") as f:
		f.write(text + "\n")


def main():
	root = _root()
	here = _here()
	base = u.load_cfg(root / "config.toml")
	grid = _grid(here / "grid.txt")
	keys, total, sets = _jobs(grid)
	best = None
	log = here / "grid.logs"
	_line(log, "grid start total=" + str(total))
	bar = tqdm.auto.tqdm(sets, total=total, desc="grid")
	for i, vals in enumerate(bar, 1):
		cfg = dict(base)
		cfg.update(dict(zip(keys, vals)))
		cfg["D"] = cfg["L"] * cfg["L"] - 1
		name = str(cfg["run_name"]) + "_grid_" + str(i).zfill(len(str(total)))
		cfg["run_name"] = name
		pipe.preprocess(cfg)
		hist = pipe.train(cfg)
		out = pipe.test(cfg)
		row = _best_row(hist)
		cur = {
			"score": out["mean_completion"],
			"val_loss": row["val_loss"],
			"epoch": row["epoch"],
			"run_name": name,
			"cfg": dict(zip(keys, vals)),
		}
		msg = "run=" + name + " score=" + f"{cur['score']:.6f}"
		msg += " val_loss=" + f"{cur['val_loss']:.6f}" + " epoch=" + str(cur["epoch"])
		msg += " cfg={" + _text(cur["cfg"]) + "}"
		_line(log, msg)
		if _better(cur, best):
			best = cur
			msg = "best run=" + best["run_name"] + " score=" + f"{best['score']:.6f}"
			msg += " val_loss=" + f"{best['val_loss']:.6f}" + " epoch=" + str(best["epoch"])
			msg += " cfg={" + _text(best["cfg"]) + "}"
			_line(log, msg)
		bar.set_postfix(best=0.0 if best is None else round(best["score"], 4))
	msg = "final best run=" + best["run_name"] + " score=" + f"{best['score']:.6f}"
	msg += " val_loss=" + f"{best['val_loss']:.6f}" + " epoch=" + str(best["epoch"])
	msg += " cfg={" + _text(best["cfg"]) + "}"
	_line(log, msg)


if __name__ == "__main__":
	main()
