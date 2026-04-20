import src.pipeline as pipe


def main():
	cfg = pipe.default_cfg()
	hist = pipe.train(cfg)
	if not hist:
		print([], flush=True)
		return
	print(hist[-1], flush=True)


if __name__ == "__main__":
	main()
