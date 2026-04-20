import src.pipeline as pipe


def main():
	cfg = pipe.default_cfg()
	meta = pipe.preprocess(cfg)
	print(meta, flush=True)


if __name__ == "__main__":
	main()
