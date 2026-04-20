import src.pipeline as pipe


def main():
	cfg = pipe.default_cfg()
	out = pipe.test(cfg)
	print(out, flush=True)


if __name__ == "__main__":
	main()
