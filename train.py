import src.pipeline as pipe


def main():
	cfg = pipe.default_cfg()
	pipe.train(cfg)


if __name__ == "__main__":
	main()
