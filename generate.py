import argparse

def read_options():
    p = argparse.ArgumentParser()
    p.add_argument("-n","--number-of-gen", dest="n", type=int, default=1)
    p.add_argument("-c","--config", dest="config", type=str, default=None)
    p.add_argument("-o","--output", dest="output", type=str, default="out_gen")
    p.add_argument("--seed", dest="seed", type=int, default=None)
    return p.parse_args()

def main():
    args = read_options()
    generate(args.n, args.output, args.config, seed=args.seed)
if __name__ == "__main__":
    main()
