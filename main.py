import argparse

import batch_convert


def main():
    parser = argparse.ArgumentParser(description="FreeVC batch ATC anonymization")
    parser.add_argument(
        "--model", choices=batch_convert.MODELS.keys(), default="freevc", help="model to use"
    )
    parser.add_argument("--n", type=int, default=1000, help="checkpoint every n clips")
    parser.add_argument("--out-root", default="../../Data/freevc", help="output directory")
    parser.add_argument("--device", default="cpu", help="device to use e.g. cpu or cuda")
    args = parser.parse_args()

    run_batch = batch_convert.MODELS[args.model]
    run_batch(out_root=args.out_root, n=args.n, device=args.device)


if __name__ == "__main__":
    main()
