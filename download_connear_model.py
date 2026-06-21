"""Download the pretrained CoNNear PyTorch weights."""

from argparse import ArgumentParser

from utils.connear import CONNEAR_WEIGHTS_URL, ensure_connear_weights


def main() -> None:
    parser = ArgumentParser("Download pretrained CoNNear weights.")
    parser.add_argument(
        "--out",
        type=str,
        default="./data/connear/Gmodel.pt",
        help="Destination path for Gmodel.pt.",
    )
    args = parser.parse_args()

    path = ensure_connear_weights(args.out, auto_download=True)
    print(f"Downloaded CoNNear weights from {CONNEAR_WEIGHTS_URL}")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
