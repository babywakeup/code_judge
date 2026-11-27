"""Training entry point: ``python -m segment.demo`` (or add ``--smoke``)."""

from .training.config import parse_args
from .training.runner import train


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        from .tools.smoke import run_smoke

        return run_smoke(args)
    return train(args)


if __name__ == "__main__":
    main()
