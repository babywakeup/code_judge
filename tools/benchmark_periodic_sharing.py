"""Serial AB/BA comparison of independent copies versus a shared x/y lattice."""
import argparse
from pathlib import Path

try:
    from .benchmark_ecc_roi import ROOT, benchmark
except ImportError:
    from benchmark_ecc_roi import ROOT, benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'periodic_template_matcher/config.sd1.labelme.opencv.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=Path, action='append', default=[])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    benchmark(args.config, args.output, args.target, args.repeats, option='shared_periods')


if __name__ == '__main__':
    main()