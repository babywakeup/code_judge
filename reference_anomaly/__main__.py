"""Run with python -m reference_anomaly --result <matcher result directory>."""
import argparse
import json

from .core import AnomalyConfig
from .io import run_result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare input and corrected normal reference for general anomaly evidence')
    parser.add_argument('--result', required=True, help='Matcher result directory or result.json')
    parser.add_argument('--out', help='Output directory; defaults to <result>/anomaly')
    parser.add_argument('--tolerance-radius', type=int, default=2, help='Allowed displacement in original pixels (0..5; default 2)')
    parser.add_argument('--high-threshold', type=float, default=1.8, help='Candidate seed score (default 1.8)')
    parser.add_argument('--low-threshold', type=float, default=1., help='Candidate connected-growth score (default 1.0)')
    parser.add_argument('--structure-weight', type=float, default=1.5, help='Structure response multiplier (default 1.5)')
    parser.add_argument('--normal-config', help='Optional matcher config: fit a baseline using its normal templates only')
    args = parser.parse_args(argv)
    try:
        cfg = AnomalyConfig(tolerance_radius=args.tolerance_radius,
                            high_threshold=args.high_threshold, low_threshold=args.low_threshold,
                            structure_weight=args.structure_weight)
        cfg.validate()
        manifest = run_result(args.result, args.out, config=cfg, normal_config=args.normal_config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    metadata = json.loads(manifest.read_text(encoding='utf-8'))
    print(json.dumps({key: metadata[key] for key in ('status', 'valid_pixels', 'candidate_pixels', 'priority_pixels', 'salience_pixels', 'region_count')}))
    print(f'Saved anomaly evidence: {manifest}')
    print(f'Overview: {manifest.parent / "anomaly_overview.jpg"}')
    print(f'Method comparison: {manifest.parent / "anomaly_methods_overview.jpg"}')
    print(f'Normal calibration: {metadata["normal_calibration"]["status"]}')


if __name__ == '__main__':
    main()
