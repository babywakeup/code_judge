"""Support ``python -m periodic_template_matcher``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import argparse
import json

from ._internal.workflow import run_config


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Periodic template registration and segmentation transfer')
    parser.add_argument('--config', required=True, help='UTF-8 JSON config; paths relative to that JSON')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    result = run_config(args.config, args.out)
    summary = {key: result[key] for key in ('best_template_index', 'accepted', 'status', 'template_score_gap')}
    print(json.dumps(summary, ensure_ascii=False))
    for match in result['matches']:
        print(f"{match['template_name']}: score={match['match_score']:.2f}, "
              f"geometry={match['geometry_accepted']}, periodic={match['periodic_detected']}")
    print(f"Saved result: {Path(args.out).resolve() / 'result.json'}")
    print(f'Viewer: python -m periodic_template_matcher.app --result "{Path(args.out).resolve()}"')


if __name__ == '__main__':
    main()
