"""Runnable synthetic class-API example; NOT real production annotations."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from periodic_template_matcher.utils import warp_labels, write_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/api_demo')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    template = np.full((240, 320, 3), 60, np.uint8)
    for _ in range(100):
        center = tuple(int(x) for x in rng.integers([10, 10], [310, 230]))
        color = tuple(int(x) for x in rng.integers(80, 245, 3))
        cv2.circle(template, center, int(rng.integers(3, 12)), color, -1)
    labels = np.zeros(template.shape[:2], np.uint16)
    labels[40:90, 60:130] = 1
    labels[140:190, 200:260] = 2
    H = np.vstack([cv2.getRotationMatrix2D((160, 120), .8, 1.01), [0, 0, 1.]])
    H[:2, 2] += [7, -5]
    target = cv2.warpPerspective(template, H, (320, 240))
    truth, _ = warp_labels(labels, H, target.shape)

    config = MatcherConfig(max_side=320, angles=(0.,), scales=(1.,), refine_candidates=3)
    matcher = PeriodicTemplateMatcher(config)
    # Binary list entry k corresponds to class k+1. A class-ID truth array also works.
    evaluation = matcher.evaluate([template], [labels], target, [truth == 1, truth == 2])
    matcher.visualize(evaluation, out / 'evaluation.jpg')
    write_image(out / 'synthetic_template.png', template)
    write_image(out / 'synthetic_input.png', target)
    write_image(out / 'synthetic_truth.png', truth)
    if evaluation.match.mapped_labels is not None:
        write_image(out / 'predicted_labels.png', evaluation.match.mapped_labels)
        write_image(out / 'label_valid.png', evaluation.match.label_valid.astype(np.uint8) * 255)
    summary = dict(note='SYNTHETIC DEMO ONLY / NOT PRODUCTION ACCURACY',
                   registration=evaluation.match.registration, evaluation=evaluation.metrics)
    (out / 'evaluation.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(match_score=evaluation.match.match_score,
                          segmentation_score=evaluation.score, output=str(out.resolve())), indent=2))


if __name__ == '__main__':
    main()