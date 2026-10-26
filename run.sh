#!/usr/bin/env bash
set -e

python -m periodic_template_matcher --config periodic_template_matcher/config.sd1.labelme.opencv.json --out periodic_template_matcher/results/sd1_opencv
python -m periodic_template_matcher.app --result periodic_template_matcher/results/sd1_opencv
