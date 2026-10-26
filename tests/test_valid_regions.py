"""Valid-height policy must agree across matching, completion, evaluation and IO."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher, run_config
from periodic_template_matcher.utils import read_image, write_image
from tests.helpers import texture


class ValidRegionTests(unittest.TestCase):
    def test_height_overrides_fraction_and_intersects_mask_without_mutation(self):
        from periodic_template_matcher.utils import ValidRegion
        image = np.zeros((160, 200, 3), np.uint8)
        mask = np.ones((160, 200), np.uint8)
        mask[:10, :20] = 0
        original = mask.copy()
        region = ValidRegion(valid_height=150, bottom_ignore_fraction=.2)
        actual = region.build_mask(image, mask)
        self.assertEqual(actual.dtype, np.uint8)
        self.assertTrue(actual[149, 199])
        self.assertFalse(actual[150:].any())
        self.assertFalse(actual[:10, :20].any())
        np.testing.assert_array_equal(mask, original)
        self.assertEqual(region.crop(image).shape, (150, 200, 3))
        self.assertEqual(ValidRegion(bottom_ignore_fraction=.1).resolved_height(image), 144)

    def test_invalid_heights_fail_clearly(self):
        from periodic_template_matcher.utils import ValidRegion
        image = np.zeros((160, 200), np.uint8)
        for height in (0, -1, 161, 10.5, True, '150'):
            with self.subTest(height=height), self.assertRaises(ValueError):
                ValidRegion(valid_height=height).build_mask(image)
        with self.assertRaises(ValueError):
            ValidRegion(valid_height=4).build_mask(image)
        for height in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                MatcherConfig(valid_height=height).validate()

    def test_match_resolves_independent_heights_before_registration(self):
        image = texture()
        cfg = MatcherConfig(valid_height=140, bottom_ignore_fraction=.2)
        from periodic_template_matcher.matcher import select_templates as real_select
        with patch('periodic_template_matcher.matcher.select_templates', wraps=real_select) as select:
            result = PeriodicTemplateMatcher(cfg).match(
                [image], [np.ones(image.shape[:2], np.uint16)], image,
                template_valid_heights=[150], target_valid_height=145)
        self.assertTrue(result.accepted)
        self.assertTrue(result.label_valid[140:145, 20:180].all())
        self.assertFalse(result.label_valid[145:].any())
        self.assertEqual(select.call_args.args[2].bottom_ignore_fraction, 0)
        self.assertIsNone(select.call_args.args[2].valid_height)
        self.assertTrue(select.call_args.args[3][0][149].all())
        self.assertFalse(select.call_args.args[3][0][150:].any())
        self.assertEqual(result.registration['valid_regions']['target']['valid_height'], 145)
        self.assertEqual(result.registration['config']['valid_height'], 140)

    def test_footer_does_not_change_matching_for_either_backend(self):
        image = texture()
        changed = image.copy()
        changed[130:] = np.random.default_rng(5).integers(0, 256, changed[130:].shape, dtype=np.uint8)
        for backend in ('fft', 'opencv'):
            with self.subTest(backend=backend):
                matcher = PeriodicTemplateMatcher(MatcherConfig(backend=backend, valid_height=130,
                    max_side=200, angles=(0.,), scales=(1.,), refine_candidates=2, ecc_iterations=20))
                first = matcher.match([image], [None], image)
                second = matcher.match([changed], [None], image)
                self.assertTrue(second.accepted)
                self.assertAlmostEqual(first.match_score, second.match_score, places=5)
                np.testing.assert_allclose(first.matrix_template_to_target, second.matrix_template_to_target, atol=1e-5)

    def test_excluded_footer_cannot_leak_through_downsampling(self):
        rng = np.random.default_rng(2)
        image = cv2.GaussianBlur(rng.integers(20, 230, (500, 600, 3), dtype=np.uint8), (0, 0), 2)
        changed = image.copy()
        changed[391:] = 255
        for backend in ('fft', 'opencv'):
            matcher = PeriodicTemplateMatcher(MatcherConfig(valid_height=391, max_side=200,
                backend=backend, angles=(0.,), scales=(1.,), refine_candidates=1, ecc_iterations=10))
            first = matcher.match([image], [None], image)
            second = matcher.match([changed], [None], image)
            with self.subTest(backend=backend):
                self.assertAlmostEqual(first.match_score, second.match_score, places=5)
                np.testing.assert_allclose(first.matrix_template_to_target, second.matrix_template_to_target, atol=1e-5)

    def test_evaluation_respects_height_and_target_mask(self):
        image = texture()
        labels = np.ones(image.shape[:2], np.uint16)
        mask = np.ones(image.shape[:2], np.uint8)
        mask[:10] = 0
        result = PeriodicTemplateMatcher(MatcherConfig(max_side=200, angles=(0.,), scales=(1.,))).evaluate(
            [image], [labels], image, labels, template_valid_heights=[150],
            target_valid_height=130, target_valid=mask)
        self.assertEqual(result.metrics['evaluated_pixels'], 120*200)
        self.assertFalse(result.ground_truth_valid[130:].any())
        self.assertFalse(result.match.label_valid[:10].any())

    def test_json_height_exports_and_duplicate_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = texture()
            write_image(root/'image.png', image)
            write_image(root/'labels.png', np.ones(image.shape[:2], np.uint16))
            spec = dict(target='image.png', target_valid_height=130,
                        templates=[dict(image='image.png', labels='labels.png', valid_height=150)],
                        options=dict(valid_height=140, bottom_ignore_fraction=.2, max_side=200,
                                     angles=[0], scales=[1], refine_candidates=2, ecc_iterations=20))
            path = root/'config.json'
            path.write_text(json.dumps(spec), encoding='utf-8')
            result = run_config(path, root/'out')
            mask = read_image(root/'out/label_valid.png', 0)
            self.assertEqual(mask.shape, image.shape[:2])
            self.assertTrue(mask[125:130, 20:180].all())
            self.assertFalse(mask[130:].any())
            self.assertEqual(result['valid_regions']['templates'][0]['valid_height'], 150)
            spec['templates'].append(dict(spec['templates'][0], valid_height=149))
            path.write_text(json.dumps(spec), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                run_config(path, root/'out')


if __name__ == '__main__':
    unittest.main()
