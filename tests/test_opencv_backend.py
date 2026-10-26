"""Alternative coarse search must preserve signed shifts and annotation transfer."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher, run_config
from periodic_template_matcher._internal.registration import _translation_peaks
from periodic_template_matcher.utils import write_image
from tests.helpers import texture


class OpenCVBackendTests(unittest.TestCase):
    def test_configuration(self):
        self.assertEqual(MatcherConfig().backend, 'fft')
        MatcherConfig(backend='opencv').validate()
        with self.assertRaises(ValueError):
            MatcherConfig(backend='unknown').validate()

    def test_coarse_search_matches_fft_with_masks_and_signed_shifts(self):
        from periodic_template_matcher._internal.opencv_registration import translation_peaks
        rng = np.random.default_rng(18)
        source = rng.random((48, 60), dtype=np.float32)
        sm = np.ones_like(source, np.uint8)
        sm[8:16, 14:28] = 0
        source *= sm
        cfg = MatcherConfig(min_overlap=.45)
        for size in ((60, 48), (72, 55)):
            for dx, dy in ((7, -5), (-8, 6), (0, 0)):
                with self.subTest(size=size, shift=(dx, dy)):
                    H = np.float32([[1, 0, dx], [0, 1, dy]])
                    target = cv2.warpAffine(source, H, size)
                    tm = np.ones_like(target, np.uint8)
                    tm[18:25, 18:25] = 0
                    # Invalid target values must be ignored.
                    target[tm == 0] = 100
                    masked_target = target * tm
                    actual = translation_peaks(source, target, sm, tm, cfg)
                    expected = _translation_peaks(source, masked_target, sm, tm, cfg)
                    self.assertEqual(actual[0][1:], (float(dx), float(dy)))
                    self.assertEqual(actual[0][1:], expected[0][1:])
                    self.assertAlmostEqual(actual[0][0], expected[0][0], places=4)

    def test_empty_overlap_and_zero_energy(self):
        from periodic_template_matcher._internal.opencv_registration import translation_peaks
        source = np.zeros((40, 50), np.float32)
        ones = np.ones_like(source, np.uint8)
        self.assertEqual(translation_peaks(source, source, ones, ones, MatcherConfig()), [])
        self.assertEqual(translation_peaks(ones.astype(np.float32), ones.astype(np.float32),
                                          ones * 0, ones, MatcherConfig()), [])

    def test_public_match_transfers_labels(self):
        source = texture()
        labels = np.zeros(source.shape[:2], np.uint16)
        labels[35:70, 40:90] = 1001
        target = cv2.warpAffine(source, np.float32([[1, 0, -8], [0, 1, 6]]), (200, 160))
        cfg = MatcherConfig(backend='opencv', max_side=200, angles=(0.,), scales=(1.,),
                            refine_candidates=3, ecc_iterations=35)
        result = PeriodicTemplateMatcher(cfg).match([source], [labels], target)
        self.assertTrue(result.accepted)
        np.testing.assert_allclose(result.matrix_template_to_target[:2, 2], [-8, 6], atol=1)
        self.assertEqual(result.mapped_labels[50, 50], 1001)
        self.assertFalse(result.label_valid[0, 0])

    def test_json_routes_backend_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = texture()
            labels = np.zeros(image.shape[:2], np.uint16)
            labels[35:70, 40:90] = 1
            write_image(root / 'source.png', image)
            write_image(root / 'labels.png', labels)
            spec = dict(target='source.png', templates=[dict(image='source.png', labels='labels.png')],
                        options=dict(backend='opencv', max_side=200, angles=[0], scales=[1],
                                     refine_candidates=2, ecc_iterations=20))
            path = root / 'config.json'
            path.write_text(json.dumps(spec), encoding='utf-8')
            result = run_config(path, root / 'out')
            self.assertEqual(result['config']['backend'], 'opencv')
            self.assertTrue(result['labels_exported'])
            self.assertTrue((root / 'out' / 'mapped_labels.png').exists())


if __name__ == '__main__':
    unittest.main()
