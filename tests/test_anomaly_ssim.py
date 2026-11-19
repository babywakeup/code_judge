"""Multi-window SSIM and normal-only calibration regressions."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import cv2
import numpy as np

from reference_anomaly.ssim import multiscale_ssim
from reference_anomaly import analyze_pair
from tests.test_reference_anomaly import pattern
from reference_anomaly.normal_baseline import fit_normal_baseline, build_normal_baseline
from periodic_template_matcher.utils import write_image
from reference_anomaly.io import save_analysis


class SsimTests(unittest.TestCase):
    def test_source_change_at_window_corner_is_not_comparable(self):
        image = np.full((32, 32), 100, np.uint8)
        ids = np.ones(image.shape, np.uint16)
        ids[12, 12] = 2
        _, known, _ = multiscale_ssim(image, image, np.ones(image.shape, bool), ids)
        self.assertFalse(known[10, 10])

    def test_large_integer_source_ids_do_not_merge_at_seams(self):
        image = np.full((32, 32), 100, np.uint8)
        ids = np.full(image.shape, 2**53, np.uint64)
        ids[:, 16:] = 2**53+1
        _, known, _ = multiscale_ssim(image, image, np.ones(image.shape, bool), ids)
        self.assertFalse(known[:, 15:17].any())

    def test_identical_and_local_missing_structure(self):
        reference = pattern()
        valid = np.ones(reference.shape[:2], bool)
        raw, known, windows = multiscale_ssim(reference, reference, valid)
        self.assertLess(float(raw.max()), 1e-4)
        image = reference.copy()
        image[35:50, 40:65] = 15
        raw, known, windows = multiscale_ssim(image, reference, valid)
        self.assertGreater(float(raw[35:50, 40:65].mean()), .4)
        self.assertEqual(set(windows), {5, 11, 21})
        self.assertTrue(np.isfinite(raw).all())

    def test_invalid_holes_and_seams_do_not_enter_windows(self):
        reference = pattern()
        image = reference.copy()
        valid = np.ones(reference.shape[:2], bool)
        valid[30:60, 40:65] = False
        reference[~valid] = 0
        ids = np.ones(valid.shape, np.uint16)
        ids[:, 80:] = 2
        raw, known, windows = multiscale_ssim(image, reference, valid, ids)
        self.assertLess(float(raw.max()), 1e-4)
        self.assertFalse(known[:, 79:81].any())
        self.assertFalse(known[~valid].any())

    def test_small_window_preserves_small_defect(self):
        reference = np.full((64, 64), 100, np.uint8)
        image = reference.copy()
        image[32, 32] = 200
        raw, known, windows = multiscale_ssim(image, reference, np.ones(image.shape, bool))
        self.assertGreater(float(windows[5][32, 32]), float(windows[21][32, 32]))
        self.assertAlmostEqual(float(raw[32, 32]), max(float(x[32, 32]) for x in windows.values()))

    def test_array_integration(self):
        image = pattern()
        result = analyze_pair(image, image, np.ones(image.shape[:2], bool))
        self.assertEqual(result.ssim_score.dtype, np.float32)
        self.assertLess(float(result.ssim_score.max()), .001)
        self.assertEqual(result.metadata['ssim']['windows'], [5, 11, 21])
        self.assertFalse(result.supported_salience_score.any())
        changed = image.copy()
        changed[30:50, 45:65] = 10
        result = analyze_pair(changed, image, np.ones(image.shape[:2], bool))
        self.assertTrue((result.supported_salience_score >= result.salience_score).all())
        self.assertTrue((result.supported_salience_score <= 2*result.salience_score+1e-6).all())
        self.assertGreater(float(result.supported_salience_score[35:45, 50:60].mean()),
                           float(result.salience_score[35:45, 50:60].mean()))


class NormalBaselineTests(unittest.TestCase):
    def test_unknown_semantic_pixels_do_not_use_background_baseline(self):
        shape = (64, 64)
        valid = np.ones(shape, bool)
        labels = np.zeros(shape, np.uint16)
        texture = np.zeros(shape, np.float32)
        model = fit_normal_baseline([dict(responses=tuple(texture.copy() for _ in range(4)),
                                         texture=texture, labels=labels, valid=valid)])
        image = np.full((*shape, 3), 100, np.uint8)
        semantic_valid = valid.copy()
        semantic_valid[10:20] = False
        result = analyze_pair(image, image, valid, normal_baseline=model,
                              semantic_labels=labels, semantic_valid=semantic_valid)
        self.assertFalse(result.normal_valid[10:20].any())
        self.assertTrue(result.normal_valid[30, 10])

    def test_builder_reads_normal_templates_without_loading_defect_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = pattern()
            second = np.clip(np.rint(first.astype(float)*.92+5), 0, 255).astype(np.uint8)
            second = cv2.GaussianBlur(second, (3, 3), .3)
            labels = np.zeros(first.shape[:2], np.uint16)
            labels[20:70, 20:70] = 1
            for name, image in (('a.png', first), ('b.png', second), ('labels.png', labels)):
                write_image(root / name, image)
            spec = dict(target='nonexistent_defect_input.png', templates=[
                dict(image='a.png', labels='labels.png', valid_height=90),
                dict(image='b.png', labels='labels.png', valid_height=90)],
                options=dict(backend='opencv', max_side=128, angles=[0], scales=[1],
                             ecc_iterations=15, refine_candidates=2, min_score=50))
            path = root / 'normal.json'
            path.write_text(json.dumps(spec), encoding='utf-8')
            model = build_normal_baseline(path)
            self.assertEqual(model.metadata['status'], 'available')
            self.assertTrue(all(p['accepted'] for p in model.metadata['normal_pairs']))
            self.assertTrue(model.groups)
            with patch('periodic_template_matcher._internal.registration.core.select_templates', return_value=dict(
                    accepted=False, status='rejected', ranking=[0], matches=[dict(match_score=0., geometry_accepted=False)])):
                rejected = build_normal_baseline(path)
            self.assertFalse(rejected.groups)
            alias = root / 'anomaly_heatmap.png'
            write_image(alias, first)
            spec['templates'].append(dict(image=alias.name, labels='labels.png', valid_height=90))
            path.write_text(json.dumps(spec), encoding='utf-8')
            alias_model = build_normal_baseline(path)
            self.assertIn(alias.resolve(), alias_model.protected_paths)
            before = alias.read_bytes()
            identical = analyze_pair(first, first, np.ones(first.shape[:2], bool))
            with self.assertRaises(ValueError):
                save_analysis(identical, first, first, root, protected_paths=alias_model.protected_paths)
            self.assertEqual(alias.read_bytes(), before)
            spec['templates'].pop()
            path.write_text(json.dumps(spec), encoding='utf-8')
            # An identical file copy is one unique normal image, not independent normal evidence.
            write_image(root / 'b.png', first)
            empty = build_normal_baseline(path)
            self.assertFalse(empty.groups)
            self.assertEqual(empty.metadata['status'], 'insufficient_normal_samples')

    def test_only_normal_training_statistics_score_future_outliers(self):
        valid = np.ones((64, 64), bool)
        labels = np.ones(valid.shape, np.uint16)
        texture = np.zeros(valid.shape, np.float32)
        responses = tuple(np.full(valid.shape, v, np.float32) for v in (8, 3, 6, .1))
        model = fit_normal_baseline([dict(responses=responses, texture=texture, labels=labels, valid=valid)])
        normal, known, _ = model.score(responses, texture, valid, labels, np.ones_like(texture), 1.5)
        self.assertFalse(normal.any())
        altered = tuple(x.copy() for x in responses)
        altered[0][20:30, 20:30] = 40
        score, known, _ = model.score(altered, texture, valid, labels, np.ones_like(texture), 1.5)
        self.assertGreater(float(score[25, 25]), 3)
        self.assertEqual(float(score[10, 10]), 0)
        np.testing.assert_array_equal(responses[0], 8)
        self.assertTrue(known.all())

    def test_missing_class_or_insufficient_samples_are_unknown(self):
        valid = np.ones((64, 64), bool)
        texture = np.zeros(valid.shape, np.float32)
        labels = np.ones(valid.shape, np.uint16)
        responses = tuple(np.ones_like(texture) for _ in range(4))
        model = fit_normal_baseline([dict(responses=responses, texture=texture, labels=labels, valid=valid)])
        score, known, _ = model.score(responses, texture, valid, labels*2, texture+1, 1.5)
        self.assertFalse(known.any())
        self.assertFalse(score.any())
        _, known, _ = model.score(responses, texture+10, valid, labels, texture+1, 1.5)
        self.assertFalse(known.any())
        empty = fit_normal_baseline([])
        score, known, metadata = empty.score(responses, texture, valid, labels, texture+1, 1.5)
        self.assertFalse(known.any())
        self.assertEqual(metadata['status'], 'insufficient_normal_samples')


if __name__ == '__main__':
    unittest.main()
