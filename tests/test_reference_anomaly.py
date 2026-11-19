"""General anomaly evidence, without defect-color or bounding-box priors."""
import unittest
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from reference_anomaly import AnomalyConfig, analyze_pair
from reference_anomaly.io import run_result, save_analysis
from periodic_template_matcher.utils import read_image, write_image


def pattern():
    rng = np.random.default_rng(28)
    gray = cv2.GaussianBlur(rng.integers(40, 210, (96, 128), dtype=np.uint8), (5, 5), 1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class AnomalyCoreTests(unittest.TestCase):
    def test_structure_weight_amplifies_both_branches_without_changing_raw_response(self):
        reference = pattern()
        image = np.roll(reference, 1, axis=1)
        image[35:45, 50:65] = 30
        valid = np.ones(image.shape[:2], bool)
        base = analyze_pair(image, reference, valid, config=AnomalyConfig(structure_weight=1.))
        stronger = analyze_pair(image, reference, valid, config=AnomalyConfig(structure_weight=1.5))
        self.assertTrue((stronger.strict_score >= base.strict_score-1e-6).all())
        self.assertTrue((stronger.tolerant_score >= base.tolerant_score-1e-6).all())
        self.assertTrue((stronger.strict_score > base.strict_score+.01).any())
        self.assertTrue((stronger.tolerant_score > base.tolerant_score+.01).any())
        np.testing.assert_array_equal(stronger.structure_response, base.structure_response)
        scales = stronger.metadata['calibration']
        expected = np.maximum.reduce([stronger.color_response/scales['color']['scale'],
                                      1.5*stronger.structure_response/scales['structure']['scale'],
                                      stronger.multiscale_response/scales['multiscale']['scale']])
        np.testing.assert_allclose(stronger.strict_score, expected, rtol=1e-6)
        self.assertEqual(AnomalyConfig().structure_weight, 1.5)
        for weight in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                AnomalyConfig(structure_weight=weight).validate()

    def test_bright_dark_and_color_defects_are_preserved(self):
        reference = pattern()
        for color in ((245, 245, 245), (10, 10, 10), (170, 70, 130)):
            with self.subTest(color=color):
                image = reference.copy()
                image[30:55, 40:70] = color
                before = image.copy()
                result = analyze_pair(image, reference, np.ones(image.shape[:2], bool))
                self.assertGreater(float(np.median(result.strict_score[35:50, 45:65])), 1)
                self.assertTrue(result.candidate_mask[35:50, 45:65].all())
                self.assertTrue(result.priority_mask[35:50, 45:65].all())
                np.testing.assert_array_equal(image, before)

    def test_identical_and_empty_validity_have_no_candidates(self):
        reference = pattern()
        for valid in (np.ones(reference.shape[:2], bool), np.zeros(reference.shape[:2], bool)):
            result = analyze_pair(reference, reference, valid)
            self.assertFalse(result.candidate_mask.any())
            self.assertTrue(np.isfinite(result.strict_score).all())
            self.assertEqual(float(result.strict_score.max()), 0)

    def test_tolerance_reduces_shift_but_does_not_erase_strict_evidence(self):
        reference = pattern()
        image = np.roll(reference, 1, axis=1)
        result = analyze_pair(image, reference, np.ones(image.shape[:2], bool))
        interior = np.s_[8:-8, 8:-8]
        self.assertGreater(float(result.strict_score[interior].mean()), .2)
        self.assertLess(float(result.tolerant_score[interior].mean()), float(result.strict_score[interior].mean())*.1)
        self.assertTrue(result.candidate_mask.any())
        self.assertTrue((result.tolerant_score <= result.strict_score+1e-6).all())

    def test_broad_change_is_not_subtracted_as_background(self):
        reference = np.full((64, 64, 3), 100, np.uint8)
        image = np.full_like(reference, 135)
        result = analyze_pair(image, reference, np.ones(reference.shape[:2], bool))
        self.assertGreater(float(result.strict_score.mean()), 1)
        self.assertTrue(result.candidate_mask.all())

    def test_invalid_holes_and_seams_are_explicit(self):
        reference = np.full((64, 64, 3), 100, np.uint8)
        image = reference.copy()
        valid = np.ones((64, 64), bool)
        valid[20:40, 20:40] = False
        reference[~valid] = 0
        ids = np.ones((64, 64), np.uint16)
        ids[:, 32:] = 2
        ids[~valid] = 0
        result = analyze_pair(image, reference, valid, source_map=ids)
        self.assertFalse(result.candidate_mask.any())
        self.assertTrue((result.confidence[~valid] == 0).all())
        self.assertLess(int(result.confidence[10, 32]), int(result.confidence[10, 10]))
        self.assertTrue((result.strict_score[~valid] == 0).all())

    def test_tiny_defect_kept_when_tolerance_can_explain_it(self):
        reference = pattern()
        image = reference.copy()
        image[40, 50] = 255
        result = analyze_pair(image, reference, np.ones(image.shape[:2], bool))
        self.assertTrue(result.candidate_mask[40, 50])

    def test_tolerance_does_not_borrow_across_template_seam(self):
        reference = np.full((96, 128, 3), 80, np.uint8)
        reference[:, :64] = 180
        image = reference.copy()
        image[30:60, 64:66] = 180
        ids = np.ones((96, 128), np.uint16)
        ids[:, 64:] = 2
        result = analyze_pair(image, reference, np.ones(ids.shape, bool), source_map=ids)
        self.assertGreater(float(result.tolerant_score[45, 64]), 1.8)
        self.assertTrue(result.candidate_mask[45, 64])
        self.assertFalse(result.priority_mask[45, 64])  # Explicit seam uncertainty.

    def test_structure_response_and_photometric_invalidity(self):
        reference = np.full((96, 128, 3), 80, np.uint8)
        reference[20:70, 20:90] = 180
        image = reference.copy()
        image[40:45, 20:90] = 80
        valid = np.ones((96, 128), bool)
        photo = valid.copy()
        photo[40:45, 70:90] = False
        result = analyze_pair(image, reference, valid, photometric_valid=photo)
        self.assertGreater(float(result.structure_response[39:46, 30:60].max()), 20)
        self.assertTrue(result.candidate_mask[42, 40])
        self.assertFalse(result.candidate_mask[42, 80])
        self.assertTrue(valid.all())  # No mutation of supplied validity.

    def test_seam_seed_cannot_promote_weak_high_confidence_pixels(self):
        reference = np.full((64, 64, 3), 100, np.uint8)
        image = np.full_like(reference, 110)
        image[:, 31:33] = 255
        ids = np.ones((64, 64), np.uint16)
        ids[:, 32:] = 2
        result = analyze_pair(image, reference, np.ones(ids.shape, bool), source_map=ids)
        trusted = result.confidence == 255
        self.assertLess(float(result.tolerant_score[trusted].max()), 1.8)
        self.assertTrue(result.candidate_mask.any())
        self.assertFalse(result.priority_mask.any())

    def test_channels_and_validation(self):
        gray = cv2.cvtColor(pattern(), cv2.COLOR_BGR2GRAY)
        for image in (gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)):
            result = analyze_pair(image, image, np.ones(gray.shape, bool))
            self.assertEqual(result.strict_score.shape, gray.shape)
        with self.assertRaises(ValueError):
            analyze_pair(gray, gray, np.ones((2, 2), bool))
        with self.assertRaises(ValueError):
            analyze_pair(gray.astype(float), gray, np.ones(gray.shape, bool))
        with self.assertRaises(ValueError):
            analyze_pair(gray, gray, np.ones(gray.shape, bool), config=AnomalyConfig(tolerance_radius=-1))


class AnomalyFileTests(unittest.TestCase):
    def saved_pair(self, root):
        reference = pattern()
        image = reference.copy()
        image[30:55, 40:70] = (30, 70, 180)
        valid = np.ones(image.shape[:2], np.uint8)*255
        artifacts = dict(input='input.png', supposed_to_be_good='supposed_to_be_good.png',
                         reference_valid='reference_valid.png', reference_photometric_valid='reference_photometric_valid.png',
                         reference_source_map='reference_source_map.png')
        for key, array in dict(input=image, supposed_to_be_good=reference, reference_valid=valid,
                               reference_photometric_valid=valid, reference_source_map=(valid > 0).astype(np.uint16)).items():
            write_image(root / artifacts[key], array)
        metadata = dict(format_version=1, accepted=True, artifacts=artifacts,
                        normal_reference={'model': 'per_fragment_robust_affine'})
        manifest = root / 'result.json'
        manifest.write_text(json.dumps(metadata), encoding='utf-8')
        return image, reference, manifest

    def test_saved_pair_export_and_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, reference, manifest = self.saved_pair(root)
            original = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
            output = run_result(manifest)
            metadata = json.loads(output.read_text(encoding='utf-8'))
            self.assertGreater(metadata['candidate_pixels'], 0)
            self.assertEqual(output.parent, (root / 'anomaly').resolve())
            expected = analyze_pair(image, reference, np.ones(image.shape[:2], bool))
            with np.load(output.parent / 'anomaly_scores.npz') as scores:
                self.assertEqual(scores['strict_score'].dtype, np.float32)
                np.testing.assert_array_equal(scores['strict_score'], expected.strict_score)
                np.testing.assert_array_equal(scores['salience_score'], expected.salience_score)
                self.assertTrue((scores['salience_weight'] <= 1).all())
            self.assertTrue((output.parent / 'anomaly_salience_heatmap.png').exists())
            self.assertTrue((output.parent / 'anomaly_salience_overview.jpg').exists())
            candidate = read_image(output.parent / 'anomaly_candidates.png', -1)
            self.assertTrue((candidate[35:50, 45:65] == 255).all())
            self.assertTrue(metadata['regions'])
            peak = metadata['regions'][0]['peak_xy']
            self.assertGreater(candidate[peak[1], peak[0]], 0)
            write_image(root / 'input.png', reference)
            run_result(manifest)
            self.assertFalse(read_image(output.parent / 'anomaly_candidates.png', -1).any())
            for name, content in original.items():
                if name != 'input.png':
                    self.assertEqual((root / name).read_bytes(), content)

    def test_manifest_paths_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, reference, manifest = self.saved_pair(root)
            metadata = json.loads(manifest.read_text(encoding='utf-8'))
            # A valid source artifact happens to use the same name as an output.
            metadata['artifacts']['input'] = 'anomaly_overlay.png'
            write_image(root / 'anomaly_overlay.png', image)
            manifest.write_text(json.dumps(metadata), encoding='utf-8')
            before = (root / 'anomaly_overlay.png').read_bytes()
            with self.assertRaises(ValueError):
                run_result(manifest, root)
            self.assertEqual((root / 'anomaly_overlay.png').read_bytes(), before)
            metadata['artifacts']['input'] = '../outside.png'
            manifest.write_text(json.dumps(metadata), encoding='utf-8')
            with self.assertRaises(ValueError):
                run_result(manifest)

    def test_nonfinite_export_preserves_old_result(self):
        image = pattern()
        result = analyze_pair(image, image, np.ones(image.shape[:2], bool))
        with tempfile.TemporaryDirectory() as tmp:
            path = save_analysis(result, image, image, tmp)
            original = path.read_bytes()
            result.strict_score[0, 0] = np.nan
            with self.assertRaises(ValueError):
                save_analysis(result, image, image, tmp)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
