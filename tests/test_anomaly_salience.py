"""Experimental salience suppresses expected residuals without changing raw evidence."""
import unittest

import numpy as np

from reference_anomaly.salience import compute_salience
from reference_anomaly import analyze_pair
from tests.test_reference_anomaly import pattern


class SalienceTests(unittest.TestCase):
    def test_texture_background_is_suppressed_and_local_outliers_survive(self):
        texture = np.zeros((64, 64), np.float32)
        texture[:, 32:] = 10
        strict = np.full(texture.shape, .8, np.float32)
        strict[:, 32:] = 2.5
        strict[20:25, 10:15] = 6
        strict[20:25, 45:50] = 6
        valid = np.ones(texture.shape, bool)
        score, weight, metadata = compute_salience(strict, strict.copy(), texture, valid,
                                                  np.full(texture.shape, 255, np.uint8))
        self.assertLess(float(score[40:, :].mean()), .1)
        self.assertGreater(float(score[20:25, 10:15].mean()), 3)
        self.assertGreater(float(score[20:25, 45:50].mean()), 3)
        self.assertGreater(len(metadata['texture_groups']), 1)
        self.assertTrue((weight == 1).all())

    def test_shift_explained_response_and_seams_are_downweighted(self):
        strict = np.ones((64, 64), np.float32)
        strict[20:25, 10:15] = 6
        strict[20:25, 30:35] = 6
        strict[20:25, 50:55] = 6
        tolerant = strict.copy()
        tolerant[20:25, 30:35] = 0
        valid = np.ones(strict.shape, bool)
        confidence = np.full(strict.shape, 255, np.uint8)
        confidence[:, 48:] = 128
        score, weight, _ = compute_salience(strict, tolerant, np.zeros_like(strict), valid, confidence)
        self.assertGreater(float(score[22, 12]), float(score[22, 32])*3)
        self.assertLess(float(score[22, 52]), float(score[22, 12]))
        self.assertAlmostEqual(float(weight[22, 32]), .25)

    def test_empty_and_identical_inputs_are_finite_and_zero(self):
        for valid in (np.ones((64, 64), bool), np.zeros((64, 64), bool)):
            zeros = np.zeros(valid.shape, np.float32)
            score, _, _ = compute_salience(zeros, zeros, zeros, valid, valid.astype(np.uint8)*255)
            self.assertTrue(np.isfinite(score).all())
            self.assertFalse(score.any())

    def test_integration_and_unknown_pixels(self):
        reference = pattern()
        image = reference.copy()
        image[30:55, 40:70] = (10, 10, 10)
        valid = np.ones(image.shape[:2], bool)
        valid[70:] = False
        result = analyze_pair(image, reference, valid)
        self.assertGreater(float(result.salience_score[35:50, 45:65].mean()), 2)
        self.assertTrue((result.salience_score[~valid] == 0).all())
        self.assertTrue((result.salience_mask <= result.candidate_mask).all())


if __name__ == '__main__':
    unittest.main()
