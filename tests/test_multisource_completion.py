import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher


class MultiSourceCompletionTests(unittest.TestCase):
    def test_alternate_recovers_discrete_foreground_inside_known_background(self):
        rng = np.random.default_rng(27)
        target = cv2.GaussianBlur(rng.integers(10, 240, (192, 240), dtype=np.uint8), (5, 5), 1)
        alternate = target.copy()
        alternate[96:, 144:] = np.flipud(alternate[96:, 144:])
        primary = np.zeros(target.shape, np.uint16)
        primary[105:135, 30:60] = 3
        labels = np.zeros_like(primary)
        labels[30:62, 40:72] = 1
        labels[105:135, 30:60] = 1
        labels[130:162, 175:207] = 1
        cfg = MatcherConfig(periodic_completion=True, shared_periods=True)
        registration = dict(accepted=True, status='accepted', best_template_index=0, ranking=[0, 1],
            matches=[dict(template_index=i, match_score=90-i, geometry_accepted=True,
                matrix_template_to_target=np.eye(3).tolist()) for i in range(2)])
        with patch('periodic_template_matcher.matcher.select_templates', return_value=registration):
            result = PeriodicTemplateMatcher(cfg).match([target, alternate], [primary, labels], target)
        np.testing.assert_array_equal(result.mapped_labels[30:62, 40:72], 1)
        np.testing.assert_array_equal(result.mapped_labels[105:135, 30:60], 3)
        np.testing.assert_array_equal(result.mapped_labels[130:162, 175:207], 0)
        np.testing.assert_array_equal(result.direct_labels, primary)

    def test_verified_partial_identity_foreground_overrides_base_phase(self):
        from periodic_template_matcher._internal.models import MatchResult, RegisteredMatch, PreparedInputs
        from periodic_template_matcher._internal.objects.matching import ObjectPrediction
        from periodic_template_matcher._internal.objects.steps import LocatedObjects
        from periodic_template_matcher._internal.objects.fusion import PredictionFusionStep
        image = np.zeros((16, 16), np.uint8)
        base = np.ones(image.shape, np.uint16)
        obj = np.zeros_like(base); obj[4:10, 5:12] = 2
        valid = np.ones(image.shape, bool)
        inputs = PreparedInputs([image], [base], image, [valid], [None], valid, {}, [[]])
        result = MatchResult({'accepted': True}, image, image, image, valid,
                             mapped_labels=base.copy(), label_valid=valid.copy())
        prediction = ObjectPrediction(obj, (obj > 0).astype(np.uint16)*3, valid,
                                      {'status': 'accepted'})
        cfg = MatcherConfig(ordered_object_class=2, object_cycle=list('ABCD'), object_visible_count=3)
        PredictionFusionStep().run(inputs, RegisteredMatch(result, 0, np.eye(3)),
                                   LocatedObjects(prediction, [valid]), cfg)
        np.testing.assert_array_equal(result.mapped_labels[obj > 0], 2)
        np.testing.assert_array_equal(result.mapped_labels[obj == 0], 1)
        np.testing.assert_array_equal(result.base_labels, base)
        np.testing.assert_array_equal(result.object_identities[obj > 0], 3)

    def test_other_verified_template_fills_unknown_without_overwriting_primary(self):
        rng = np.random.default_rng(42)
        image = cv2.GaussianBlur(rng.integers(20, 230, (160, 240), dtype=np.uint8), (5, 5), 1)
        labels = np.zeros(image.shape, np.uint16)
        labels[20:100, 20:60] = 1
        labels[20:100, 120:150] = 1
        other = labels.copy()
        other[20:100, 120:150] = 7
        valid = np.ones(image.shape, np.uint8)
        valid[:, :80] = 0
        cfg = MatcherConfig(max_side=240, periodic_completion=True, period_vectors=((80., 0.),),
                            angles=(0.,), scales=(1.,), ecc_iterations=15)
        for approved in (True, False):
            registration = dict(accepted=True, status='accepted', best_template_index=0, ranking=[0, 1],
                matches=[dict(template_index=i, match_score=90-i, geometry_accepted=(i == 0 or approved),
                    matrix_template_to_target=np.eye(3).tolist()) for i in range(2)])
            with self.subTest(approved=approved), patch('periodic_template_matcher.matcher.select_templates', return_value=registration):
                result = PeriodicTemplateMatcher(cfg).match([image, image], [labels, other], image,
                    template_valids=[valid, np.ones_like(valid)])
                np.testing.assert_array_equal(result.mapped_labels[:, 80:], labels[:, 80:])
                if approved:
                    self.assertTrue(result.label_valid.all())
                    np.testing.assert_array_equal(result.mapped_labels, labels)
                else:
                    self.assertFalse(result.label_valid[:, :60].any())

    def test_local_object_refinement_corrects_small_alignment_error(self):
        from periodic_template_matcher._internal.objects.matching import refine_object_transform
        rng = np.random.default_rng(17)
        source = cv2.GaussianBlur(rng.integers(20, 240, (180, 160), dtype=np.uint8), (5, 5), 1)
        truth = np.array([[1., .006, 2.3], [0., 1., -1.7], [0., 0., 1.]])
        target = cv2.warpPerspective(source, truth, (160, 180))
        mask = np.zeros(source.shape, np.uint8); mask[20:160, 40:100] = 1
        valid = np.ones(source.shape, np.uint8)
        cfg = MatcherConfig(max_side=180, ecc_iterations=60)
        actual = refine_object_transform(source, mask, valid, target, valid, np.eye(3), cfg)
        points = np.array([[[50., 40.], [90., 140.]]], np.float32)
        error = np.linalg.norm(cv2.perspectiveTransform(points, actual)-cv2.perspectiveTransform(points, truth), axis=2)
        self.assertLess(float(error.max()), .5)
