import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from tests.test_periodic_completion import fixture as periodic_fixture


class MultiSourceCompletionTests(unittest.TestCase):
    def test_independently_registered_templates_fill_direct_layer_without_periodic_search(self):
        rng = np.random.default_rng(40)
        target = cv2.GaussianBlur(rng.integers(10, 240, (160, 240), dtype=np.uint8), (5, 5), 1)
        first = target[:, :160].copy()
        second = target[:, 80:].copy()
        first_labels = np.zeros(first.shape, np.uint16)
        second_labels = np.zeros(second.shape, np.uint16)
        first_labels[40:88, 108:132] = 1
        second_labels[40:88, 28:52] = 3
        second_labels[100:130, 60:76] = 1
        second_labels[40:88, 95:122] = 1
        matrices = [np.eye(3), np.array([[1., 0, 80], [0, 1., 0], [0, 0, 1.]])]
        registration = dict(accepted=True, status='accepted', best_template_index=0, ranking=[0, 1],
            matches=[dict(template_index=i, match_score=90-i, geometry_accepted=True,
                          matrix_template_to_target=matrix.tolist())
                     for i, matrix in enumerate(matrices)])
        cfg = MatcherConfig(periodic_completion=False, template_workers=1)
        with patch('periodic_template_matcher.matcher.select_templates', return_value=deepcopy(registration)):
            result = PeriodicTemplateMatcher(cfg).match([first, second], [first_labels, second_labels], target)
        self.assertTrue(result.direct_valid.all())
        np.testing.assert_array_equal(result.direct_labels[40:88, 108:132], 1)
        np.testing.assert_array_equal(result.direct_labels[100:130, 140:156], 1)
        np.testing.assert_array_equal(result.direct_labels[40:88, 175:202], 1)
        self.assertIsNone(result.completion_labels)
        bad = second.copy()
        bad[:, 80:] = np.flipud(bad[:, 80:])
        with patch('periodic_template_matcher.matcher.select_templates', return_value=deepcopy(registration)):
            rejected_gap = PeriodicTemplateMatcher(cfg).match([first, bad], [first_labels, second_labels], target)
        self.assertFalse(rejected_gap.direct_valid[:, 160:].any())
        self.assertFalse(rejected_gap.mapped_labels[40:88, 175:202].any())

    def test_secondary_background_does_not_hide_a_verified_periodic_contour(self):
        source, labels, target, _, primary_pose = periodic_fixture()
        labels[:] = 0
        labels[20:42, 70:90] = 1
        labels[20:42, 150:170] = 1
        registration = dict(accepted=True, status='accepted', best_template_index=0, ranking=[0, 1],
            matches=[dict(template_index=i, match_score=90-i, geometry_accepted=True,
                          matrix_template_to_target=pose.tolist())
                     for i, pose in enumerate((primary_pose, np.eye(3)))])
        cfg = MatcherConfig(periodic_completion=True, period_vectors=((80., 0.), (0., 80.)))
        with patch('periodic_template_matcher.matcher.select_templates', return_value=registration):
            result = PeriodicTemplateMatcher(cfg).match([source, target], [labels, np.zeros_like(labels)], target)
        np.testing.assert_array_equal(result.completion_labels[11:33, 218:238], 1)
        self.assertFalse(result.direct_labels[11:33, 218:238].any())
        self.assertTrue(result.label_valid.all())

    def test_tnxxx0_52_extends_verified_objects_along_y(self):
        root = Path(__file__).resolve().parents[1]
        image = root / ('source/SD-1/imgs/TNXXX0/'
                        '18851_AFE55CN029A_TDAOI810_52_306.465_706.372_O_L_20260119_085609.jpg')
        if not image.is_file():
            self.skipTest('Local SD-1 photograph is not installed')
        result = PeriodicTemplateMatcher(root/'periodic_template_matcher/config.sd1.labelme.opencv.json').infer(image)
        self.assertEqual(result.registration['ordered_objects']['visible_order'], 'CDA')
        self.assertTrue(all(not c['geometry_accepted'] for c in result.registration['periodic_completion']['copies']))
        for x, y, identity in ((448, 1650, 3), (463, 1840, 3), (2203, 1650, 1), (2224, 1850, 1)):
            with self.subTest(x=x, y=y):
                self.assertEqual(result.object_labels[y, x], 2)
                self.assertEqual(result.object_layer_identities[y, x], identity)
                self.assertEqual(result.mapped_labels[y, x], 2)
        self.assertFalse(result.object_labels[1891:].any())

    def test_tnxxx0_retains_direct_label1_at_verified_object_locations(self):
        root = Path(__file__).resolve().parents[1]
        photo = root / ('source/SD-1/imgs/TNXXX0/'
                        '18851_AFE55CS01EA_TDAOI810_126_-256.960_840.476_O_S_20260124_112121.jpg')
        if not photo.is_file():
            self.skipTest('Local SD-1 photograph is not installed')
        matcher = PeriodicTemplateMatcher(root/'periodic_template_matcher/config.sd1.labelme.opencv.json')
        result = matcher.infer(photo)
        for x, y, identity in ((177, 134, 4), (175, 1018, 4), (2055, 200, 2), (2055, 1096, 2)):
            self.assertEqual(result.direct_labels[y, x], 1)
            self.assertEqual(result.object_layer_identities[y, x], identity)
            self.assertEqual(result.object_labels[y, x], 2)
            self.assertEqual(result.mapped_labels[y, x], 2)

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
        primary[30:62, 40:72] = 1
        np.testing.assert_array_equal(result.direct_labels, primary)
        self.assertFalse(result.completion_labels[30:62, 40:72].any())

    def test_tnxxx0_uses_registered_direct_contours_and_rejects_clipped_periodic_seeds(self):
        root = Path(__file__).resolve().parents[1]
        photo = root / ('source/SD-1/imgs/TNXXX0/'
                        '18851_AFE55CS01FA_TDAOI810_55_-234.247_257.834_O_L_20260124_112608.jpg')
        if not photo.is_file():
            self.skipTest('Local SD-1 photograph is not installed')
        matcher = PeriodicTemplateMatcher(root/'periodic_template_matcher/config.sd1.labelme.opencv.json')
        result = matcher.infer(photo)
        self.assertTrue(result.accepted)
        for y in (147, 1025, 1878):
            self.assertEqual(result.direct_labels[y, 819], 1)
            self.assertEqual(result.completion_labels[y, 819], 0)
            self.assertEqual(result.mapped_labels[y, 819], 1)
        for x in (343, 1220):
            self.assertFalse(result.completion_labels[879:893, x:x+34].any())
            self.assertFalse(result.mapped_labels[879:893, x:x+34].any())
        self.assertFalse(result.direct_labels[1891:].any())

    def test_tnxxx0_transfers_template_contours_from_verified_alignment_phases(self):
        root = Path(__file__).resolve().parents[1]
        photo = root / ('source/SD-1/imgs/TNXXX0/'
                        '18851_AFE55CN01NA_TDAOI810_40_-367.244_-848.925_O_M_20260119_124707.jpg')
        if not photo.is_file():
            self.skipTest('Local SD-1 photograph is not installed')
        matcher = PeriodicTemplateMatcher(root/'periodic_template_matcher/config.sd1.labelme.opencv.json')
        result = matcher.infer(photo)
        self.assertTrue(result.accepted)
        for x, y in ((127, 72), (2013, 1031)):
            with self.subTest(x=x, y=y):
                self.assertEqual(result.direct_labels[y, x], 1)
                self.assertTrue(result.direct_valid[y, x])
                self.assertEqual(result.completion_labels[y, x], 0)
                self.assertEqual(result.mapped_labels[y, x], 1)
                source = next(copy for entry in result.registration['direct_completion']
                              for copy in entry['copies']
                              if copy['target_bbox'][0] <= x < copy['target_bbox'][2]
                              and copy['target_bbox'][1] <= y < copy['target_bbox'][3])
                H = np.array(source['matrix_template_to_target'])
                sx, sy = np.rint(cv2.perspectiveTransform(
                    np.array([[[x, y]]], dtype=np.float64), np.linalg.inv(H))[0, 0]).astype(int)
                self.assertEqual(matcher._templates.labels[source['template_index']][sy, sx], 1)
        self.assertFalse(result.direct_labels[1891:].any())

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
