"""Translation symmetry is per class and direction, not an occurrence count."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from tests.test_periodic_completion import fixture as periodic_fixture, selection
from tests import test_object_completion


def fixture():
    source, _, target, _, H = periodic_fixture()
    tile = np.zeros((80, 80), np.uint16)
    tile[5:25, 5:25] = 1
    tile[48:65, 3:20] = 1001
    tile[3:8, 48:65] = 1001
    labels = np.tile(tile, (3, 3))
    return source, labels, target, np.roll(labels, (-9, -12), axis=(0, 1)), H


class ClassPeriodicityTests(unittest.TestCase):
    def config(self, policies):
        return MatcherConfig(max_side=240, periodic_completion=True, periodic_max_steps=1,
            period_vectors=((80., 0.), (0., 80.)), class_periodicity=policies)

    def match(self, policies):
        source, labels, target, _, H = fixture()
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, H)):
            return PeriodicTemplateMatcher(self.config(policies)).match([source], [labels], target)

    def test_grouped_options_roundtrip_and_legacy_defaults(self):
        cfg = MatcherConfig.from_options(json.loads(
            '{"periodic":{"class_periodicity":{"1":["x","y"],"2":["y"]}}}'))
        self.assertEqual(cfg.periodic_axes(1), ('x', 'y'))
        self.assertEqual(cfg.periodic_axes(2), ('y',))
        self.assertEqual(cfg.periodic_axes(999), ())
        legacy = MatcherConfig(ordered_object_class=2)
        self.assertEqual(legacy.periodic_axes(1), ('x', 'y'))
        self.assertEqual(legacy.periodic_axes(2), ('y',))
        self.assertEqual(self.config({1: []}).periodic_axes(1), ())

    def test_invalid_policies_and_unsupported_ordered_horizontal_repeat(self):
        for policies in (None, [], {'0': []}, {'65536': []}, {'01': []}, {'bad': []},
                         {'1': 'xy'}, {'1': ['z']}, {'1': ['x', 'x']}, {'1': [True]},
                         {True: []}, {1: [], '1': ['y']}):
            with self.subTest(policies=policies), self.assertRaises(ValueError):
                self.config(policies).validate()
        with self.assertRaisesRegex(ValueError, 'ordered.*horizontal'):
            replace(self.config({'2': ['x']}), ordered_object_class=2).validate()

    def test_four_axis_modes_and_direct_labels_survive(self):
        source, labels, target, _, H = fixture()
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, H)):
            direct = PeriodicTemplateMatcher(MatcherConfig()).match([source], [labels], target)
        for axes in ([], ['x'], ['y'], ['x', 'y']):
            with self.subTest(axes=axes):
                result = self.match({'1': axes, '1001': axes})
                np.testing.assert_array_equal(result.mapped_labels[direct.label_valid],
                                              direct.mapped_labels[direct.label_valid])
                self.assertEqual(bool(result.label_valid[:231, 228:].any()), 'x' in axes)
                self.assertEqual(bool(result.label_valid[231:, :228].any()), 'y' in axes)
                self.assertEqual(bool(result.label_valid[231:, 228:].any()), len(axes) == 2)

    def test_mixed_classes_do_not_turn_forbidden_foreground_into_background(self):
        result = self.match({'1': ['x', 'y'], '1001': []})
        source, labels, target, _, H = fixture()
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, H)):
            direct = PeriodicTemplateMatcher(MatcherConfig()).match([source], [labels], target)
        new = result.label_valid & ~direct.label_valid
        self.assertTrue(new.any())
        self.assertTrue((result.mapped_labels[new] == 1).all())
        np.testing.assert_array_equal(result.mapped_labels == 1001, direct.mapped_labels == 1001)

    def test_missing_classes_are_nonperiodic_in_explicit_mode(self):
        result = self.match({'1': ['x', 'y']})
        self.assertFalse((result.mapped_labels[:, 228:] == 1001).any())
        self.assertFalse(result.label_valid[199:216, 231:238].any())

    def test_different_classes_use_different_axes_without_borrowed_background(self):
        result = self.match({'1': ['x'], '1001': ['y']})
        right = result.label_valid[:231, 228:]
        bottom = result.label_valid[231:, :228]
        self.assertTrue(right.any())
        self.assertTrue(bottom.any())
        self.assertTrue((result.mapped_labels[:231, 228:][right] == 1).all())
        self.assertTrue((result.mapped_labels[231:, :228][bottom] == 1001).all())
        self.assertFalse(result.label_valid[231:, 228:].any())

    def test_explicit_legacy_equivalent_policy_is_pixel_exact(self):
        legacy = self.match({})
        explicit = self.match({'1': ['x', 'y'], '1001': ['x', 'y']})
        np.testing.assert_array_equal(explicit.mapped_labels, legacy.mapped_labels)
        np.testing.assert_array_equal(explicit.label_valid, legacy.label_valid)
        np.testing.assert_array_equal(explicit.label_periodic_valid, legacy.label_periodic_valid)

    def test_object_geometry_does_not_depend_on_base_label_copy_permission(self):
        from periodic_template_matcher._internal.objects.matching import ObjectPrediction
        source, labels, target, _, H = fixture()
        refs = [dict(identity=n, mask=np.ones(source.shape, np.uint8)) for n in 'ABC']
        prediction = ObjectPrediction(np.zeros_like(labels), np.zeros_like(labels),
                                      np.zeros(labels.shape, bool), dict(status='accepted'))
        cfg = replace(self.config({'1': [], '1001': [], '2': ['y']}), ordered_object_class=2)
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, H)), \
                patch('periodic_template_matcher.matcher.match_ordered_objects', return_value=prediction), \
                patch('periodic_template_matcher.matcher.complete_object_support') as complete:
            result = PeriodicTemplateMatcher(cfg).match([source], [labels], target, template_objects=[refs])
        complete.assert_called_once()
        copies = result.registration['periodic_completion']['copies']
        self.assertTrue(any(c['geometry_accepted'] for c in copies))
        self.assertFalse(any(c['accepted'] for c in copies))

    def test_full_base_coverage_does_not_skip_object_geometry(self):
        from periodic_template_matcher._internal.objects.matching import ObjectPrediction
        source, labels, _, _, _ = fixture()
        refs = [dict(identity=n, mask=np.ones(source.shape, np.uint8)) for n in 'ABC']
        prediction = ObjectPrediction(np.zeros_like(labels), np.zeros_like(labels),
                                      np.zeros(labels.shape, bool), dict(status='accepted'))
        cfg = replace(self.config({'1': [], '1001': [], '2': ['y']}), ordered_object_class=2)
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, np.eye(3))), \
                patch('periodic_template_matcher.matcher.match_ordered_objects', return_value=prediction), \
                patch('periodic_template_matcher.matcher.complete_object_support') as complete:
            result = PeriodicTemplateMatcher(cfg).match([source], [labels], source, template_objects=[refs])
        self.assertTrue(result.base_valid.all())
        complete.assert_called_once()
        self.assertTrue(any(c['geometry_accepted'] for c in result.registration['periodic_completion']['copies']))

    def test_vector_direction_checks_each_generator_not_cancelled_sum(self):
        cfg = self.config({'1': ['y'], '2': ['x', 'y']})
        self.assertTrue(cfg.allows_periodic_shift(1, [[.5, 80.]]))
        self.assertFalse(cfg.allows_periodic_shift(1, [[80., 0.]]))
        self.assertFalse(cfg.allows_periodic_shift(1, [[80., 80.], [-80., 80.]]))
        self.assertTrue(cfg.allows_periodic_shift(2, [[80., 80.]]))

    def test_nonperiodic_ordered_class_does_not_extend_vertical_support(self):
        from periodic_template_matcher._internal.objects.completion import complete_object_support
        source, target, refs, _, prediction = test_object_completion.ObjectCompletionTests().fixture()
        before = prediction.valid.copy()
        cfg = MatcherConfig(max_side=300, ordered_object_class=2, class_periodicity={'2': []})
        complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
                                target, np.ones(target.shape, np.uint8), [[0., 100.]], cfg)
        np.testing.assert_array_equal(prediction.valid, before)

    def test_explicit_vertical_ordered_support_preserves_white_defect_transfer(self):
        from periodic_template_matcher._internal.objects.completion import complete_object_support
        source, target, refs, truth, prediction = test_object_completion.ObjectCompletionTests().fixture()
        target[230:260, 26:54] = 255
        cfg = MatcherConfig(max_side=300, ordered_object_class=2, class_periodicity={'2': ['y']})
        complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
                                target, np.ones(target.shape, np.uint8), [[0., 100.]], cfg)
        np.testing.assert_array_equal(prediction.labels, truth)
        self.assertTrue(prediction.valid.all())


if __name__ == '__main__':
    unittest.main()