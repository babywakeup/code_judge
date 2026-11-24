"""Shared lattice contracts: axis provenance, identity, holes and bounded work."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from periodic_template_matcher._internal.objects.axes import transpose_diagnostics
from periodic_template_matcher._internal.objects.completion import complete_object_support, support_vectors
from periodic_template_matcher._internal.objects.matching import ObjectPrediction, refine_object_transform
from periodic_template_matcher._internal.registration.core import _features
from periodic_template_matcher._internal.registration.scoring import _ecc_translation
from periodic_template_matcher._internal.segmentation.lattice import _occurrence_hints, calibrate_lattice
from tests import test_object_completion as object_tests
from tests.test_periodic_completion import fixture, selection


class SharedPeriodTests(unittest.TestCase):
    def config(self, **kwargs):
        values = dict(max_side=240, angles=(0.,), scales=(1.,), ecc_iterations=25,
                      periodic_completion=True, shared_periods=True,
                      period_vectors=((80., 0.), (0., 80.)))
        values.update(kwargs)
        return MatcherConfig(**values)

    def match(self, cfg=None, target=None):
        source, labels, image, truth, matrix = fixture()
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(True, matrix)):
            result = PeriodicTemplateMatcher(cfg or self.config()).match(
                [source], [labels], image if target is None else target)
        return result, truth

    def test_flag_is_explicit_and_boolean(self):
        self.assertFalse(MatcherConfig().shared_periods)
        self.assertTrue(MatcherConfig.from_options({'periodic': {'shared_periods': True}}).shared_periods)
        for value in (1, 'true', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MatcherConfig(shared_periods=value).validate()

    def test_x_y_and_diagonal_copies_reuse_one_verified_lattice(self):
        with patch('periodic_template_matcher._internal.segmentation.periodic._refine',
                   side_effect=AssertionError('Verified lattice must not refine each copy')):
            result, truth = self.match()
        np.testing.assert_array_equal(result.mapped_labels, truth)
        self.assertTrue(result.label_valid.all())
        info = result.registration['periodic_completion']
        self.assertTrue(all(axis['verified'] for axis in info['lattice']))
        self.assertGreater(info['shared_geometry_copies'], 0)
        self.assertEqual(info['copy_refinement_calls'], 0)
        H = result.matrix_template_to_target
        for copy in info['copies']:
            expected_shift = sum((n*np.asarray(axis['target_vector'])
                                  for n, axis in zip(copy['offset'], info['lattice'])), np.zeros(2))
            matrix = np.asarray(copy['matrix_template_to_target'])
            np.testing.assert_allclose(matrix[:2, :2], H[:2, :2], atol=1e-12)
            np.testing.assert_allclose(matrix[:2, 2]-H[:2, 2], expected_shift, atol=1e-12)

    def test_inferred_anchor_lattice_completes_exact_labels(self):
        result, truth = self.match(self.config(period_vectors=()))
        np.testing.assert_array_equal(result.mapped_labels, truth)
        self.assertEqual(result.registration['periodic_completion']['period_source'], 'anchor_class')

    def test_wrong_configured_period_remains_unverified_and_does_not_fill(self):
        result, _ = self.match(self.config(period_vectors=((40., 0.),), periodic_max_steps=1))
        info = result.registration['periodic_completion']
        self.assertFalse(info['lattice'][0]['verified'])
        self.assertGreater(info['copy_refinement_calls'], 0)
        self.assertFalse(result.periodic_valid.any())

    def test_valid_x_does_not_certify_an_invalid_y_or_diagonal(self):
        result, _ = self.match(self.config(period_vectors=((80., 0.), (0., 57.))))
        info = result.registration['periodic_completion']
        self.assertTrue(info['lattice'][0]['verified'])
        self.assertFalse(info['lattice'][1]['verified'])
        for copy in info['copies']:
            if copy['offset'][1]:
                self.assertEqual(copy['refinement_mode'], 'independent_fallback')

    def test_changed_target_and_changed_missing_border_stay_unknown(self):
        rng = np.random.default_rng(59)
        _, _, target, _, _ = fixture()
        damaged = target.copy()
        damaged[:, -12:] = rng.integers(0, 255, (240, 12), dtype=np.uint8)
        result, _ = self.match(target=damaged)
        self.assertFalse(result.label_valid[:, -12:].any())
        result, _ = self.match(target=rng.integers(0, 255, (240, 240), dtype=np.uint8))
        self.assertFalse(result.periodic_valid.any())

    def test_axis_permissions_protect_nonperiodic_classes(self):
        cfg = self.config(class_periodicity={'1': ['x'], '1001': ['y']})
        result, _ = self.match(cfg)
        # A prohibited class blocks negative background knowledge too.
        self.assertFalse(result.label_valid[-8:, -8:].all())
        for copy in result.registration['periodic_completion']['copies']:
            if 'blocked_class_ids' not in copy:
                continue
            if copy['offset'][0]:
                self.assertIn(1001, copy['blocked_class_ids'])
            if copy['offset'][1]:
                self.assertIn(1, copy['blocked_class_ids'])

    def test_target_skew_uses_source_axis_not_target_axis(self):
        cfg = MatcherConfig(ordered_object_class=2, class_periodicity={'2': ['y']})
        self.assertEqual(support_vectors([[40., 100.]], cfg), [])
        np.testing.assert_array_equal(support_vectors([[40., 100.]], cfg,
                                      source_vectors=[[0., 100.]]), [[40., 100.]])
        self.assertEqual(support_vectors([[0., 100.]], cfg, source_vectors=[[100., 0.]]), [])

    def test_context_transpose_roundtrip_keeps_all_vector_coordinates(self):
        context = dict(order_axis='y', periodic_lattice=[dict(source_vector=[100., 0.],
            target_vector=[100., 40.], initial_source_vector=[101., 0.], verified=True)],
            support_completion=dict(source_period_vectors=[[100., 0.]], target_vector=[100., 40.]))
        swapped = transpose_diagnostics(context)
        self.assertEqual(swapped['periodic_lattice'][0]['source_vector'], [0., 100.])
        self.assertEqual(swapped['periodic_lattice'][0]['target_vector'], [40., 100.])
        self.assertEqual(transpose_diagnostics(swapped), context)

    def test_shared_period_never_changes_confirmed_identity(self):
        source, target, refs, truth, prediction = object_tests.ObjectCompletionTests().fixture()
        prediction.diagnostics['periodic_lattice'] = [dict(source_vector=[0., 100.],
            target_vector=[0., 100.], verified=True)]
        cfg = MatcherConfig(max_side=300, ordered_object_class=2, shared_periods=True)
        complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
                                target, np.ones(target.shape, np.uint8), [[0., 100.]], cfg)
        np.testing.assert_array_equal(prediction.labels, truth)
        self.assertTrue((prediction.identities[200:, 26:54] == 1).all())

    def test_sheared_object_support_uses_the_y_lattice(self):
        source, target, refs, truth, prediction = object_tests.ObjectCompletionTests().fixture()
        shear = np.array([[1., .4, 0.], [0., 1., 0.], [0., 0., 1.]])
        size = (360, 300)
        warp = lambda a: cv2.warpPerspective(a.astype(np.uint16), shear, size, flags=cv2.INTER_NEAREST)
        target = cv2.warpPerspective(target, shear, size)
        roi = warp(np.ones(source.shape, np.uint8)).astype(np.uint8)
        info = deepcopy(prediction.diagnostics)
        info['objects'][0]['matrix_template_to_target'] = (shear @ np.asarray(
            info['objects'][0]['matrix_template_to_target'])).tolist()
        info['periodic_lattice'] = [dict(source_vector=[0., 100.], target_vector=[40., 100.], verified=True)]
        prediction = ObjectPrediction(warp(prediction.labels), warp(prediction.identities),
                                      warp(prediction.valid).astype(bool), info)
        complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
            target, roi, [[40., 100.]], MatcherConfig(max_side=360, ordered_object_class=2, shared_periods=True))
        own = np.zeros(truth.shape, np.uint8)
        own[200:, 26:54] = truth[200:, 26:54] == 2
        foreground = warp(own) > 0
        self.assertGreater(int((prediction.valid & foreground).sum()), int(foreground.sum())*.9)
        self.assertTrue((prediction.identities[prediction.valid & foreground] == 1).all())

    def test_one_translation_fit_is_reused_by_fragmented_copies(self):
        source, target, refs, truth, prediction = object_tests.ObjectCompletionTests().fixture()
        target = np.roll(target, 3, axis=1)
        prediction.labels[:, 26:54] = 0
        prediction.identities[:, 26:54] = 0
        prediction.valid[:, 26:54] = False
        cfg = MatcherConfig(max_side=300, ordered_object_class=2, shared_periods=True)
        with patch('periodic_template_matcher._internal.objects.completion._ecc_translation',
                   wraps=_ecc_translation) as ecc:
            complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
                                    target, np.ones(target.shape, np.uint8), [[0., 100.]], cfg)
        self.assertEqual(ecc.call_count, 1)
        np.testing.assert_array_equal(prediction.labels[40:260, 26:54], np.roll(truth, 3, axis=1)[40:260, 26:54])

    def test_occurrence_hints_fold_multiples_but_never_mix_identities(self):
        def ref(y):
            H = np.eye(3)
            H[1, 2] = y
            return dict(template_index=0, reference_index=0, matrix_template_to_target=H.tolist())
        objects = dict(objects=[dict(identity='A', supporting_references=[ref(0), ref(200)]),
                                dict(identity='B', supporting_references=[ref(50)])])
        period, count = _occurrence_hints(objects, np.array([0., 100.]), np.eye(3), MatcherConfig())
        np.testing.assert_array_equal(period, [0., 100.])
        self.assertEqual(count, 1)

    def test_failed_group_translation_is_not_retried_for_each_tile(self):
        source, target, refs, _, prediction = object_tests.ObjectCompletionTests().fixture(damage=True)
        before = prediction.valid.copy()
        with patch('periodic_template_matcher._internal.objects.completion._ecc_translation',
                   side_effect=cv2.error('Failed group fit')) as ecc:
            complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
                target, np.ones(target.shape, np.uint8), [[0., 100.]],
                MatcherConfig(max_side=300, ordered_object_class=2, shared_periods=True))
        self.assertEqual(ecc.call_count, 1)
        self.assertGreater(prediction.diagnostics['support_completion']['shared_translation_reuses'], 0)
        np.testing.assert_array_equal(prediction.valid, before)

    def test_shared_local_proposal_is_rescored_and_bounded(self):
        source, _, _, _, _ = fixture()
        target = np.roll(source, 3, axis=1)
        valid = np.ones(source.shape, np.uint8)
        cfg = self.config()
        with patch('periodic_template_matcher._internal.objects.matching._refine',
                   side_effect=AssertionError('A shared proposal is not another ECC solve')):
            for dx in (3., -3., 20.):
                proposed = np.eye(3)
                proposed[0, 2] = dx
                actual = refine_object_transform(source, valid, valid, target, valid,
                                                 np.eye(3), cfg, _proposal=proposed)
                np.testing.assert_allclose(actual, proposed if dx == 3 else np.eye(3), atol=1e-10)

    def test_an_occurrence_hint_without_target_image_evidence_cannot_certify_a_period(self):
        source, _, _, _, _ = fixture()
        features = _features(source, np.ones(source.shape, np.uint8))
        hint = np.eye(3)
        hint[1, 2] = 74.
        objects = dict(objects=[dict(identity='A', supporting_references=[
            dict(template_index=0, reference_index=0, matrix_template_to_target=np.eye(3).tolist()),
            dict(template_index=0, reference_index=0, matrix_template_to_target=hint.tolist())])])
        with patch('periodic_template_matcher._internal.segmentation.lattice._refine',
                   side_effect=lambda src, dst, H, cfg, **kwargs: (H, False)):
            axis = calibrate_lattice(features, features, [[0., 80.]], np.eye(3), np.eye(3),
                                     np.eye(3), self.config(), configured=True, objects=objects)[0]
        self.assertEqual(axis['occurrence_hint_count'], 1)
        self.assertTrue(axis['source_verification']['verified'])
        self.assertFalse(axis['verified'])
        np.testing.assert_array_equal(axis['source_vector'], [0., 80.])
        np.testing.assert_array_equal(axis['target_vector'], [0., 80.])


if __name__ == '__main__':
    unittest.main()