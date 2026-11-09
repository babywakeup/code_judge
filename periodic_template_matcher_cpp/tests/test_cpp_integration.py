"""Public-API integration parity: real preparation, registration and both pipelines.

No patched selectors, injected steps, native substitutes or optional-native skips.
Fixtures are shared with the Python suite; acceptance thresholds stay unchanged.
"""

from copy import deepcopy
from dataclasses import asdict
import json
import os
import sys
import unittest

# A coordinator-provided dependency target is diagnostic-only. No installation,
# mutation of the active environment, or alternate baseline in normal tests.
if __name__ == '__main__' and '--diagnose-registration' in sys.argv:
    target = os.environ.get('PTM_DIAGNOSTIC_OPENCV_TARGET')
    if target:
        sys.path.insert(0, target)

import numpy as np
import cv2

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher as PythonMatcher
from periodic_template_matcher_cpp import PeriodicTemplateMatcher as CppMatcher
from tests.helpers import texture
from tests import test_ordered_objects as ordered_fixtures
from tests import test_partial_instance_cycle as partial_fixtures
from tests.test_rotation_equivariance import turn


class CppPipelineIntegrationTests(unittest.TestCase):
    backends = ('fft', 'opencv')
    array_fields = (
        'input_image', 'template_image', 'warped_template', 'geometric_valid',
        'mapped_labels', 'label_valid', 'completed_template', 'completed_valid',
        'periodic_valid', 'label_periodic_valid', 'object_identities', 'object_valid',
        'base_labels', 'base_valid',
    )

    def config(self, backend, **options):
        return MatcherConfig(backend=backend, max_side=240, angles=(0.,), scales=(1.,),
                             refine_candidates=2, ecc_iterations=15, **options)

    def assert_tree_equal(self, actual, expected):
        """Compare metadata/metrics without ambiguous ndarray dictionary equality."""
        if isinstance(expected, np.ndarray):
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_tree_equal(left, right)
        elif isinstance(expected, float):
            self.assertAlmostEqual(actual, expected, places=6)
        else:
            self.assertEqual(actual, expected)

    def assert_match_parity(self, native, python, *, compare_completed_image=True):
        self.assertEqual(native.accepted, python.accepted)
        self.assertEqual(native.best_template_index, python.best_template_index)
        self.assertEqual(native.registration['status'], python.registration['status'])
        self.assertEqual(native.registration['ranking'], python.registration['ranking'])
        for key in ('base_registration_accepted', 'base_registration_status',
                    'prediction_scope', 'template_selection_ambiguous'):
            self.assertEqual(native.registration.get(key), python.registration.get(key))
        base_geometry_accepted = python.registration['matches'][python.registration['ranking'][0]][
            'geometry_accepted']
        if python.matrix_template_to_target is None:
            self.assertIsNone(native.matrix_template_to_target)
        elif base_geometry_accepted:
            # Same original-pixel tolerance as the existing native pipeline test.
            np.testing.assert_allclose(native.matrix_template_to_target,
                                       python.matrix_template_to_target, atol=.1, rtol=0)
        for field in self.array_fields:
            with self.subTest(field=field):
                left, right = getattr(native, field), getattr(python, field)
                if right is None:
                    self.assertIsNone(left)
                else:
                    self.assertIsNotNone(left)
                    self.assertEqual(left.dtype, right.dtype)
                    self.assertEqual(left.shape, right.shape)
                    # A rejected transform is diagnostic, not an accuracy claim.
                    # The objects-only fixture below checks its own image/mask
                    # against cv2 with that very matrix, without a tolerance.
                    if not base_geometry_accepted and field in ('warped_template','geometric_valid'):
                        continue
                    # The supplemental fixture checks its exact per-source
                    # composition below. Its two independently optimized ECC
                    # matrices can cross an interpolation quantization boundary.
                    if field == 'completed_template' and not compare_completed_image:
                        continue
                    np.testing.assert_array_equal(left, right)
        for key in ('ordered_objects', 'periodic_completion'):
            if key in python.registration:
                self.assertIn(key, native.registration)
                self.assertEqual(native.registration[key]['status'],
                                 python.registration[key]['status'])

    def run_pair(self, cfg, method, args, kwargs, check, *, compare_completed_image=True):
        """Validate each real implementation against GT, then compare their results.

        Separate subtests retain Python fixture evidence if the native boundary
        raises; native exceptions remain test errors, never skips or successes.
        """
        results = {}
        for name, implementation in (('python', PythonMatcher), ('cpp', CppMatcher)):
            with self.subTest(implementation=name):
                result = getattr(implementation(cfg), method)(*args, **kwargs)
                check(result)
                results[name] = result
        if len(results) == 2:
            native, python = results['cpp'], results['python']
            if method == 'evaluate':
                self.assert_match_parity(native.match, python.match,
                                         compare_completed_image=compare_completed_image)
                self.assert_tree_equal(native.metrics, python.metrics)
                np.testing.assert_array_equal(native.ground_truth, python.ground_truth)
                np.testing.assert_array_equal(native.ground_truth_valid, python.ground_truth_valid)
            else:
                self.assert_match_parity(native, python,
                                         compare_completed_image=compare_completed_image)

    def test_roi_height_annotation_knowledge_and_evaluation_roi(self):
        image = texture()
        labels = np.ones(image.shape[:2], np.uint16)
        source_roi = np.ones_like(labels, np.uint8)
        source_roi[20:30, :30] = 0
        annotation = np.ones_like(source_roi)
        annotation[70:90, 50:80] = 0
        target_roi = np.ones_like(source_roi)
        target_roi[:10] = 0
        target_roi[40:60, 110:140] = 0
        truth_roi = np.zeros_like(source_roi)
        truth_roi[35:100, 30:170] = 1
        expected_valid = (source_roi > 0) & (annotation > 0) & (target_roi > 0)
        expected_valid[145:] = False
        evaluation_roi = (target_roi > 0) & (truth_roi > 0)
        evaluation_roi[145:] = False

        def check(result):
            self.assertTrue(result.match.accepted)
            np.testing.assert_array_equal(result.match.label_valid, expected_valid)
            np.testing.assert_array_equal(result.match.mapped_labels,
                                          expected_valid.astype(np.uint16))
            np.testing.assert_array_equal(result.ground_truth_valid, evaluation_roi)
            self.assertEqual(result.metrics['evaluated_pixels'], int(evaluation_roi.sum()))
            self.assertEqual(result.match.registration['config']['valid_height'], 140)
            regions = result.match.registration['valid_regions']
            self.assertEqual(regions['templates'][0]['valid_height'], 150)
            self.assertEqual(regions['target']['valid_height'], 145)
            # GT ROI must not restrict inference to the evaluation rectangle.
            self.assertTrue(result.match.label_valid[110:130, 30:170].all())

        for backend in self.backends:
            with self.subTest(backend=backend):
                self.run_pair(self.config(backend, valid_height=140), 'evaluate',
                    ([image], [labels], image, labels), dict(template_valids=[source_roi],
                    annotation_valids=[annotation], target_valid=target_roi,
                    template_valid_heights=[150], target_valid_height=145,
                    input_valid=truth_roi), check)

    def test_rejection_with_and_without_geometry_never_transfers_labels(self):
        image = texture()
        labels = np.ones(image.shape[:2], np.uint16)
        targets = (
            ('noise', np.random.default_rng(17).integers(0, 255, image.shape, dtype=np.uint8)),
            ('flat', np.full_like(image, 80)),
        )
        for backend in self.backends:
            for name, target in targets:
                with self.subTest(backend=backend, target=name):
                    def check(result):
                        match = result.match
                        self.assertFalse(match.accepted)
                        self.assertEqual(match.registration['status'], 'low_match_score')
                        self.assertIsNone(match.mapped_labels)
                        self.assertIsNone(match.label_valid)
                        self.assertIsNone(match.completed_template)
                        self.assertIsNone(match.object_identities)
                        self.assertEqual(result.metrics['status'], 'match_rejected')
                        self.assertEqual(result.metrics['evaluated_pixels'], 0)
                        self.assertIsNone(result.score)
                        if name == 'flat':
                            self.assertIsNone(match.matrix_template_to_target)
                            self.assertIsNone(match.warped_template)
                            self.assertFalse(match.geometric_valid.any())
                        else:
                            self.assertIsNotNone(match.matrix_template_to_target)
                            self.assertIsNotNone(match.warped_template)
                    self.run_pair(self.config(backend), 'evaluate',
                                  ([image], [labels], target, labels), {}, check)

    def test_accepted_without_annotations_has_geometry_but_no_prediction(self):
        image = texture()
        truth = np.ones(image.shape[:2], np.uint16)

        def check(result):
            self.assertTrue(result.match.accepted)
            self.assertEqual(result.match.registration['status'], 'accepted')
            self.assertIsNotNone(result.match.warped_template)
            self.assertTrue(result.match.geometric_valid.all())
            self.assertIsNone(result.match.mapped_labels)
            self.assertIsNone(result.match.label_valid)
            self.assertIsNone(result.match.completed_template)
            self.assertNotIn('periodic_completion', result.match.registration)
            self.assertEqual(result.metrics['status'], 'no_template_annotations')
            self.assertEqual(result.metrics['prediction_coverage'], 0.)
            self.assertIsNone(result.score)

        for backend in self.backends:
            with self.subTest(backend=backend):
                self.run_pair(self.config(backend, periodic_completion=True), 'evaluate',
                              ([image], [None], image, truth), {}, check)

    def test_geometric_template_tie_requires_object_evidence(self):
        source, labels, refs, identities = ordered_fixtures.scene()
        for backend in self.backends:
            with self.subTest(backend=backend, objects=False):
                def rejected(result):
                    self.assertFalse(result.accepted)
                    self.assertEqual(result.registration['status'], 'templates_too_close')
                    self.assertTrue(all(m['geometry_accepted'] for m in result.registration['matches']))
                    self.assertIsNone(result.mapped_labels)
                self.run_pair(self.config(backend), 'match',
                              ([source, source], [labels, labels], source), {}, rejected)
            with self.subTest(backend=backend, objects=True):
                def resolved(result):
                    self.assertTrue(result.accepted)
                    self.assertEqual(result.registration['status'], 'accepted_ordered_objects')
                    self.assertTrue(result.registration['template_selection_ambiguous'])
                    self.assertEqual(result.registration['ordered_objects']['visible_order'], 'ABC')
                    np.testing.assert_array_equal(result.mapped_labels, labels)
                    np.testing.assert_array_equal(result.object_identities, identities)
                    self.assertFalse(result.base_labels.any())
                    self.assertFalse(np.shares_memory(result.base_labels, result.mapped_labels))
                self.run_pair(self.config(backend, ordered_object_class=2), 'match',
                              ([source, source], [labels, labels], source),
                              dict(template_objects=[refs, refs]), resolved)

    def test_objects_only_without_annotations_uses_complementary_sources(self):
        self.check_objects_only(offset=0, annotated=False, base_accepted=True)

    def test_objects_only_survives_real_base_rejection_without_base_transfer(self):
        # Shared clipped-window fixture: default min_score=60 rejects the base
        # (observed score ~54), while local identity evidence remains accepted.
        self.check_objects_only(offset=-43, annotated=True, base_accepted=False)

    def test_registration_norm_uses_float32_division_not_reciprocal_multiplication(self):
        from periodic_template_matcher_cpp import _native
        from periodic_template_matcher._internal.registration.core import _features

        for order, offset in (('ABC', 0), ('DAB', 0), ('BCD', -43)):
            with self.subTest(order=order, offset=offset):
                image, _, _ = partial_fixtures.scene(order, offset)
                before = image.copy()
                mask = np.ones(image.shape, np.uint8)
                actual = _native.features(image, mask)
                expected = _features(image, mask)
                self.assertEqual(actual[0].dtype, np.float32)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[2], expected[2])
                np.testing.assert_array_equal(actual[3], expected[3])
                np.testing.assert_array_equal(image, before)

    def check_objects_only(self, offset, annotated, base_accepted):
        first, first_refs, first_ids = partial_fixtures.scene('ABC')
        second, second_refs, second_ids = partial_fixtures.scene('DAB')
        target, _, identities = partial_fixtures.scene('BCD', offset)
        labels = [(ids > 0).astype(np.uint16)*2 for ids in (first_ids, second_ids)]
        if not annotated:
            labels = [None, None]
        truth = (identities > 0).astype(np.uint16)*2

        def check(result):
            self.assertTrue(result.accepted)
            self.assertEqual(result.registration['status'], 'accepted_objects_only')
            self.assertEqual(result.registration['prediction_scope'], 'ordered_objects')
            self.assertEqual(result.registration['base_registration_accepted'], base_accepted)
            if not base_accepted:
                self.assertEqual(result.registration['base_registration_status'], 'low_match_score')
            if result.matrix_template_to_target is not None:
                size=(target.shape[1],target.shape[0])
                matrix=result.matrix_template_to_target
                self.assertTrue(np.isfinite(matrix).all())
                np.testing.assert_array_equal(result.warped_template,
                    cv2.warpPerspective(result.template_image,matrix,size))
                source_roi=np.ones(result.template_image.shape[:2],np.uint8)
                np.testing.assert_array_equal(result.geometric_valid,
                    cv2.warpPerspective(source_roi,matrix,size,flags=cv2.INTER_NEAREST)>0)
            info = result.registration['ordered_objects']
            self.assertEqual(info['status'], 'accepted')
            self.assertEqual(info['visible_order'], 'BCD')
            self.assertEqual({obj['template_index'] for obj in info['objects']}, {0, 1})
            np.testing.assert_array_equal(result.object_identities, identities)
            np.testing.assert_array_equal(result.mapped_labels, truth)
            np.testing.assert_array_equal(result.label_valid, identities > 0)
            np.testing.assert_array_equal(result.object_valid, result.label_valid)
            self.assertFalse(result.base_labels.any())
            self.assertFalse(result.base_valid.any())
            self.assertIsNone(result.completed_template)
            self.assertNotIn('periodic_completion', result.registration)

        for backend in self.backends:
            with self.subTest(backend=backend):
                cfg = partial_fixtures.PartialCycleTests().cfg(backend=backend)
                self.run_pair(cfg, 'match', ([first, second], labels, target),
                              dict(template_objects=[first_refs, second_refs]), check)

    def test_verified_supplement_fills_unknown_without_overwriting_primary(self):
        image = texture()
        other_image = image.copy()
        # Degrade an unrelated area, not registration thresholds: both sources
        # retain accepted geometry and the unchanged primary wins by >3 points.
        other_image[20:65, 110:180] = np.random.default_rng(42).integers(
            0, 255, other_image[20:65, 110:180].shape, dtype=np.uint8)
        primary = np.zeros(image.shape[:2], np.uint16)
        primary[35:70, 40:90] = 1
        primary[95:120, 130:160] = 1
        secondary = primary.copy()
        secondary[secondary > 0] = 7
        annotation = np.ones_like(primary, np.uint8)
        annotation[:, :100] = 0
        expected = primary.copy()
        expected[:, :100] = secondary[:, :100]

        def check(result):
            self.assertTrue(result.accepted)
            self.assertEqual(result.best_template_index, 0)
            self.assertTrue(result.registration['matches'][1]['geometry_accepted'])
            self.assertTrue(result.label_valid.all())
            np.testing.assert_array_equal(result.mapped_labels, expected)
            np.testing.assert_array_equal(result.mapped_labels[:, 100:], primary[:, 100:])
            info = result.registration['periodic_completion']
            self.assertEqual(info['status'], 'complete')
            self.assertEqual(info['remaining_unknown_pixels'], 0)
            supplement, = info['supplemental_templates']
            self.assertEqual(supplement['template_index'], 1)
            self.assertEqual(supplement['added_label_pixels'], 160*100)
            self.assertEqual(supplement['periodic_completion']['status'], 'complete')
            # Verify every output pixel using each implementation's own accepted
            # matrix. This remains exact and catches wrong source, fill or warp.
            own_image=image.copy()
            secondary_matrix=np.asarray(result.registration['matches'][1]['matrix_template_to_target'])
            own_image[:,:100]=cv2.warpPerspective(other_image,secondary_matrix,
                                                (image.shape[1],image.shape[0]))[:,:100]
            np.testing.assert_array_equal(result.completed_template,own_image)

        for backend in self.backends:
            with self.subTest(backend=backend):
                self.run_pair(self.config(backend, periodic_completion=True,
                              period_vectors=((80., 0.),)), 'match',
                              ([image, other_image], [primary, secondary], image),
                              dict(annotation_valids=[annotation, None]), check,
                              compare_completed_image=False)

    def test_quarter_turns_preserve_labels_roi_and_ordered_identities(self):
        source, labels, refs, identities = ordered_fixtures.scene()
        roi = np.ones(source.shape, np.uint8)
        roi[-4:] = 0
        for backend in self.backends:
            for k in (-1, 1):
                with self.subTest(backend=backend, rotation=k):
                    cycle = tuple('ABC' if k == -1 else 'CBA')
                    expected_ids = np.zeros_like(identities)
                    for i, identity in enumerate('ABC'):
                        expected_ids[identities == i+1] = cycle.index(identity)+1
                    expected_ids[roi == 0] = 0
                    expected_labels = labels.copy()
                    expected_labels[roi == 0] = 0
                    rotated_roi = turn(roi, k) > 0

                    def check(result):
                        self.assertTrue(result.accepted)
                        info = result.registration['ordered_objects']
                        self.assertEqual(info['status'], 'accepted')
                        self.assertEqual(info['order_axis'], 'y')
                        self.assertEqual(info['visible_order'], ''.join(cycle))
                        np.testing.assert_array_equal(result.mapped_labels, turn(expected_labels, k))
                        np.testing.assert_array_equal(result.object_identities, turn(expected_ids, k))
                        np.testing.assert_array_equal(result.label_valid, rotated_roi)
                        self.assertFalse(result.base_labels.any())

                    cfg = self.config(backend, ordered_object_class=2, object_order_axis='y',
                                      object_cycle=cycle, class_periodicity={'2': ['x']})
                    self.run_pair(cfg, 'match', ([turn(source, k)], [turn(labels, k)], turn(source, k)),
                        dict(template_valids=[turn(roi, k)], target_valid=turn(roi, k),
                        template_objects=[[dict(ref, mask=turn(ref['mask'], k)) for ref in refs]]), check)

    def test_noncontiguous_inputs_and_caller_owned_object_metadata_are_unchanged(self):
        source, labels, refs, _ = ordered_fixtures.scene()
        # Transpose views exercise real native copying, including object masks.
        source, labels = source.T, labels.T
        target = source.copy()
        refs = [dict(ref, mask=ref['mask'].T, shape_index=i) for i, ref in enumerate(refs)]
        roi = np.ones(source.T.shape, np.uint8).T
        annotation = roi.copy()
        annotation[:4, :20] = 0
        args = ([source], [labels], target)
        kwargs = dict(template_valids=[roi], annotation_valids=[annotation],
                      target_valid=roi, template_objects=[refs])
        before = deepcopy((args, kwargs))
        self.assertFalse(source.flags.c_contiguous)
        self.assertFalse(labels.flags.c_contiguous)

        for backend in self.backends:
            with self.subTest(backend=backend):
                cfg = self.config(backend, ordered_object_class=2, object_order_axis='y')
                config_before = asdict(cfg)

                def check(result):
                    self.assertTrue(result.accepted)
                    self.assertEqual(result.registration['ordered_objects']['status'], 'accepted')
                    self.assert_tree_equal((args, kwargs), before)
                    self.assert_tree_equal(asdict(cfg), config_before)
                    self.assertTrue(all('background_known' not in ref for ref in refs))
                    self.assertFalse(np.shares_memory(result.base_labels, result.mapped_labels))

                self.run_pair(cfg, 'match', args, kwargs, check)
                # Also check after a native exception, not just after success.
                self.assert_tree_equal((args, kwargs), before)
                self.assert_tree_equal(asdict(cfg), config_before)


def _difference_evidence(actual, expected):
    """JSON-safe exact differences, never a pass/fail tolerance adjustment."""
    if actual is None or expected is None:
        return dict(actual_none=actual is None, expected_none=expected is None)
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return dict(actual_shape=list(actual.shape), expected_shape=list(expected.shape))
    positions = np.argwhere(actual != expected)
    return dict(shape=list(actual.shape), actual_dtype=str(actual.dtype),
                expected_dtype=str(expected.dtype), mismatches=len(positions),
                max_abs=float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))),
                first_mismatches=[dict(index=p.tolist(), actual=actual[tuple(p)].item(),
                                       expected=expected[tuple(p)].item()) for p in positions[:20]])


def registration_difference_evidence():
    """Reproduce both failures and isolate features/refinement/warp boundaries.

    Run this module with --diagnose-registration. An already-installed isolated
    target may be supplied via PTM_DIAGNOSTIC_OPENCV_TARGET before cv2 imports.
    This diagnostic does not mock, change assertions, or replace Python formulas.
    """
    import cv2
    from periodic_template_matcher_cpp import _native
    from periodic_template_matcher._internal.registration.core import (
        _prepare_image, _coarse_candidates)
    from periodic_template_matcher._internal.registration.scoring import _refine, _score

    case = CppPipelineIntegrationTests()
    first, first_refs, first_ids = partial_fixtures.scene('ABC')
    second, second_refs, second_ids = partial_fixtures.scene('DAB')
    clipped, _, _ = partial_fixtures.scene('BCD', -43)
    object_labels = [(ids > 0).astype(np.uint16)*2 for ids in (first_ids, second_ids)]
    image = texture()
    other = image.copy()
    other[20:65, 110:180] = np.random.default_rng(42).integers(
        0, 255, other[20:65, 110:180].shape, dtype=np.uint8)
    primary = np.zeros(image.shape[:2], np.uint16)
    primary[35:70, 40:90] = 1
    primary[95:120, 130:160] = 1
    secondary = primary.copy()
    secondary[secondary > 0] = 7
    annotation = np.ones_like(primary, np.uint8)
    annotation[:, :100] = 0
    evidence = dict(versions=dict(native_opencv=_native.opencv_version,
                                  python_opencv=cv2.__version__, numpy=np.__version__,
                                  python=sys.executable, cv2_path=cv2.__file__,
                                  isolated_target=os.environ.get('PTM_DIAGNOSTIC_OPENCV_TARGET')),
                    python_opencv_build=cv2.getBuildInformation(), cases=[])
    for backend in case.backends:
        fixtures = (
            ('objects_only_rejected', partial_fixtures.PartialCycleTests().cfg(backend=backend),
             [first, second], object_labels, clipped,
             dict(template_objects=[first_refs, second_refs])),
            ('supplemental', case.config(backend, periodic_completion=True,
                                        period_vectors=((80., 0.),)),
             [image, other], [primary, secondary], image,
             dict(annotation_valids=[annotation, None])),
        )
        for name, cfg, templates, labels, target_image, kwargs in fixtures:
            cv2.setRNGSeed(cfg.seed)
            cv2.setNumThreads(2)
            python = PythonMatcher(cfg).match(templates, labels, target_image, **kwargs)
            native = CppMatcher(cfg).match(templates, labels, target_image, **kwargs)
            record = dict(name=name, backend=backend,
                          native_registration=native.registration,
                          python_registration=python.registration,
                          arrays={field: _difference_evidence(getattr(native, field),
                                                             getattr(python, field))
                                  for field in case.array_fields}, refinements=[], warps=[])
            scale = cfg.matching_scale(*(im.shape for im in [*templates, target_image]))
            destination = _prepare_image(target_image, None, cfg, scale).features
            for index, template in enumerate(templates):
                source = _prepare_image(template, None, cfg, scale).features
                # Inputs to both refinement functions are these same arrays;
                # comparisons below cannot be explained by different features.
                for candidate, H in enumerate(_coarse_candidates(source, destination, cfg)):
                    nh, nok = _native.refine(source, destination, H, asdict(cfg))
                    ph, pok = _refine(source, destination, H, cfg)
                    record['refinements'].append(dict(
                        template_index=index, candidate=candidate, coarse=H.tolist(),
                        native_converged=nok, python_converged=pok,
                        native_matrix=nh.tolist(), python_matrix=ph.tolist(),
                        matrix_difference=_difference_evidence(nh, ph),
                        # Use the SAME scorer to isolate geometric effects.
                        coarse_score=_score(source, destination, H, cfg),
                        native_refinement_python_score=_score(source, destination, nh, cfg),
                        python_refinement_python_score=_score(source, destination, ph, cfg)))
                for matrix_owner, result in (('native', native), ('python', python)):
                    H = result.registration['matches'][index]['matrix_template_to_target']
                    if H is None:
                        record['warps'].append(dict(template_index=index, matrix_owner=matrix_owner,
                                                    matrix_none=True))
                        continue
                    H = np.asarray(H, np.float64)
                    shape = target_image.shape[:2]
                    actual = _native.warp(template, H, shape)
                    expected = cv2.warpPerspective(template, H, shape[::-1])
                    float_actual = _native.warp(source[0], H, shape)
                    float_expected = cv2.warpPerspective(source[0], H, shape[::-1])
                    record['warps'].append(dict(template_index=index, matrix_owner=matrix_owner,
                        same_matrix_difference=_difference_evidence(actual, expected),
                        same_matrix_float_difference=_difference_evidence(float_actual, float_expected)))
                work_image = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
                native_features = _native.features(work_image, np.ones(work_image.shape, np.uint8))
                record.setdefault('features', []).append(dict(template_index=index,
                    fields=[_difference_evidence(a, b) for a, b in zip(native_features, source)]))
            evidence['cases'].append(record)
    return evidence


if __name__ == '__main__':
    if sys.argv[1:] == ['--diagnose-registration']:
        print(json.dumps(registration_difference_evidence(), indent=2))
    else:
        unittest.main()
