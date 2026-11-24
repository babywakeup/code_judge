"""Exactness contracts for compact region processing, not wall-clock thresholds."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher._internal.segmentation.regions import MaskRegion, RegionResize, component_regions
from periodic_template_matcher import MatcherConfig
from periodic_template_matcher._internal.registration.scoring import (
    ScoreWarp, _edge_distances, _score, _refine, _ecc_affine,
    prepare_refinement_target, refinement_for_support)
from periodic_template_matcher._internal.registration.correlation import _corr, _correlations, _periods
from periodic_template_matcher._internal.objects.matching import _refinement_features, refine_object_transform


class CompactRegionTests(unittest.TestCase):
    def test_nearest_resize_equals_full_canvas_at_all_boundaries(self):
        rng = np.random.default_rng(8)
        for shape, size in [((97, 103), (38, 41)), ((43, 59), (117, 83)),
                            ((1891, 2464), (800, 614)), ((80, 96), (96, 80)),
                            ((33, 35), (1, 1))]:
            sampler = RegionResize(shape, size)
            for y, x, height, width in [(0, 0, 1, 1), (shape[0]-1, shape[1]-1, 1, 1),
                                         (0, shape[1]-9, 11, 9), (shape[0]-13, 0, 13, 17),
                                         (3, 5, 21, 23), (0, 0, *shape)]:
                with self.subTest(shape=shape, size=size, y=y, x=x):
                    mask = rng.integers(0, 2, (height, width)).astype(bool)
                    region = MaskRegion(y, x, mask)
                    full = np.zeros(shape, np.uint8)
                    full[region.slices] = mask
                    expected = cv2.resize(full, size, interpolation=cv2.INTER_NEAREST)
                    np.testing.assert_array_equal(sampler(region), expected)

    def test_connected_components_preserve_ids_order_holes_and_diagonals(self):
        mask = np.zeros((100, 120), bool)
        mask[0:4, 0:4] = True
        mask[8:30, 20:55] = True
        mask[12:25, 25:48] = False
        mask[60, 80] = mask[61, 81] = True
        mask[-5:, -7:] = True
        count, labels = cv2.connectedComponents(mask.astype(np.uint8))
        regions = list(component_regions(mask))
        self.assertEqual(len(regions), count-1)
        for component, region in enumerate(regions, 1):
            expected = labels == component
            full = np.zeros(mask.shape, bool)
            full[region.slices] = region.mask
            np.testing.assert_array_equal(full, expected)
            self.assertEqual(region.pixels, int(expected.sum()))
            self.assertTrue(region.mask.any(axis=0)[[0, -1]].all())
            self.assertTrue(region.mask.any(axis=1)[[0, -1]].all())

    def test_no_full_canvas_per_region_and_noncontiguous_tiles(self):
        mask = np.zeros((1900, 2400), bool)
        mask[700:709, 1100:1113] = True
        region, = list(component_regions(mask))
        self.assertEqual(region.mask.shape, (9, 13))
        sampler = RegionResize(mask.shape, (800, 633))
        tile = MaskRegion(region.y+1, region.x+2, region.mask[1:5, 2:8])
        self.assertFalse(tile.mask.flags.c_contiguous)
        full = np.zeros(mask.shape, np.uint8)
        full[tile.slices] = tile.mask
        np.testing.assert_array_equal(sampler(tile), cv2.resize(full, (800, 633), interpolation=cv2.INTER_NEAREST))


class ScoreWarpTests(unittest.TestCase):
    def test_sparse_edges_preserve_full_canvas_rounding(self):
        source, target = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
        source[0, 0] = target[14, 30] = True
        dt = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)
        ds = cv2.distanceTransform((~source).astype(np.uint8), cv2.DIST_L2, 5)
        np.testing.assert_array_equal(_edge_distances(source, target),
                                      np.concatenate([dt[source], ds[target]]))

    def test_edge_distances_match_full_frame_including_image_edges(self):
        rng = np.random.default_rng(67)
        for y, x, h, w in [(0, 0, 17, 28), (0, 150, 22, 29),
                            (100, 0, 30, 21), (110, 157, 20, 23),
                            (30, 70, 60, 19), (0, 0, 130, 180)]:
            we, et = np.zeros((130, 180), bool), np.zeros((130, 180), bool)
            we[y:y+h, x:x+w] = rng.random((h, w)) > .8
            et[y:y+h, x:x+w] = rng.random((h, w)) > .8
            dt = cv2.distanceTransform((~et).astype(np.uint8), cv2.DIST_L2, 5)
            ds = cv2.distanceTransform((~we).astype(np.uint8), cv2.DIST_L2, 5)
            expected = np.concatenate([dt[we], ds[et]])
            np.testing.assert_array_equal(_edge_distances(we, et), expected)

    def test_reused_transform_has_exact_scores_with_independent_roi_evidence(self):
        rng = np.random.default_rng(31)
        image = rng.normal(size=(120, 160)).astype(np.float32)
        edges = (rng.random(image.shape) > .75).astype(np.uint8)
        valid = np.ones(image.shape, np.uint8)
        src = (image, image, edges, valid)
        matrix = np.array([[1., .003, 2.2], [.002, 1., -1.3], [0., 0., 1.]])
        target = cv2.warpPerspective(image, matrix, (160, 120))
        target_edges = cv2.warpPerspective(edges, matrix, (160, 120), flags=cv2.INTER_NEAREST)
        masks = [valid.copy(), valid.copy(), np.zeros_like(valid)]
        masks[1][:40] = 0
        masks[2][60:100, 50:110] = 1
        cfg = MatcherConfig(min_overlap=.2)
        expected = [_score(src, (target, target, target_edges, mask), matrix, cfg) for mask in masks]
        with patch('cv2.warpPerspective', wraps=cv2.warpPerspective) as calls:
            prepared = ScoreWarp(src, matrix, (160, 120))
            actual = [_score(src, (target, target, target_edges, mask), matrix, cfg, warped=prepared)
                      for mask in masks]
        self.assertEqual(actual, expected)
        self.assertEqual(calls.call_count, 3)
        self.assertNotEqual(actual[0]['match_score'], actual[2]['match_score'])

    def test_low_overlap_does_not_compute_intensity_or_edges(self):
        image = np.zeros((64, 64), np.float32)
        valid = np.ones(image.shape, np.uint8)
        src = (image, image, valid, valid)
        transform = np.eye(3)
        transform[0, 2] = 100
        with patch('cv2.warpPerspective', wraps=cv2.warpPerspective) as calls:
            result = _score(src, src, transform, MatcherConfig())
        self.assertEqual(result['match_score'], 0.)
        self.assertEqual(calls.call_count, 1)


class RefinementFeatureTests(unittest.TestCase):
    def test_local_pyramids_share_intensity_but_not_occurrence_support(self):
        rng = np.random.default_rng(42)
        intensity = rng.normal(size=(101, 135)).astype(np.float32)
        valid = np.ones(intensity.shape, np.uint8)
        features = (intensity, intensity, valid, valid)
        shared = prepare_refinement_target(features)
        for start, end in [(0, 40), (30, 80), (85, 101)]:
            local = np.zeros_like(valid)
            local[start:end, 20:100] = 1
            independent = prepare_refinement_target((*features[:3], local))
            actual = refinement_for_support(shared, local)
            for base, expected, level in zip(shared, independent, actual):
                self.assertIs(level.intensity, base.intensity)
                self.assertIsNot(level.support, base.support)
                np.testing.assert_array_equal(level.intensity, expected.intensity)
                np.testing.assert_array_equal(level.support, expected.support)

    def test_ecc_exception_preserves_coarse_transform_with_cached_pyramid(self):
        image = np.zeros((96, 128), np.float32)
        mask = np.ones(image.shape, np.uint8)
        features = (image, image, mask, mask)
        transform = np.eye(3)
        transform[0, 2] = 2
        with patch('cv2.findTransformECC', side_effect=cv2.error('forced failure')) as calls:
            result, converged = _refine(features, features, transform, MatcherConfig(),
                                        prepared=prepare_refinement_target(features))
        self.assertEqual(calls.call_count, 2)
        self.assertFalse(converged)
        np.testing.assert_array_equal(result, transform)

    def test_out_of_bounds_refinement_is_not_scored(self):
        image = np.zeros((96, 128), np.uint8)
        valid = np.ones_like(image)
        rejected = np.eye(3)
        rejected[0, 2] = 30
        cfg = MatcherConfig(scale=1.)
        with patch('periodic_template_matcher._internal.objects.matching._refine',
                   return_value=(rejected, True)), patch(
                'periodic_template_matcher._internal.objects.matching._score',
                return_value={}) as score:
            actual = refine_object_transform(image, valid, valid, image, valid, np.eye(3), cfg)
        np.testing.assert_array_equal(actual, np.eye(3))
        self.assertEqual(score.call_count, 1)

    def test_cached_pair_features_preserve_exact_ecc_result(self):
        rng = np.random.default_rng(98)
        image = cv2.GaussianBlur(rng.integers(20, 240, (180, 160), dtype=np.uint8), (5, 5), 1)
        transform = np.array([[1., .003, 1.7], [.001, 1., -1.1], [0., 0., 1.]])
        target = cv2.warpPerspective(image, transform, (160, 180))
        mask, valid = np.zeros(image.shape, np.uint8), np.ones(image.shape, np.uint8)
        mask[20:160, 40:110] = 1
        cfg = MatcherConfig(max_side=150, ecc_iterations=30)
        expected = refine_object_transform(image, mask, valid, target, valid, np.eye(3), cfg)
        prepared = _refinement_features(image, valid, target, valid, cfg)
        # The raw resized ROI must survive independently of the eroded safety
        # mask returned by _features; substituting it changes edge alignment.
        self.assertGreater(int(prepared[-1].sum()), int(prepared[4][3].sum()))
        with patch('periodic_template_matcher._internal.objects.matching._features',
                   side_effect=AssertionError('Cached refinement must not extract features again')):
            actual = refine_object_transform(image, mask, valid, target, valid, np.eye(3), cfg,
                _prepared=prepared, _target_refinement=prepare_refinement_target(prepared[4]))
        np.testing.assert_array_equal(actual, expected)


class CorrelationReuseTests(unittest.TestCase):
    def test_spectra_reuse_preserves_float32_and_mask_rounding(self):
        rng = np.random.default_rng(70)
        for dtype in (np.float32, np.float64):
            a = rng.normal(size=(47, 61)).astype(dtype)
            b = rng.normal(size=(38, 49)).astype(dtype)
            mask = (rng.random((47, 61)) > .2).astype(np.uint8)
            pairs = [(a, a), (a, b), (b, a), (mask, a), (a, mask)]
            shape = (90, 120)
            expected = [_corr(x, y, shape) for x, y in pairs]
            with patch('numpy.fft.rfft2', wraps=np.fft.rfft2) as forward:
                actual = _correlations(pairs, shape)
            self.assertEqual(forward.call_count, 3)
            for old, new in zip(expected, actual):
                np.testing.assert_array_equal(old, new)

    def test_lattice_diagnostics_are_exact_with_three_forward_transforms(self):
        rng = np.random.default_rng(91)
        tile = rng.normal(size=(19, 23)).astype(np.float32)
        feature = np.tile(tile, (5, 5))
        mask = np.ones(feature.shape, np.uint8)
        def legacy(pairs, shape):
            return [_corr(a, b, shape) for a, b in pairs]
        with patch('periodic_template_matcher._internal.registration.correlation._correlations', legacy):
            expected = _periods(feature, mask)
        with patch('numpy.fft.rfft2', wraps=np.fft.rfft2) as forward:
            actual = _periods(feature, mask)
        self.assertTrue(actual)
        self.assertEqual(actual, expected)
        self.assertEqual(forward.call_count, 3)


class LocalECCTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(16)
        image = cv2.GaussianBlur(rng.normal(size=(240, 320)).astype(np.float32), (5, 5), .8)
        valid = np.zeros(image.shape, np.uint8)
        valid[70:170, 100:150] = 1
        return image, valid

    def test_explicit_config_option_defaults_off_and_rejects_non_bool(self):
        self.assertFalse(MatcherConfig().ecc_roi)
        self.assertTrue(MatcherConfig.from_options({'step3': {'ecc_roi': True}}).ecc_roi)
        for value in (1, 0, 'true', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MatcherConfig(ecc_roi=value).validate()

    def test_cropped_affine_is_conjugated_back_to_full_coordinates(self):
        image, mask = self.fixture()
        local = np.array([[1.01, .02, 2.], [-.01, .99, -1.]], np.float32)
        with patch('cv2.findTransformECC', return_value=(.99, local)) as ecc:
            actual = _ecc_affine(image, image, mask, MatcherConfig(ecc_roi=True))
        T = np.array([[1., 0., 76.], [0., 1., 46.], [0., 0., 1.]])
        expected = T @ np.vstack([local, [0., 0., 1.]]) @ np.linalg.inv(T)
        np.testing.assert_array_equal(actual, expected[:2])
        self.assertEqual(ecc.call_count, 1)
        self.assertEqual(ecc.call_args.args[0].shape, (148, 98))
        self.assertEqual(ecc.call_args.args[4], (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 70, 1e-5))
        np.testing.assert_array_equal(ecc.call_args.args[5], mask[46:194, 76:174]*255)

    def test_roi_failure_and_escape_both_retry_original_full_canvas(self):
        image, mask = self.fixture()
        escaped = np.array([[1., 0., 100.], [0., 1., 0.]], np.float32)
        identity = np.eye(2, 3, dtype=np.float32)
        for failure in (cv2.error('ROI convergence failure'), (.5, escaped)):
            with self.subTest(failure=str(failure)), patch(
                    'cv2.findTransformECC', side_effect=[failure, (.99, identity)]) as ecc:
                actual = _ecc_affine(image, image, mask, MatcherConfig(ecc_roi=True))
            np.testing.assert_array_equal(actual, identity)
            self.assertEqual(ecc.call_count, 2)
            self.assertIs(ecc.call_args.args[0], image)
            np.testing.assert_array_equal(ecc.call_args.args[5], mask*255)

    def test_disabled_or_dense_support_keeps_original_call(self):
        image, mask = self.fixture()
        for enabled, valid in ((False, mask), (True, np.ones_like(mask)), (True, np.zeros_like(mask))):
            with self.subTest(enabled=enabled), patch('cv2.findTransformECC',
                    return_value=(.99, np.eye(2, 3, dtype=np.float32))) as ecc:
                _ecc_affine(image, image, valid, MatcherConfig(ecc_roi=enabled))
            self.assertEqual(ecc.call_count, 1)
            self.assertIs(ecc.call_args.args[0], image)

    def test_real_ecc_recovers_translation_at_interior_and_real_image_edges(self):
        image, _ = self.fixture()
        truth = np.array([[1., .001, 1.2], [-.001, 1., -.8], [0., 0., 1.]])
        warped = cv2.warpPerspective(image, truth, (320, 240))
        for y, x in ((70, 100), (0, 0), (140, 260)):
            with self.subTest(origin=(y, x)):
                mask = np.zeros_like(image, np.uint8)
                mask[y:y+100, x:x+60] = 1
                actual = _ecc_affine(image, warped, mask, MatcherConfig(ecc_roi=True))
                points = np.array([[[x+10., y+10.], [x+40., y+70.]]], np.float32)
                error = np.linalg.norm(cv2.transform(points, actual)-cv2.perspectiveTransform(points, truth), axis=2)
                self.assertLess(float(error.max()), .6)


class BenchmarkComparisonTests(unittest.TestCase):
    def test_all_vertical_supports_are_counted_in_transform_difference(self):
        from tools.benchmark_ecc_roi import compare
        def result():
            support = dict(template_index=0, reference_index=0, matrix_template_to_target=np.eye(3).tolist())
            return SimpleNamespace(label_valid=np.ones((40, 50), bool), mapped_labels=np.zeros((40, 50), np.uint16),
                object_identities=None, input_image=np.zeros((40, 50), np.uint8), accepted=True,
                registration=dict(ranking=[0], matches=[dict(matrix_template_to_target=np.eye(3).tolist())],
                    ordered_objects=dict(visible_order='A', objects=[dict(identity='A',
                        supporting_references=[deepcopy(support), deepcopy(support)])])))
        before, after = result(), result()
        # The first occurrence changes; collapsing duplicate reference keys
        # would silently retain only the unchanged second occurrence.
        after.registration['ordered_objects']['objects'][0]['supporting_references'][0]['matrix_template_to_target'][0][2] = 5.
        drift = compare(before, after)
        self.assertAlmostEqual(drift['local_target_corner_max_displacement_original_px'], 5.)
        self.assertTrue(drift['local_reference_keys_unchanged'])

    def test_rejected_missing_labels_are_not_reported_as_exact_predictions(self):
        from tools.benchmark_ecc_roi import compare
        result = SimpleNamespace(accepted=False, label_valid=None)
        report = compare(result, result)
        self.assertTrue(report['accepted_unchanged'])
        self.assertFalse(report['labels_comparable'])


if __name__ == '__main__':
    unittest.main()
