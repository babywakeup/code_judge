"""Normal-reference photometry, provenance and export regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, PeriodicTemplateMatcher
from periodic_template_matcher._internal.reference.step import NormalReferenceStep, fit_photometry
from periodic_template_matcher._internal.models import MatchResult, ReferencePiece
from periodic_template_matcher._internal.pipeline import MatchPipeline
from periodic_template_matcher._internal.objects.matching import ObjectPrediction
from periodic_template_matcher._internal.objects.steps import LocatedObjects
from periodic_template_matcher._internal.segmentation.preparation import InputPreparationStep
from periodic_template_matcher.utils import read_image
from tests.test_matcher_api import selection


def ramp():
    yy, xx = np.indices((96, 128))
    return (30 + xx + yy // 3).astype(np.uint8)


class PhotometryTests(unittest.TestCase):
    def test_gain_offset_excludes_bright_defects(self):
        source = ramp()
        good = 1.15 * source.astype(float) + 18
        target = np.rint(good).astype(np.uint8)
        target[25:65, 35:90] = 245  # Not saturated: residual rejection is required.
        parameters = fit_photometry(source, target, np.ones(source.shape, bool))
        self.assertEqual(parameters['status'], 'fitted')
        self.assertAlmostEqual(parameters['gain'][0], 1.15, delta=.025)
        self.assertAlmostEqual(parameters['offset'][0], 18, delta=2)
        self.assertLess(parameters['inlier_fraction'], .9)

    def test_flat_or_saturated_samples_are_not_confident(self):
        for source, target in ((np.full((64, 64), 70, np.uint8), np.full((64, 64), 90, np.uint8)),
                               (ramp(), np.full((96, 128), 255, np.uint8))):
            parameters = fit_photometry(source, target, np.ones(source.shape, bool))
            self.assertNotEqual(parameters['status'], 'fitted')
            json.dumps(parameters, allow_nan=False)


class ReferenceIntegrationTests(unittest.TestCase):
    def match(self, source, target, accepted=True, **kwargs):
        matcher = PeriodicTemplateMatcher(MatcherConfig())
        labels = np.zeros(source.shape[:2], np.uint16)
        labels[10:20, 10:20] = 1
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selection(accepted)):
            result = matcher.match([source], [labels], target, **kwargs)
        return matcher, result

    def test_reference_preserves_normal_pixels_under_defect_and_roi(self):
        source = ramp()
        good = np.rint(source.astype(float) * 1.15 + 18).astype(np.uint8)
        target = good.copy()
        target[25:65, 35:90] = 245
        original = target.copy()
        _, result = self.match(source, target, target_valid_height=80)
        self.assertIsNotNone(result.supposed_to_be_good)
        self.assertLess(np.abs(result.supposed_to_be_good[:80].astype(float) - good[:80]).mean(), 2)
        self.assertLess(int(result.supposed_to_be_good[40, 50]), 160)
        self.assertFalse(result.reference_valid[80:].any())
        self.assertTrue(result.reference_photometric_valid[:80].all())
        self.assertFalse(result.reference_source_map[80:].any())
        np.testing.assert_array_equal(target, original)
        self.assertEqual(result.mapped_labels[15, 15], 1)

    def test_channels_and_export_and_stale_cleanup(self):
        gray = ramp()
        for source in (gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)):
            target = source.copy()
            matcher, result = self.match(source, target)
            self.assertEqual(result.supposed_to_be_good.shape, target.shape)
            np.testing.assert_allclose(result.supposed_to_be_good, target, atol=1)
            with tempfile.TemporaryDirectory() as tmp:
                manifest = matcher.save_result(result, tmp)
                metadata = json.loads(manifest.read_text(encoding='utf-8'))
                self.assertEqual(metadata['artifacts']['supposed_to_be_good'], 'supposed_to_be_good.png')
                np.testing.assert_array_equal(read_image(Path(tmp) / 'supposed_to_be_good.png', -1),
                                              result.supposed_to_be_good)
                self.assertTrue((Path(tmp) / 'reference_valid.png').exists())
                self.assertTrue((Path(tmp) / 'reference_photometric_valid.png').exists())
                self.assertEqual(read_image(Path(tmp) / 'reference_source_map.png', -1).dtype, np.uint16)
                _, rejected = self.match(source, target, accepted=False)
                matcher.save_result(rejected, tmp)
                self.assertFalse((Path(tmp) / 'supposed_to_be_good.png').exists())
                self.assertFalse((Path(tmp) / 'reference_source_map.png').exists())

    def test_each_template_fragment_has_its_own_correction(self):
        source = ramp()
        other = np.rint(source.astype(float)*.8+8).astype(np.uint8)
        target = np.rint(source.astype(float)*1.1+12).astype(np.uint8)
        inputs = InputPreparationStep(MatcherConfig()).run([source, other], [None, None], target)
        left = np.zeros(source.shape, bool)
        left[:, :64] = True
        result = MatchResult(selection(), target, source, source.copy(), np.ones(source.shape, bool))
        result.reference_pieces = [ReferencePiece(0, np.eye(3), left, 'direct'),
                                   ReferencePiece(1, np.eye(3), ~left, 'periodic')]
        NormalReferenceStep().run(inputs, result)
        np.testing.assert_allclose(result.supposed_to_be_good, target, atol=2)
        self.assertTrue((result.reference_source_map[:, :64] == 1).all())
        self.assertTrue((result.reference_source_map[:, 64:] == 2).all())
        parameters = result.registration['normal_reference']['pieces']
        self.assertNotAlmostEqual(parameters[0]['photometry']['gain'][0], parameters[1]['photometry']['gain'][0])

    def test_invalid_reference_does_not_erase_saved_run(self):
        matcher, result = self.match(ramp(), ramp())
        with tempfile.TemporaryDirectory() as tmp:
            manifest = matcher.save_result(result, tmp)
            original = manifest.read_bytes()
            result.reference_source_map = np.zeros((2, 2), np.uint16)
            with self.assertRaises(ValueError):
                matcher.save_result(result, tmp)
            self.assertEqual(manifest.read_bytes(), original)

    def test_periodic_reference_covers_border_with_real_source_pixels(self):
        from tests.test_periodic_completion import fixture, selection as periodic_selection
        source, labels, target, _, matrix = fixture()
        cfg = MatcherConfig(max_side=240, angles=(0.,), scales=(1.,), ecc_iterations=25,
                            periodic_completion=True, period_vectors=((80., 0.), (0., 80.)))
        with patch('periodic_template_matcher.matcher.select_templates', return_value=periodic_selection(True, matrix)):
            result = PeriodicTemplateMatcher(cfg).match([source], [labels], target)
        self.assertTrue(result.reference_valid.all())
        np.testing.assert_allclose(result.supposed_to_be_good, target, atol=1)
        pieces = result.registration['normal_reference']['pieces']
        self.assertTrue(any(p['kind'] == 'periodic' for p in pieces))
        for piece in pieces:
            warped = cv2.warpPerspective(source, np.asarray(piece['matrix_template_to_target']), (240, 240))
            mask = result.reference_source_map == piece['source_id']
            np.testing.assert_allclose(warped[mask], target[mask], atol=1)

    def test_object_only_match_exports_only_verified_objects(self):
        from tests.test_ordered_objects import scene
        source, labels, objects, _ = scene()
        target, _, _, _ = scene('BCA')
        cfg = MatcherConfig(ordered_object_class=2, object_cycle=('A', 'B', 'C'), object_visible_count=3,
                            object_min_score=.65, object_min_gap=.03, max_side=240,
                            angles=(0.,), scales=(1.,), refine_candidates=2, ecc_iterations=15)
        selected = selection(False)
        selected['matches'][0]['matrix_template_to_target'] = None
        with patch('periodic_template_matcher.matcher.select_templates', return_value=selected):
            result = PeriodicTemplateMatcher(cfg).match([source], [labels], target, template_objects=[objects])
        self.assertEqual(result.registration['status'], 'accepted_objects_only')
        self.assertTrue(result.reference_valid.any())
        self.assertFalse(result.reference_valid[result.object_identities == 0].any())
        self.assertLess(np.abs(result.supposed_to_be_good[result.reference_valid].astype(float)
                               - target[result.reference_valid]).mean(), 3)

    def test_displaced_objects_do_not_leave_old_template_foreground(self):
        source = ramp()
        valid = np.ones(source.shape, bool)
        inputs = InputPreparationStep(MatcherConfig()).run([source], [None], source)
        raw = np.zeros_like(valid)
        raw[10:70, 20:30] = True
        shifted = np.roll(raw, 20, axis=1)
        inputs.objects = [[dict(identity='A', mask=raw)]]
        result = MatchResult(selection(), source, source, source, valid,
                             object_identities=shifted.astype(np.uint16))
        result.reference_pieces = [ReferencePiece(0, np.eye(3), valid.copy(), 'direct')]
        matrix = np.array([[1., 0., 20.], [0., 1., 0.], [0., 0., 1.]])
        prediction = ObjectPrediction(shifted.astype(np.uint16)*2, shifted.astype(np.uint16), valid, {},
                                     [ReferencePiece(0, matrix, shifted.copy(), 'object')])
        from periodic_template_matcher._internal.models import RegisteredMatch
        MatchPipeline._object_reference(inputs, RegisteredMatch(result, 0, np.eye(3)),
                                        LocatedObjects(prediction, [valid]))
        NormalReferenceStep().run(inputs, result)
        self.assertFalse(result.reference_valid[raw].any())
        self.assertTrue(result.reference_valid[shifted].all())

    def test_fractional_warp_does_not_mix_excluded_source_rows(self):
        source = ramp()
        source[48:] = 255
        inputs = InputPreparationStep(MatcherConfig()).run([source], [None], ramp(),
                                                          template_valid_heights=[48])
        matrix = np.array([[1., 0., 0.], [0., 1., -.5], [0., 0., 1.]])
        valid = np.ones(source.shape, bool)
        result = MatchResult(selection(), inputs.target, source, source, valid)
        result.reference_pieces = [ReferencePiece(0, matrix, valid, 'direct')]
        NormalReferenceStep().run(inputs, result)
        self.assertTrue(result.reference_valid[46].all())
        self.assertFalse(result.reference_valid[47:].any())

    def test_global_fallback_and_uncorrected_fallback_are_flagged(self):
        source = ramp()
        target = np.rint(source.astype(float)*1.1+12).astype(np.uint8)
        inputs = InputPreparationStep(MatcherConfig()).run([source], [None], target)
        valid = np.ones(source.shape, bool)
        tiny = np.zeros_like(valid)
        tiny[40, 50] = True
        result = MatchResult(selection(), target, source, source, valid)
        result.reference_pieces = [ReferencePiece(0, np.eye(3), valid, 'direct'),
                                   ReferencePiece(0, np.eye(3), tiny, 'object_support')]
        NormalReferenceStep().run(inputs, result)
        self.assertEqual(result.registration['normal_reference']['pieces'][1]['photometry']['status'], 'global_fallback')
        self.assertFalse(result.reference_photometric_valid[40, 50])
        self.assertAlmostEqual(int(result.supposed_to_be_good[40, 50]), int(target[40, 50]), delta=1)
        _, flat = self.match(np.full((64, 64), 80, np.uint8), np.full((64, 64), 110, np.uint8))
        self.assertEqual(flat.registration['normal_reference']['pieces'][0]['photometry']['status'], 'uncorrected')
        self.assertFalse(flat.reference_photometric_valid.any())
        self.assertTrue((flat.supposed_to_be_good == 80).all())


if __name__ == '__main__':
    unittest.main()
