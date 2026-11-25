import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig
from periodic_template_matcher._internal.segmentation.occlusion import (
    LatticeOcclusionCompletionStep, _connected_object_extension,
)
from periodic_template_matcher._internal.segmentation.layers import projected_components, record_periodic_prediction


class LatticeOcclusionTests(unittest.TestCase):
    def _run(self, *, hide=True, textured=False, periodic_donor=False, verified=True,
             repeat_axes=('x',), annotated=True):
        rng = np.random.default_rng(13)
        tile = rng.integers(0, 255, (128, 128), dtype=np.uint8)
        tile = cv2.GaussianBlur(tile, (5, 5), 1.)
        target = np.tile(tile, (1, 3))
        offset = 128 if periodic_donor else 0
        if hide:
            target[36:91, 160+offset:212+offset] = (
                rng.integers(25, 115, (55, 52), dtype=np.uint8) if textured else 80)

        roi = np.ones(target.shape, bool)
        if not periodic_donor:
            roi[60, 180] = False
        overlap = roi.copy()
        overlap[:, 128:] = False
        direct = np.zeros(target.shape, np.uint16)
        direct[48:78, 45:68] = 1
        direct[48:78, 90:100] = 1
        mapped = direct.copy()
        periodic = np.zeros_like(roi)
        known = overlap.copy()
        if periodic_donor:
            direct[48:78, 45:68] = 0
            mapped[48:78, 45:68] = 0
            donor = np.s_[48:78, 173:196]
            mapped[donor], periodic[donor], known[donor] = 1, True, True
        else:
            mapped[59, 179] = 3
        lattice = [dict(verified=True, source_vector=[128, 0], target_vector=[128, 0],
                        target_verification=dict(verified=verified))]
        source_labels = direct.copy()
        if periodic_donor:
            source_labels[donor] = 1
        if not annotated:
            source_labels[:] = 0
        result = SimpleNamespace(direct_labels=direct, direct_valid=overlap, mapped_labels=mapped,
                                 label_valid=known,
                                 label_periodic_valid=periodic, template_image=tile,
                                 completion_valid=None, completion_labels=None,
                                 registration=dict(periodic_completion=dict(lattice=lattice), ranking=[0],
                                     matches=[dict(geometry_accepted=True,
                                                   matrix_template_to_target=np.eye(3).tolist())]))
        cfg = MatcherConfig(robust_appearance=True, periodic_completion=True, periodic_max_steps=1,
                            class_periodicity={1: repeat_axes})
        inputs = SimpleNamespace(target=target, target_roi=roi, labels=[source_labels],
                                 template_valids=[np.ones_like(roi)], annotation_valids=[None])
        LatticeOcclusionCompletionStep().run(inputs, result, cfg)
        return result

    def test_uses_visible_peer_to_fill_hidden_foreground_only(self):
        result = self._run()
        self.assertGreater(np.count_nonzero(result.label_periodic_valid[48:78, 173:196]), 500)
        self.assertGreater(np.count_nonzero(result.occlusion_mask[48:78, 173:196]), 500)
        self.assertFalse(result.occlusion_mask[60, 180])
        self.assertEqual(result.registration['occlusion']['candidate_pixels'], result.occlusion_mask.sum())
        self.assertEqual(result.mapped_labels[59, 179], 3)
        self.assertEqual(result.mapped_labels[60, 180], 0)
        self.assertFalse(result.label_periodic_valid[:, 128:160].any())
        self.assertFalse(result.label_periodic_valid[:, 215:].any())
        self.assertFalse(result.direct_labels[:, 128:].any())

    def test_requires_missing_texture_verified_period_and_allowed_axis(self):
        for settings in (dict(hide=False), dict(verified=False), dict(repeat_axes=('y',)), dict(annotated=False)):
            with self.subTest(settings=settings):
                self.assertFalse(self._run(**settings).label_periodic_valid.any())

    def test_dark_textured_defect_uses_template_annotation_outside_direct_footprint(self):
        result = self._run(textured=True, periodic_donor=True)
        np.testing.assert_array_equal(result.mapped_labels[48:78, 301:324], 1)
        self.assertTrue(result.label_periodic_valid[48:78, 301:324].all())
        self.assertFalse(result.direct_labels[:, 128:].any())
        self.assertFalse(result.label_periodic_valid[:, 270:288].any())

    def test_periodic_layer_records_original_contour_across_direct_overlap(self):
        labels = np.zeros((48, 72), np.uint16)
        labels[8:20, 4:20] = 1
        labels[8:20, 26:42] = 1
        labels[8:20, 48:64] = 1
        periodic = np.zeros_like(labels, bool)
        periodic[8:20, 14:20] = True
        merged = np.zeros_like(labels)
        merged[8:20, 4:14] = 2
        result = SimpleNamespace(mapped_labels=merged, completion_labels=None, completion_valid=None)
        footprint = projected_components(labels, np.ones_like(periodic), periodic)
        record_periodic_prediction(result, labels, footprint)
        np.testing.assert_array_equal(result.completion_labels[8:20, 4:20], 1)
        self.assertFalse(result.completion_valid[:, 26:].any())
        np.testing.assert_array_equal(result.mapped_labels[8:20, 4:14], 2)

    def test_object_extension_discards_disconnected_and_other_identity_fragments(self):
        confirmed = np.zeros((32, 32), np.uint16)
        confirmed[10:20, 5:8] = 1
        confirmed[10:20, 23:26] = 2
        mapped = np.zeros_like(confirmed)
        mapped[5:10, 5:8] = 1
        mapped[24:28, 5:8] = 1
        mapped[5:10, 23:26] = 1
        approved = _connected_object_extension(mapped > 0, mapped, confirmed)
        expected = np.zeros_like(approved)
        expected[5:10, 5:8] = True
        np.testing.assert_array_equal(approved, expected)
