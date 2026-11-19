"""Fill missing vertical support of a confirmed identity, never horizontal copies."""
import unittest

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig
from periodic_template_matcher._internal.objects.matching import ObjectPrediction
from tests.test_ordered_objects import scene


class ObjectCompletionTests(unittest.TestCase):
    def fixture(self, damage=False):
        source, labels, refs, ids = scene()
        source, labels, ids = (np.tile(a, (3, 1)) for a in (source, labels, ids))
        refs = [dict(r, mask=np.tile(r['mask'], (3, 1))) for r in refs]
        target = source.copy()
        if damage:
            target[200:, :80] = np.random.default_rng(77).integers(0, 255, (100, 80), dtype=np.uint8)
        known = np.ones(labels.shape, bool)
        known[200:, 26:54] = False
        partial = labels.copy()
        partial[~known] = 0
        partial_ids = ids.copy()
        partial_ids[~known] = 0
        H = np.eye(3)
        H[1, 2] = -100
        info = dict(status='accepted', objects=[dict(identity='A', template_index=0, reference_index=0,
                    matrix_template_to_target=H.tolist(), supporting_transforms=[H.tolist()])])
        return source, target, refs, labels, ObjectPrediction(partial, partial_ids, known, info)

    def test_vertical_support_fills_confirmed_identity(self):
        from periodic_template_matcher._internal.objects.completion import complete_object_support
        source, target, refs, truth, prediction = self.fixture()
        complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
                                target, np.ones(target.shape, np.uint8), [[0., 100.]], MatcherConfig(max_side=300, ordered_object_class=2))
        np.testing.assert_array_equal(prediction.labels, truth)
        self.assertTrue(prediction.valid.all())
        self.assertGreater(prediction.diagnostics['support_completion']['added_valid_pixels'], 0)
        self.assertTrue(prediction.reference_pieces)
        for piece in prediction.reference_pieces:
            self.assertEqual(piece.kind, 'object_support')
            self.assertEqual(piece.template_index, 0)
            warped = cv2.warpPerspective(source, piece.matrix, (target.shape[1], target.shape[0]))
            np.testing.assert_array_equal(warped[piece.mask], target[piece.mask])

    def test_nonmatching_missing_region_stays_unknown(self):
        from periodic_template_matcher._internal.objects.completion import complete_object_support
        source, target, refs, _, prediction = self.fixture(damage=True)
        before = prediction.labels.copy()
        complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
                                target, np.ones(target.shape, np.uint8), [[0., 100.]], MatcherConfig(max_side=300, ordered_object_class=2))
        np.testing.assert_array_equal(prediction.labels, before)
        self.assertFalse(prediction.valid[200:, 26:54].any())

    def test_horizontal_period_is_not_used_for_instances(self):
        from periodic_template_matcher._internal.objects.completion import complete_object_support
        source, target, refs, _, prediction = self.fixture()
        before = prediction.valid.copy()
        complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
                                target, np.ones(target.shape, np.uint8), [[80., 0.]], MatcherConfig(max_side=300, ordered_object_class=2))
        np.testing.assert_array_equal(prediction.valid, before)

    def test_thin_mismatching_edge_cannot_borrow_evidence_from_context(self):
        from periodic_template_matcher._internal.objects.completion import complete_object_support
        for height in (1, 5):
            source, target, refs, truth, prediction = self.fixture()
            prediction.labels[:] = truth
            prediction.identities[:] = np.tile(scene()[3], (3, 1))
            prediction.valid[:] = True
            prediction.labels[-height:, 26:54] = 0
            prediction.identities[-height:, 26:54] = 0
            prediction.valid[-height:, 26:54] = False
            target[-height:, 26:54] = np.random.default_rng(2).integers(0, 255, (height, 28), dtype=np.uint8)
            with self.subTest(height=height):
                complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
                    target, np.ones(target.shape, np.uint8), [[0., 100.]], MatcherConfig(max_side=300, ordered_object_class=2))
                self.assertFalse(prediction.valid[-height:, 26:54].any())

    def test_white_occlusion_receives_complete_template_segmentation(self):
        from periodic_template_matcher._internal.objects.completion import complete_object_support
        source, target, refs, truth, prediction = self.fixture()
        prediction.labels[:, 26:54] = 0
        prediction.identities[:, 26:54] = 0
        prediction.valid[:, 26:54] = False
        target[100:200, 26:54] = 255
        complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
            target, np.ones(target.shape, np.uint8), [[0., 100.]], MatcherConfig(max_side=300, ordered_object_class=2))
        self.assertTrue(prediction.valid[:32, 26:54].all())
        self.assertTrue(prediction.valid[-32:, 26:54].all())
        self.assertTrue(prediction.valid[100:200, 26:54].all())
        np.testing.assert_array_equal(prediction.labels, truth)
        self.assertTrue((prediction.identities[100:200, 26:54] == 1).all())

    def test_completely_occluded_support_without_alignment_evidence_stays_unknown(self):
        from periodic_template_matcher._internal.objects.completion import complete_object_support
        source, target, refs, _, prediction = self.fixture()
        target[:, :80] = 255
        before = prediction.valid.copy()
        complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
            target, np.ones(target.shape, np.uint8), [[0., 100.]], MatcherConfig(max_side=300, ordered_object_class=2))
        np.testing.assert_array_equal(prediction.valid, before)

    def test_white_defect_evidence_mask_keeps_roi_and_source_masks_separate(self):
        from periodic_template_matcher._internal.objects.completion import complete_object_support
        for value in (230, 245, 255):
            with self.subTest(value=value):
                source, target, refs, truth, prediction = self.fixture()
                target[230:260, 26:54] = value
                target_valid = np.ones(target.shape, np.uint8)
                target_valid[240:250, 26:54] = 0
                source_valid = np.ones(source.shape, np.uint8)
                before_source = source_valid.copy()
                before_target = target_valid.copy()
                complete_object_support(prediction, [source], [refs], [source_valid], target,
                    target_valid, [[0., 100.]], MatcherConfig(max_side=300, ordered_object_class=2))
                self.assertTrue(prediction.valid[230:240, 26:54].all())
                self.assertTrue(prediction.valid[250:260, 26:54].all())
                self.assertFalse(prediction.valid[240:250, 26:54].any())
                np.testing.assert_array_equal(prediction.labels[230:240], truth[230:240])
                np.testing.assert_array_equal(source_valid, before_source)
                np.testing.assert_array_equal(target_valid, before_target)

    def test_small_local_translation_uses_corrected_mask(self):
        from periodic_template_matcher._internal.objects.completion import complete_object_support
        source, target, refs, truth, prediction = self.fixture()
        target = np.roll(target, 3, axis=1)
        prediction.labels[:, 26:54] = 0
        prediction.identities[:, 26:54] = 0
        prediction.valid[:, 26:54] = False
        complete_object_support(prediction, [source], [refs], [np.ones(source.shape, np.uint8)],
            target, np.ones(target.shape, np.uint8), [[0., 100.]], MatcherConfig(max_side=300, ordered_object_class=2))
        self.assertTrue(prediction.valid[40:260, 26:54].all())
        np.testing.assert_array_equal(prediction.labels[40:260, 26:54], np.roll(truth, 3, axis=1)[40:260, 26:54])

