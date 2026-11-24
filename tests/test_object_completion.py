"""Fill missing vertical support of a confirmed identity, never horizontal copies."""
import unittest

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

    def test_bridge_follows_a_curved_component_to_both_anchors(self):
        from periodic_template_matcher._internal.objects.support_anchors import interior_bridges

        mapped = np.zeros((120, 100), np.uint16)
        mapped[15:36, 40:65] = 2
        mapped[35:84, 32:42] = 2
        mapped[84:106, 40:65] = 2
        # Both visible ends bend out of the gap's horizontal span.
        known = mapped > 0
        known[36:84] = False
        prediction = ObjectPrediction(np.where(known, mapped, 0),
                                      known.astype(np.uint16), known, {})
        rng = np.random.default_rng(34)
        gray = rng.integers(10, 240, mapped.shape, dtype=np.uint8)
        approved, records = interior_bridges(mapped, prediction, 1, gray, gray,
            np.zeros_like(known), 90, 12, lambda region, ncc: {'ncc': ncc} if ncc > .99 else None)
        np.testing.assert_array_equal(approved, (mapped > 0) & ~known)
        self.assertEqual(len(records), 1)

    def test_same_identity_repeat_fills_only_matching_unknown_contour(self):
        from periodic_template_matcher._internal.objects.support_anchors import verified_repeat_regions

        rng = np.random.default_rng(23)
        texture = rng.integers(20, 220, (64, 112), dtype=np.uint8)
        gray = np.tile(texture, (2, 1))
        gray[88:108, 45:65] = np.flipud(gray[88:108, 45:65])
        ids = np.zeros(gray.shape, np.uint16)
        mapped = np.zeros_like(ids)
        for x, identity in [(10, 1), (45, 1), (80, 2)]:
            ids[24:44, x:x+20] = identity
            mapped[88:108, x:x+20] = 2
        known = ids > 0
        prediction = ObjectPrediction(known.astype(np.uint16)*2, ids, known, {})
        approved, records = verified_repeat_regions(mapped > 0, mapped, prediction, 1, gray,
            np.zeros_like(known), np.array([0., 64.]), 32, .8,
            lambda region, ncc, step: dict(ncc=ncc))
        expected = np.zeros_like(approved)
        expected[88:108, 10:30] = True
        np.testing.assert_array_equal(approved, expected)
        self.assertEqual(len(records), 1)
