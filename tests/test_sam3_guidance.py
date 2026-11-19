"""Synthetic reference-guidance tests, with no model, files or actual GT."""
import unittest

import cv2
import numpy as np

from reference_anomaly.box_graphcut import DetectionBox
from reference_anomaly.sam3_guidance import (
    GUIDANCE_CONFIG, GuidanceEvidence, choose_guided_candidate, guidance_evidence,
)
from reference_anomaly.sam3_prediction import select_candidate
from reference_anomaly.sam3_prompts import plan_prompts


class GuidedPromptTests(unittest.TestCase):
    def setUp(self):
        self.shape = (100, 120)
        self.box = DetectionBox(40, 30, 80, 70)
        self.score = np.zeros(self.shape, np.float32)
        self.score[40:49, 50:59] = 5
        self.score[64:68, 74:78] = 4
        self.confidence = np.full(self.shape, 255, np.uint8)

    def plan(self, boxes=None):
        return plan_prompts(self.shape, [self.box] if boxes is None else boxes,
            mode='anomaly_guided', score=self.score, valid=self.confidence != 0,
            confidence=self.confidence, context_fraction=.5)

    def test_one_joint_prompt_has_core_positives_and_near_normal_on_both_sides(self):
        record = self.plan()[0]
        self.assertEqual(record['roi_xyxy'], [20, 10, 100, 90])
        self.assertEqual(len(record['prompts']), 1)
        task = record['prompts'][0]
        self.assertEqual(task['box_xyxy'], [20, 20, 60, 60])
        self.assertEqual(task['component_ids'], [1, 2])
        self.assertEqual(task['inside_negative_count'], 8)
        self.assertEqual(task['outside_negative_count'], 8)
        inside_count = outside_count = 0
        distance = cv2.distanceTransform((self.score < 3).astype(np.uint8), cv2.DIST_L2, 5)
        for (x, y), label in zip(task['points_xy'], task['point_labels']):
            x, y = x + 20, y + 10
            self.assertEqual(self.confidence[y, x], 255)
            if label:
                self.assertGreaterEqual(self.score[y, x], 3)
                self.assertTrue(40 <= x < 80 and 30 <= y < 70)
            else:
                self.assertLessEqual(self.score[y, x], 1)
                self.assertGreaterEqual(distance[y, x], 3)
                self.assertLessEqual(distance[y, x], 8)
                if 40 <= x < 80 and 30 <= y < 70:
                    inside_count += 1
                else:
                    outside_count += 1
        self.assertEqual((inside_count, outside_count), (8, 8))

    def test_unknown_and_128_confidence_never_generate_points(self):
        self.confidence[35:42, 44:66] = 128
        self.confidence[60:65, 70:85] = 0
        record = self.plan()[0]
        self.assertEqual(record['status'], 'ok')
        for (x, y), _ in zip(record['prompts'][0]['points_xy'], record['prompts'][0]['point_labels']):
            self.assertEqual(self.confidence[y + 10, x + 20], 255)

    def test_outside_negatives_exclude_other_detection_boxes(self):
        boxes = [self.box, DetectionBox(80, 50, 95, 85)]
        record = self.plan(boxes)[0]
        for (x, y), label in zip(record['prompts'][0]['points_xy'], record['prompts'][0]['point_labels']):
            x, y = x + 20, y + 10
            if not label and not (40 <= x < 80 and 30 <= y < 70):
                self.assertFalse(80 <= x < 95 and 50 <= y < 85)

    def test_missing_one_negative_pool_is_reported_but_both_missing_abstains(self):
        self.score[64:68, 74:78] = 0  # Remaining anomaly is too far from box exterior.
        record = self.plan()[0]
        self.assertEqual(record['status'], 'ok')
        self.assertIn('no_near_normal_outside_boxes', record['warnings'])
        self.score[:] = 5
        record = self.plan()[0]
        self.assertEqual(record['status'], 'no_background_seeds')
        self.assertEqual(record['prompts'], [])
        self.score[:] = 0
        self.assertEqual(self.plan()[0]['status'], 'no_foreground_seeds')
        self.confidence[:] = 128
        self.assertEqual(self.plan()[0]['status'], 'no_reliable_reference')

    def test_truncation_and_joint_positive_budget_are_explicit(self):
        self.score[:] = 0
        self.score[32:68:6, 42:78:6] = 3
        record = self.plan()[0]
        self.assertIn('seed_components_truncated', record['warnings'])
        task = record['prompts'][0]
        self.assertEqual(task['component_ids'], list(range(1, 17)))
        self.assertEqual(task['point_labels'].count(1), 16)
        self.assertLessEqual(task['point_labels'].count(0), 16)

    def test_determinism_no_array_mutation_and_tracker_box_ignores_evidence(self):
        before = self.score.copy(), self.confidence.copy()
        self.assertEqual(self.plan(), self.plan())
        np.testing.assert_array_equal(self.score, before[0])
        np.testing.assert_array_equal(self.confidence, before[1])
        record = plan_prompts(self.shape, [self.box], mode='tracker_box', context_fraction=.5,
                              score=object(), valid=object(), confidence=object())[0]
        self.assertEqual(record['prompts'], [dict(box_xyxy=[20, 20, 60, 60],
            points_xy=[], point_labels=[], component_id=None)])

    def test_core_and_normal_evidence_are_disjoint_and_unknown_neutral(self):
        reliable = self.confidence == 255
        reliable[42:44, 52:54] = False
        inside = np.zeros(self.shape, bool)
        inside[30:70, 40:80] = True
        evidence = guidance_evidence(self.score, reliable, inside)
        self.assertFalse((evidence.core & evidence.normal).any())
        self.assertFalse(evidence.core[~reliable].any())
        self.assertFalse(evidence.normal[~reliable].any())
        self.assertTrue((evidence.core <= inside).all())
        self.assertEqual(GUIDANCE_CONFIG['policy'], 'reference_anomaly_core_near_normal_v1')


class GuidedCandidateTests(unittest.TestCase):
    def setUp(self):
        self.core = np.zeros((8, 10), bool)
        self.core[2:4, 2:4] = True
        self.normal = np.zeros_like(self.core)
        self.normal[5:7, 2:4] = True
        self.evidence = GuidanceEvidence(self.core, self.normal, self.normal,
                                         self.core.astype(np.int32), [1], 1)

    def choose(self, masks, quality, points=None, labels=None):
        masks = np.asarray(masks)
        records = select_candidate(masks, quality, self.core.shape, points or [], labels or [])[2]
        return choose_guided_candidate(masks, quality, records, self.evidence)

    def test_evidence_beats_model_quality_and_preserves_exact_candidate(self):
        bloated = self.core | self.normal
        bloated[5, 2] = False  # Still passes normal-fraction gate, but ranks lower.
        result, selected, records = self.choose([bloated, self.core], [.99, .1])
        self.assertEqual(selected, 1)
        np.testing.assert_array_equal(result, self.core)
        self.assertGreater(records[1]['evidence_score'], records[0]['evidence_score'])
        self.assertEqual(records[0]['rejection_reasons'], [])

    def test_unknown_extra_pixels_are_neutral_and_not_clipped(self):
        extra = self.core.copy()
        extra[0, :] = True  # No known normal/anomaly evidence here.
        result, selected, records = self.choose([self.core, extra], [.5, .6])
        self.assertEqual(selected, 1)
        self.assertEqual(records[0]['evidence_score'], records[1]['evidence_score'])
        np.testing.assert_array_equal(result, extra)
        self.assertTrue(result[0].all())

    def test_reject_normal_swallowing_missing_core_empty_and_point_conflicts(self):
        all_pixels = np.ones_like(self.core)
        normal = self.normal.copy()
        # Add extra known normal so all-pixels exceeds 50% normal fraction.
        self.normal[7, :] = True
        _, selected, records = self.choose([all_pixels, normal, np.zeros_like(normal)], [.9, .8, 1.])
        self.assertIsNone(selected)
        self.assertIn('excess_reliable_normal', records[0]['rejection_reasons'])
        self.assertIn('insufficient_core_coverage', records[1]['rejection_reasons'])
        self.assertIn('empty_mask', records[2]['rejection_reasons'])
        self.assertIsNone(self.choose([self.core], [.9], [[2, 2]], [0])[1])

    def test_inclusive_gates_and_stable_quality_index_ties(self):
        half = np.zeros_like(self.core)
        half[2, 2:4] = True
        half[5, 2:4] = True
        _, selected, records = self.choose([half], [.5])
        self.assertEqual(selected, 0)
        self.assertEqual(records[0]['core_coverage'], .5)
        self.assertEqual(records[0]['normal_fraction'], .5)
        self.assertEqual(self.choose([self.core, self.core], [.5, .5])[1], 0)
        self.assertEqual(self.choose([self.core, self.core], [.5, .7])[1], 1)


if __name__ == '__main__':
    unittest.main()