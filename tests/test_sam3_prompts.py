"""Synthetic prompt contracts only: no model, files, real images or GT."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from reference_anomaly.box_graphcut import DetectionBox
from reference_anomaly.sam3_prompts import CONFIG, plan_prompts


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.shape = (100, 120)
        self.box = DetectionBox(40, 30, 80, 70)
        self.score = np.zeros(self.shape, np.float32)
        self.valid = np.ones(self.shape, bool)
        self.confidence = np.full(self.shape, 255, np.uint8)
        self.score[40:49, 50:59] = 5

    def plan(self, boxes=None, **kwargs):
        arguments = dict(score=self.score, valid=self.valid, confidence=self.confidence)
        arguments.update(kwargs)
        return plan_prompts(self.shape, [self.box] if boxes is None else boxes, **arguments)

    @staticmethod
    def global_points(record, task, label):
        x0, y0, _, _ = record['roi_xyxy']
        return [(x + x0, y + y0) for (x, y), value in
                zip(task['points_xy'], task['point_labels']) if value == label]

    def test_coordinates_crop_conversion_and_schema(self):
        record = self.plan()[0]
        self.assertEqual(set(record), {'index', 'box_xyxy', 'roi_xyxy', 'status', 'warnings', 'prompts'})
        self.assertEqual(record['index'], 0)
        self.assertEqual(record['box_xyxy'], [40, 30, 80, 70])
        self.assertEqual(record['roi_xyxy'], [24, 14, 96, 86])
        self.assertEqual(record['status'], 'ok')
        task = record['prompts'][0]
        self.assertEqual(set(task), {'box_xyxy', 'points_xy', 'point_labels', 'component_id'})
        self.assertEqual(task['box_xyxy'], [16, 16, 56, 56])
        self.assertEqual(task['points_xy'][0], [30, 30])
        self.assertEqual(task['point_labels'], [1] * 3 + [0] * 8)
        self.assertTrue(all(type(v) is int for point in task['points_xy'] for v in point))

    def test_padding_edges_original_bounds_and_fraction_round_up(self):
        boxes = [DetectionBox(-5, -4, 25, 25), DetectionBox(110, 90, 130, 110)]
        records = self.plan(boxes, mode='box')
        self.assertEqual(records[0]['box_xyxy'], [-5, -4, 25, 25])
        self.assertEqual(records[0]['roi_xyxy'], [0, 0, 41, 41])
        self.assertEqual(records[0]['prompts'][0]['box_xyxy'], [0, 0, 25, 25])
        self.assertEqual(records[1]['roi_xyxy'], [94, 74, 120, 100])
        self.assertEqual(records[1]['prompts'][0]['box_xyxy'], [16, 16, 26, 26])
        large = plan_prompts((500, 600), [DetectionBox(100, 100, 301, 281)], mode='box')[0]
        self.assertEqual(large['roi_xyxy'], [79, 81, 322, 300])

    def test_outside_image_in_each_direction_and_exact_edges(self):
        boxes = [DetectionBox(-20, 10, 0, 20), DetectionBox(120, 10, 130, 20),
                 DetectionBox(10, -20, 20, 0), DetectionBox(10, 100, 20, 110)]
        for mode in ('box', 'anomaly_points'):
            for index, record in enumerate(self.plan(boxes, mode=mode)):
                self.assertEqual(record['index'], index)
                self.assertEqual(record['status'], 'outside_image')
                self.assertIsNone(record['roi_xyxy'])
                self.assertEqual(record['prompts'], [])

    def test_anomaly_coordinates_for_clipped_bottom_right_detection(self):
        self.score[:] = 0
        self.score[94:100, 114:120] = 4
        record = self.plan([DetectionBox(110, 90, 130, 110)])[0]
        self.assertEqual(record['status'], 'ok')
        self.assertEqual(record['roi_xyxy'], [94, 74, 120, 100])
        task = record['prompts'][0]
        self.assertEqual(task['box_xyxy'], [16, 16, 26, 26])
        for (x, y), label in zip(task['points_xy'], task['point_labels']):
            self.assertTrue(0 <= x < 26 and 0 <= y < 26)
            if label:
                self.assertGreaterEqual(self.score[y + 74, x + 94], 3)

    def test_box_mode_no_response_no_reference_and_ignores_arrays(self):
        for evidence in ({}, dict(score=object(), valid=object(), confidence=object())):
            record = plan_prompts(self.shape, [self.box], mode='box', **evidence)[0]
            self.assertEqual(record['status'], 'ok')
            self.assertEqual(record['prompts'], [dict(box_xyxy=[16, 16, 56, 56],
                points_xy=[], point_labels=[], component_id=None)])

    def test_multiple_components_no_color_prior_or_cross_component_negatives(self):
        self.score[56:64, 65:73] = 7
        baseline = self.plan()[0]
        self.assertEqual(len(baseline['prompts']), 2)
        self.assertEqual([t['component_id'] for t in baseline['prompts']], [2, 1])
        # The same residual score can come from bright, dark or colored defects;
        # the planner has no image/color argument and must preserve both islands.
        for color in ((245, 245, 245), (10, 10, 10), (200, 30, 150)):
            synthetic_image = np.full((*self.shape, 3), 115, np.uint8)
            synthetic_image[self.score == 5] = color
            synthetic_image[self.score == 7] = tuple(255 - v for v in color)
            self.assertEqual(self.plan()[0], baseline)
        negatives = []
        for task in baseline['prompts']:
            positives = self.global_points(baseline, task, 1)
            self.assertTrue(all(self.score[y, x] >= 3 for x, y in positives))
            negatives.append(self.global_points(baseline, task, 0))
            self.assertTrue(all(self.score[y, x] <= 1 for x, y in negatives[-1]))
        self.assertEqual(negatives[0], negatives[1])

    def test_128_is_valid_but_excluded_from_all_seeds(self):
        self.confidence[:] = 128
        self.assertEqual(self.plan()[0]['status'], 'no_reliable_reference')
        self.confidence[30:70, 40:80] = 255
        self.assertEqual(self.plan()[0]['status'], 'no_background_seeds')
        self.confidence[20:25, 30:35] = 255
        self.confidence[40:44, 50:59] = 128
        record = self.plan()[0]
        self.assertEqual(record['status'], 'ok')
        for task in record['prompts']:
            for label in (0, 1):
                self.assertTrue(all(self.confidence[y, x] == 255 for x, y in
                                    self.global_points(record, task, label)))

    def test_background_excludes_all_boxes_unknown_and_high_scores(self):
        boxes = [self.box, DetectionBox(24, 14, 40, 86), DetectionBox(80, 14, 96, 86)]
        self.valid[14:30, 40:80] = False
        self.confidence[~self.valid] = 0
        self.score[70:80, 40:80] = 2
        self.score[80:86, 40:80] = 1  # Inclusive background threshold.
        record = self.plan(boxes)[0]
        self.assertEqual(record['status'], 'ok')
        for x, y in self.global_points(record, record['prompts'][0], 0):
            self.assertTrue(self.valid[y, x])
            self.assertEqual(self.confidence[y, x], 255)
            self.assertLessEqual(self.score[y, x], 1)
            self.assertTrue(all(not (b.x0 <= x < b.x1 and b.y0 <= y < b.y1) for b in boxes))

    def test_abstention_no_reliable_foreground_or_background(self):
        self.valid[:] = False
        self.confidence[:] = 0
        self.assertEqual(self.plan()[0]['status'], 'no_reliable_reference')
        self.valid[:] = True
        self.confidence[:] = 255
        self.score[:] = 0
        self.assertEqual(self.plan()[0]['status'], 'no_foreground_seeds')
        self.score[:] = 5
        record = self.plan()[0]
        self.assertEqual(record['status'], 'no_background_seeds')
        self.assertEqual(record['prompts'], [])
        self.assertEqual(self.plan([DetectionBox(0, 0, 120, 100)])[0]['status'], 'no_background_seeds')

    def test_components_peak_order_cap_ties_and_point_budget(self):
        self.score[:] = 0
        for y in range(32, 68, 6):
            for x in range(42, 78, 6):
                self.score[y, x] = 3
        self.score[62, 72] = 8
        record = self.plan()[0]
        self.assertEqual(record['warnings'], ['seed_components_truncated'])
        tasks = record['prompts']
        self.assertEqual(len(tasks), 16)
        self.assertEqual([t['component_id'] for t in tasks], [36] + list(range(1, 16)))
        for task in tasks:
            self.assertEqual(task['point_labels'].count(1), 1)
            self.assertLessEqual(task['point_labels'].count(0), 8)
        self.assertEqual(CONFIG['max_components'], 16)

    def test_integer_component_peaks_preserve_exact_order(self):
        score = np.zeros(self.shape, np.uint64)
        score[40, 50] = 2**63
        score[50, 60] = 2**63 + 1
        record = self.plan(score=score)[0]
        self.assertEqual([task['component_id'] for task in record['prompts']], [2, 1])

    def test_deterministic_spread_interior_and_no_input_mutation(self):
        self.score[:] = 0
        self.score[35:65, 45:75] = 3
        before = [a.copy() for a in (self.score, self.valid, self.confidence)]
        record = self.plan()[0]
        self.assertEqual(self.plan()[0], record)
        points = self.global_points(record, record['prompts'][0], 1)
        self.assertEqual(len(set(points)), 3)
        self.assertTrue(all(51 <= x <= 68 and 41 <= y <= 58 for x, y in points))
        self.assertGreater(min((x - xx)**2 + (y - yy)**2 for i, (x, y) in enumerate(points)
                               for xx, yy in points[i+1:]), 50)
        for actual, expected in zip((self.score, self.valid, self.confidence), before):
            np.testing.assert_array_equal(actual, expected)

    def test_thin_diagonal_component_uses_eight_connectivity(self):
        self.score[:] = 0
        self.score[np.arange(40, 50), np.arange(50, 60)] = 3
        record = self.plan()[0]
        self.assertEqual(len(record['prompts']), 1)
        points = self.global_points(record, record['prompts'][0], 1)
        self.assertEqual(points[:2], [(50, 40), (59, 49)])
        self.assertEqual(len(points), 3)

    def test_single_background_pixel_and_tiny_foreground(self):
        self.valid[:] = False
        self.confidence[:] = 0
        for x, y in ((50, 40), (30, 20)):
            self.valid[y, x] = True
            self.confidence[y, x] = 255
        record = self.plan()[0]
        self.assertEqual(record['status'], 'ok')
        self.assertEqual(record['prompts'][0]['point_labels'], [1, 0])

    def test_no_seed_work_when_no_foreground(self):
        self.score[:] = 0
        with patch('reference_anomaly.sam3_prompts._spread', side_effect=AssertionError('unexpected seed work')):
            self.assertEqual(self.plan()[0]['status'], 'no_foreground_seeds')

    def test_score_validation_including_nonfinite_outside_roi(self):
        for value in (np.nan, np.inf, -np.inf, -1):
            score = self.score.copy()
            score[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.plan(score=score)
        for score in (None, self.score[:, :-1], self.score[..., None],
                      self.score.astype(bool), self.score.astype(complex),
                      self.score.astype(object)):
            with self.subTest(dtype=getattr(score, 'dtype', None)), self.assertRaises(ValueError):
                self.plan(score=score)

    def test_valid_and_confidence_validation(self):
        for valid in (None, self.valid.astype(np.uint8), self.valid[:, :-1]):
            with self.assertRaises(ValueError):
                self.plan(valid=valid)
        for confidence in (None, self.confidence.astype(float), self.confidence[:, :-1],
                           np.full(self.shape, 254, np.uint8)):
            with self.assertRaises(ValueError):
                self.plan(confidence=confidence)
        for valid_value, confidence_value in ((True, 0), (False, 128), (False, 255)):
            valid, confidence = self.valid.copy(), self.confidence.copy()
            valid[0, 0], confidence[0, 0] = valid_value, confidence_value
            with self.assertRaises(ValueError):
                self.plan(valid=valid, confidence=confidence)

    def test_detection_shape_mode_and_empty_list_validation(self):
        for box in (None, [40, 30, 80, 70], DetectionBox(40, 30, 40, 70),
                    DetectionBox(40.0, 30, 80, 70), DetectionBox(True, 30, 80, 70)):
            with self.assertRaises(ValueError):
                self.plan([box], mode='box')
        for shape in ((0, 120), (100, -1), (100,), (100, 120, 3), [100, 120], (True, 120)):
            with self.assertRaises(ValueError):
                plan_prompts(shape, [], mode='box')
        with self.assertRaises(ValueError):
            self.plan(mode='unsupported')
        with self.assertRaises(ValueError):
            plan_prompts(self.shape, (self.box,), mode='box')
        self.assertEqual(self.plan([]), [])
        self.assertEqual(plan_prompts(self.shape, [], mode='box'), [])
        with self.assertRaises(ValueError):
            self.plan([], score=None)

    def test_component_points_match_connected_component_not_just_box(self):
        self.score[55:60, 65:70] = 4
        record = self.plan()[0]
        _, labels = cv2.connectedComponents((self.score >= 3).astype(np.uint8), connectivity=8)
        for task in record['prompts']:
            self.assertTrue(all(labels[y, x] == task['component_id'] for x, y in
                                self.global_points(record, task, 1)))


if __name__ == '__main__':
    unittest.main()