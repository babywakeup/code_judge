"""Synthetic graph-cut contract tests; no actual defect annotations are read."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher.utils import read_image, write_image
from reference_anomaly.box_graphcut import (
    BoxGraphCutConfig, DetectionBox, binary_cut, predict_boxes, read_boxes,
)
from reference_anomaly.box_prediction import run_prediction


class GraphCutTests(unittest.TestCase):
    def test_terminal_direction_and_smoothness(self):
        # Source reachable means foreground: low D_fg must select foreground.
        fg = binary_cut(np.array([[.1, 4.]]), np.array([[4., .1]]),
                        np.zeros((1, 1)), np.zeros((0, 2)))
        np.testing.assert_array_equal(fg, [[True, False]])
        coupled = binary_cut(np.array([[.1, 4.]]), np.array([[5., .1]]),
                             np.full((1, 1), 20.), np.zeros((0, 2)))
        self.assertTrue(coupled.all())

    def test_grid_energy_matches_exhaustive_search(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            fg, bg = rng.integers(0, 20, (2, 2, 3))/4
            horizontal = rng.integers(0, 12, (2, 2))/4
            vertical = rng.integers(0, 12, (1, 3))/4
            def energy(labels):
                return (np.where(labels, fg, bg).sum()
                        + (horizontal*(labels[:, 1:] != labels[:, :-1])).sum()
                        + (vertical*(labels[1:] != labels[:-1])).sum())
            possibilities = [np.array([(bits >> k) & 1 for k in range(6)], bool).reshape(2, 3)
                             for bits in range(64)]
            result = binary_cut(fg, bg, horizontal, vertical)
            self.assertEqual(energy(result), min(energy(labels) for labels in possibilities))

    @staticmethod
    def pair():
        reference = np.full((80, 100, 3), 115, np.uint8)
        return reference.copy(), reference, np.ones((80, 100), bool), np.zeros((80, 100), np.float32)

    def test_loose_box_bright_dark_colored_multiple_components(self):
        for color in ((245, 245, 245), (10, 10, 10), (200, 30, 150)):
            image, reference, valid, score = self.pair()
            truth = np.zeros_like(valid)
            truth[24:38, 30:43] = True
            truth[45:57, 55:68] = True
            image[truth] = color
            score[truth] = 5
            before = image.copy()
            result = predict_boxes(image, reference, valid, score, [DetectionBox(20, 15, 80, 65)])
            self.assertEqual(result.status, 'ok')
            self.assertGreater((result.mask & truth).sum() / truth.sum(), .98)
            self.assertLess((result.mask & ~truth).sum(), 10)
            np.testing.assert_array_equal(before, image)

    def test_low_heat_in_box_is_not_hard_background_and_growth_crosses_box(self):
        image, reference, valid, score = self.pair()
        image[25:55, 30:62] = (20, 40, 180)
        score[25:55, 30:62] = 1.4
        score[33:40, 35:42] = 5
        result = predict_boxes(image, reference, valid, score, [DetectionBox(32, 20, 58, 60)])
        self.assertTrue(result.mask[45, 50])
        self.assertTrue(result.mask[40, 30])
        self.assertIn('touches_box', result.boxes[0]['warnings'])

    def test_different_colors_in_one_box_and_unreliable_seeds(self):
        image, reference, valid, score = self.pair()
        image[24:38, 30:43] = (230, 230, 230)
        image[45:57, 55:68] = (15, 20, 70)
        score[24:38, 30:43] = score[45:57, 55:68] = 5
        confidence = np.full(valid.shape, 255, np.uint8)
        confidence[24:38, 30:43] = 128
        box = DetectionBox(20, 15, 80, 65)
        both = predict_boxes(image, reference, valid, score, [box])
        self.assertTrue(both.mask[28, 35])
        self.assertTrue(both.mask[50, 60])
        self.assertTrue(both.mask[both.foreground_seeds].all())
        self.assertFalse(both.mask[both.background_seeds].any())
        seam = predict_boxes(image, reference, valid, score, [box], confidence=confidence)
        self.assertFalse(seam.foreground_seeds[24:38, 30:43].any())
        self.assertTrue(seam.foreground_seeds[45:57, 55:68].any())

    def test_abstention_and_invalid_reference(self):
        image, reference, valid, score = self.pair()
        box = DetectionBox(20, 15, 80, 65)
        empty = predict_boxes(image, reference, valid, score, [box])
        self.assertEqual(empty.boxes[0]['status'], 'no_foreground_seeds')
        self.assertFalse(empty.mask.any())
        score[:] = 5
        invalid = predict_boxes(image, reference, np.zeros_like(valid), score, [box])
        self.assertEqual(invalid.boxes[0]['status'], 'no_reliable_reference')
        self.assertFalse(invalid.mask.any())
        no_bg = predict_boxes(image, reference, valid, score, [box])
        self.assertEqual(no_bg.boxes[0]['status'], 'no_background_seeds')

    def test_multibox_edges_invalid_pixels_and_resource_limit(self):
        image, reference, valid, score = self.pair()
        image[0:15, 0:16] = 240
        image[45:55, 70:80] = 10
        score[0:15, 0:16] = 5
        score[45:55, 70:80] = 5
        valid[7:10, 7:10] = False
        boxes = [DetectionBox(-5, -4, 25, 25), DetectionBox(65, 40, 85, 60), DetectionBox(150, 0, 170, 20)]
        result = predict_boxes(image, reference, valid, score, boxes)
        self.assertTrue(result.mask[2, 2])
        self.assertTrue(result.mask[49, 74])
        self.assertFalse(result.mask[~valid].any())
        self.assertEqual(result.boxes[2]['status'], 'outside_image')
        self.assertIn('touches_roi', result.boxes[0]['warnings'])
        limited = predict_boxes(image, reference, valid, score, boxes[:1], config=BoxGraphCutConfig(max_roi_pixels=10))
        self.assertEqual(limited.boxes[0]['status'], 'roi_too_large')

    def test_input_validation(self):
        image, reference, valid, score = self.pair()
        for bad in (np.full_like(score, np.nan), -np.ones_like(score), score[:, :-1]):
            with self.assertRaises(ValueError):
                predict_boxes(image, reference, valid, bad, [])
        for kwargs in ({'seed_threshold': float('nan')}, {'context_fraction': -1}, {'max_components': 0}):
            with self.assertRaises(ValueError):
                BoxGraphCutConfig(**kwargs).validate()


class BoxFileTests(unittest.TestCase):
    def test_xml_coordinate_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'boxes.xml'
            path.write_text('<annotation><size><width>100</width><height>80</height></size>'
                            '<object><name>defect</name><bndbox><xmin>1</xmin><ymin>2</ymin>'
                            '<xmax>10</xmax><ymax>20</ymax></bndbox></object></annotation>')
            box = read_boxes(path, (80, 100))[0]
            self.assertEqual((box.x0, box.y0, box.x1, box.y1), (0, 1, 10, 20))
            self.assertEqual(read_boxes(path, (80, 100), coordinates='zero_half_open')[0].x0, 1)
            with self.assertRaises(ValueError):
                read_boxes(path, (81, 100))
            path.write_text('<annotation><object><bndbox><xmin>nan</xmin></bndbox></object></annotation>')
            with self.assertRaises(ValueError):
                read_boxes(path, (80, 100))

    def test_prediction_config_ignores_gt_freezes_outputs_and_protects_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            result = root / 'result'
            anomaly = result / 'anomaly'
            anomaly.mkdir(parents=True)
            image, reference, valid, score = GraphCutTests.pair()
            image[30:45, 35:50] = 240
            score[30:45, 35:50] = 5
            for name, array in [('input.png', image), ('reference.png', reference), ('valid.png', valid.astype(np.uint8)*255)]:
                write_image(result / name, array)
            write_image(root / 'target.png', image)
            (result / 'result.json').write_text(json.dumps(dict(format_version=1, accepted=True,
                artifacts=dict(input='input.png', supposed_to_be_good='reference.png'))))
            write_image(anomaly / 'valid.png', valid.astype(np.uint8)*255)
            write_image(anomaly / 'confidence.png', valid.astype(np.uint8)*255)
            np.savez_compressed(anomaly / 'scores.npz', salience_score=score)
            (anomaly / 'anomaly.json').write_text(json.dumps(dict(artifacts=dict(
                scores='scores.npz', valid='valid.png', confidence='confidence.png'),
                source=dict(manifest=str(result / 'result.json')))))
            (root / 'boxes.xml').write_text('<annotation><object><bndbox><xmin>20</xmin><ymin>20</ymin>'
                '<xmax>65</xmax><ymax>60</ymax></bndbox></object></annotation>')
            config = root / 'config.json'
            config.write_text(json.dumps(dict(target='target.png', box='boxes.xml', GT_labelme='DO_NOT_READ.json')))
            original = Path.open
            def guarded(path, *args, **kwargs):
                if path.name == 'DO_NOT_READ.json':
                    raise AssertionError('Prediction accessed GT')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'open', guarded):
                manifest = run_prediction(result, config_path=config)
            metadata = json.loads(manifest.read_text())
            self.assertEqual(metadata['status'], 'ok')
            self.assertNotIn('GT_labelme', json.dumps(metadata))
            self.assertTrue(read_image(manifest.parent / metadata['artifacts']['mask']).any())
            from reference_anomaly.box_evaluation import _frozen_prediction
            _frozen_prediction(manifest)
            with self.assertRaises(ValueError):
                run_prediction(result, config_path=config)
            with self.assertRaises(ValueError):
                run_prediction(result, config_path=config, output_dir=root)
            config.write_text(json.dumps(dict(target='target.png', box='boxes.xml', GT_labelme='OTHER.json')))
            second = run_prediction(result, config_path=config, output_dir=root / 'second')
            second_meta = json.loads(second.read_text())
            self.assertEqual(metadata['artifact_sha256'], second_meta['artifact_sha256'])


if __name__ == '__main__':
    unittest.main()