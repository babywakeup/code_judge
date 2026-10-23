"""Deterministic registration, annotation IO and output safety regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig, run_config
from periodic_template_matcher._internal.registration import register, select_templates
from periodic_template_matcher._internal.annotations import load_labels
from periodic_template_matcher._internal.workflow import unique_templates
from periodic_template_matcher.utils import read_image, warp_labels, write_image
from tests.helpers import texture


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = MatcherConfig(max_side=200, angles=(0.,), scales=(1.,),
                          refine_candidates=3, peaks_per_hypothesis=3, ecc_iterations=35)

    def test_translation_and_rotation_direction(self):
        source = texture()
        H = np.vstack([cv2.getRotationMatrix2D((100, 80), .8, 1.01), [0, 0, 1]])
        H[:2, 2] += [5, -4]
        target = cv2.warpPerspective(source, H, (200, 160))
        result = register(source, target, self.cfg)
        actual = np.asarray(result['matrix_template_to_target'])
        points = np.array([[[40., 40.], [150., 110.], [100., 80.]]], np.float32)
        error = np.linalg.norm(cv2.perspectiveTransform(points, H) -
                               cv2.perspectiveTransform(points, actual), axis=2)
        self.assertTrue(result['geometry_accepted'])
        self.assertLess(float(error.max()), 1.5)

    def test_ranking_and_equal_template_rejection(self):
        source = texture()
        result = select_templates([texture(99), source], source, self.cfg)
        self.assertEqual(result['best_template_index'], 1)
        self.assertTrue(result['accepted'])
        ambiguous = select_templates([source, source], source, self.cfg)
        self.assertEqual(ambiguous['status'], 'templates_too_close')
        self.assertFalse(ambiguous['accepted'])

    def test_blank_rejected(self):
        blank = np.zeros((100, 100, 3), np.uint8)
        result = select_templates([blank], blank, self.cfg)
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['best_template_index'])

    def test_integer_labels_valid_background_and_direction(self):
        labels = np.zeros((40, 50), np.uint16)
        labels[10:15, 12:18] = 1001
        valid = np.zeros_like(labels)
        valid[5:25, 5:30] = 1
        H = np.array([[1., 0, 7], [0, 1., 3], [0, 0, 1.]])
        mapped, known = warp_labels(labels, H, (45, 60), valid)
        self.assertEqual(mapped[13, 19], 1001)
        self.assertTrue(known[8, 12])
        self.assertEqual(mapped[8, 12], 0)  # Real background remains known.
        self.assertFalse(known[0, 0])
        self.assertEqual(set(np.unique(mapped)), {0, 1001})
        self.assertEqual(mapped.dtype, np.uint16)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write_image(self.root / 'template.png', texture())
        write_image(self.root / 'target.png', texture())

    def config(self, **extra):
        spec = {'target': 'target.png', 'templates': [{'image': 'template.png', 'name': '<template>'}],
                'options': {'max_side': 200, 'angles': [0], 'scales': [1], 'refine_candidates': 2}}
        spec.update(extra)
        path = self.root / 'config.json'
        path.write_text(json.dumps(spec), encoding='utf-8')
        return path

    def test_duplicate_content_collapsed_and_conflicts_rejected(self):
        write_image(self.root / 'copy.png', texture())
        specs = [{'image': 'template.png'}, {'image': 'copy.png'}]
        unique, aliases = unique_templates(specs, lambda p: self.root / p)
        self.assertEqual(len(unique), 1)
        self.assertEqual(len(aliases[0]), 2)
        specs[1]['labels'] = 'different.png'
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            unique_templates(specs, lambda p: self.root / p)

    def test_labelme_classes_and_invalid_shapes(self):
        annotation = {'imageHeight': 160, 'imageWidth': 200, 'shapes': [
            {'label': 'wire', 'shape_type': 'polygon', 'points': [[20, 20], [40, 20], [30, 40]]},
            {'label': 'via', 'shape_type': 'circle', 'points': [[70, 60], [76, 60]]},
            {'label': 'wire', 'shape_type': 'rectangle', 'points': [[120, 100], [110, 90]]}]}
        path = self.root / 'annotation.json'
        path.write_text(json.dumps(annotation), encoding='utf-8')
        spec = {'labelme': 'annotation.json', 'class_map': {'wire': 257, 'via': 2}}
        labels = load_labels(spec, lambda p: self.root / p, (160, 200))
        self.assertEqual(labels[25, 30], 257)
        self.assertEqual(labels[60, 70], 2)
        self.assertEqual(labels[95, 115], 257)
        self.assertEqual(labels[0, 0], 0)
        spec['class_map'] = {'wire': 1}
        with self.assertRaisesRegex(ValueError, 'Unmapped'):
            load_labels(spec, lambda p: self.root / p, (160, 200))
        annotation['shapes'][0]['shape_type'] = 'line'
        path.write_text(json.dumps(annotation), encoding='utf-8')
        spec['class_map'] = {'wire': 1, 'via': 2}
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            load_labels(spec, lambda p: self.root / p, (160, 200))

    def test_real_transfer_then_demo_cleans_stale_outputs(self):
        labels = np.zeros((160, 200), np.uint16)
        labels[35:65, 40:70] = 321
        write_image(self.root / 'labels.png', labels)
        known = np.zeros((160, 200), np.uint8)
        known[20:80, 20:90] = 255
        write_image(self.root / 'known.png', known)
        out = self.root / 'out'
        config = self.config(templates=[{'image': 'template.png', 'labels': 'labels.png',
                                         'annotation_valid': 'known.png', 'name': '<template>'}])
        result = run_config(config, out)
        self.assertTrue(result['labels_exported'])
        mapped = read_image(out / 'mapped_labels.png', cv2.IMREAD_UNCHANGED)
        valid = read_image(out / 'label_valid.png', cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mapped[50, 55], 321)
        self.assertEqual(valid[0, 0], 0)
        self.assertEqual(valid[25, 25], 255)
        self.assertIn('&lt;template&gt;', (out / 'report.html').read_text())
        config = self.config(demo_annotations=True)
        result = run_config(config, out)
        self.assertFalse(result['labels_exported'])
        self.assertTrue(result['demo_labels_exported'])
        self.assertFalse((out / 'mapped_labels.png').exists())
        self.assertFalse((out / 'label_valid.png').exists())
        self.assertTrue((out / 'demo_mapped_labels.png').exists())
        # Rejected reruns must not leave even illustrative maps behind.
        write_image(self.root / 'target.png', np.zeros((160, 200, 3), np.uint8))
        result = run_config(config, out)
        self.assertFalse(result['accepted'])
        self.assertFalse((out / 'demo_mapped_labels.png').exists())
        self.assertTrue((out / 'report.html').exists())

    def test_instance_mode_blocks_periodic_label_export(self):
        labels = np.ones((160, 200), np.uint8)
        write_image(self.root / 'labels.png', labels)
        config = self.config(label_mode='instance', templates=[{'image': 'template.png', 'labels': 'labels.png'}])
        # Exercise the IO safety gate with explicit periodic/ambiguous evidence.
        original = select_templates([texture()], texture(), MatcherConfig(max_side=200, angles=(0,), scales=(1,)))
        original.update(accepted=False, status='absolute_cell_unverified', absolute_mapping_accepted=False)
        original['matches'][0].update(periodic_detected=True, absolute_cell_verified=False)
        with patch('periodic_template_matcher.matcher.select_templates', return_value=original):
            result = run_config(config, self.root / 'out')
        self.assertFalse(result['labels_exported'])
        self.assertEqual(result['annotation_status'], 'blocked_by_match')
        self.assertFalse((self.root / 'out' / 'mapped_labels.png').exists())


if __name__ == '__main__':
    unittest.main()