"""Synthetic-only tests for independent, frozen box-prediction evaluation."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from reference_anomaly.box_evaluation import (
    evaluate_prediction, pixel_metrics, rasterize_labelme,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_image(path, image):
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    path.write_bytes(encoded.tobytes())


class BoxEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pred = self.root / 'prediction'
        self.pred.mkdir()
        self.image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        self.mask = np.zeros((6, 8), np.uint8)
        self.mask[1:3, 1:3] = 255
        self.valid = np.full((6, 8), 255, np.uint8)
        self.target = self.root / 'original.png'
        save_image(self.target, self.image)
        self.artifacts = dict(mask='prediction_mask.png', valid='comparison_valid.png',
                              input='input.png', overlay='prediction_overlay.png',
                              seeds='seeds.png', contours='contours.json')
        for key in ('input', 'overlay', 'seeds'):
            save_image(self.pred / self.artifacts[key], self.image)
        save_image(self.pred / self.artifacts['mask'], self.mask)
        save_image(self.pred / self.artifacts['valid'], self.valid)
        (self.pred / 'contours.json').write_text('[]', encoding='utf-8')
        self.manifest = dict(
            format_version=1, kind='box_graphcut_prediction', image_shape=[6, 8],
            artifacts=self.artifacts, artifact_sha256={},
            source={'target': {'path': str(self.target), 'sha256': digest(self.target)}},
            config={}, boxes=[],
        )
        self.freeze()
        self.gt = self.root / 'synthetic_gt.json'
        self.annotation = dict(imageHeight=6, imageWidth=8, imagePath='original.png',
                               imageData=None, shapes=[self.rectangle()])
        self.write_gt()

    @staticmethod
    def rectangle(**changes):
        return dict(label='defect', shape_type='rectangle', points=[[1, 1], [2, 2]],
                    flags={}, **changes)

    def freeze(self):
        self.manifest['artifact_sha256'] = {
            key: digest(self.pred / name) for key, name in self.artifacts.items()
        }
        (self.pred / 'prediction.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def write_gt(self):
        self.gt.write_text(json.dumps(self.annotation), encoding='utf-8')

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.pred.iterdir()}

    def test_pixel_arithmetic(self):
        result = pixel_metrics(np.array([[1, 1], [0, 0]], bool),
                               np.array([[1, 0], [1, 0]], bool))
        for key in ('TP', 'FP', 'FN', 'TN'):
            self.assertEqual(result[key], 1)
        for key in ('Precision', 'Recall', 'Dice'):
            self.assertEqual(result[key], .5)
        self.assertAlmostEqual(result['IoU'], 1 / 3)

    def test_empty_masks_have_finite_explicit_scores(self):
        empty, full = np.zeros((2, 2), bool), np.ones((2, 2), bool)
        for pred, gt, expected in ((empty, empty, (1., 1., 1., 1.)),
                                   (empty, full, (0., 0., 0., 0.)),
                                   (full, empty, (0., 0., 0., 0.))):
            result = pixel_metrics(pred, gt)
            self.assertEqual(tuple(result[k] for k in ('Precision', 'Recall', 'IoU', 'Dice')),
                             expected)
        with self.assertRaises(ValueError):
            pixel_metrics(empty, np.zeros((1, 1), bool))

    def test_polygon_rectangle_union_and_ignore(self):
        annotation = dict(self.annotation, shapes=[self.rectangle(),
            dict(label='other', shape_type='polygon', points=[[4, 1], [6, 1], [4, 3]]),
            dict(label='ignored', shape_type='rectangle', points=[[0, 0], [7, 5]],
                 flags={'ignore': True})])
        mask = rasterize_labelme(annotation, (6, 8))
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(int(mask.sum()), 10)  # Inclusive OpenCV endpoints.
        self.assertTrue(mask[3, 4])
        self.assertFalse(mask[0, 0])

    def test_malformed_shapes_are_not_silently_discarded(self):
        bad_shapes = [
            dict(label='x', shape_type='circle', points=[[1, 1], [2, 2]]),
            dict(label='x', shape_type='polygon', points=[[1, 1], [2, 2]]),
            dict(label='x', shape_type='polygon', points=[[1, 1], [2, 2], [3, 3]]),
            dict(label='x', shape_type='rectangle', points=[[1, 1]]),
            dict(label='x', shape_type='rectangle', points=[[1, 1], [1, 3]]),
            dict(label='x', shape_type='rectangle', points=[[1, 1], [float('nan'), 3]]),
            dict(label='x', shape_type='rectangle', points=[['1', 1], [2, 3]]),
            dict(label='x', shape_type='rectangle', points=[[-1, 1], [2, 3]]),
            dict(label='x', shape_type='point', points=[[1, 1]], flags={'ignore': True}),
            None,
        ]
        for shape in bad_shapes:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                rasterize_labelme(dict(self.annotation, shapes=[shape]), (6, 8))

    def test_evaluation_saves_provenance_and_never_changes_prediction(self):
        before = self.snapshot()
        result = evaluate_prediction(self.pred, self.gt)
        self.assertEqual(result['TP'], 4)
        self.assertEqual(result['IoU'], 1.)
        self.assertEqual(result['prediction_manifest_sha256'], digest(self.pred / 'prediction.json'))
        self.assertEqual(result['gt_sha256'], digest(self.gt))
        self.assertEqual(result['image_match'], 'decoded_pixels')
        output = self.root / 'prediction_evaluation'
        self.assertTrue((output / 'gt_mask.png').is_file())
        self.assertTrue((output / 'comparison.png').is_file())
        comparison = cv2.imdecode(np.frombuffer((output / 'comparison.png').read_bytes(),
                               np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(comparison[1, 1], [0, 255, 0])
        self.assertEqual(json.loads((output / 'metrics.json').read_text()), result)
        self.assertEqual(self.snapshot(), before)
        with self.assertRaises(ValueError):
            evaluate_prediction(self.pred, self.gt)  # Never overwrite an earlier evaluation.

    def test_reference_uncovered_gt_still_counts_as_false_negative(self):
        self.valid[:, 2:] = 0
        self.mask[1:3, 2] = 0
        save_image(self.pred / self.artifacts['valid'], self.valid)
        save_image(self.pred / self.artifacts['mask'], self.mask)
        self.freeze()
        result = evaluate_prediction(self.pred, self.gt)
        self.assertEqual(result['TP'], 2)
        self.assertEqual(result['FN'], 2)
        self.assertEqual(result['Recall'], .5)
        self.assertEqual(result['gt_comparable_pixels'], 2)
        self.assertEqual(result['gt_uncovered_pixels'], 2)
        self.assertEqual(result['gt_coverage_fraction'], .5)

    def test_missing_gt_image_allows_name_dimensions_only(self):
        self.target.unlink()
        self.annotation['imagePath'] = 'missing/ORIGINAL.PNG'
        self.write_gt()
        result = evaluate_prediction(self.pred / 'prediction.json', self.gt)
        self.assertEqual(result['image_match'], 'name_dimensions_only')

    def test_gt_image_identity_failures(self):
        for changes in (dict(imageWidth=9), dict(imageHeight=True),
                        dict(imagePath='different.png'), dict(imageData='embedded-data')):
            with self.subTest(changes=changes):
                self.annotation.update(changes)
                self.write_gt()
                with self.assertRaises(ValueError):
                    evaluate_prediction(self.pred, self.gt)
                self.annotation = dict(imageHeight=6, imageWidth=8, imagePath='original.png',
                                       imageData=None, shapes=[self.rectangle()])
        self.write_gt()
        save_image(self.target, np.zeros_like(self.image))
        with self.assertRaises(ValueError):
            evaluate_prediction(self.pred, self.gt)
        self.assertFalse((self.root / 'prediction_evaluation').exists())

    def test_embedded_copy_and_external_identity_are_both_reported(self):
        for equal in (True, False):
            image = self.image if equal else np.zeros_like(self.image)
            ok, encoded = cv2.imencode('.png', image)
            self.assertTrue(ok)
            self.annotation['imageData'] = base64.b64encode(encoded).decode('ascii')
            self.write_gt()
            result = evaluate_prediction(self.pred, self.gt, self.root / f'embedded_{equal}')
            self.assertEqual(result['embedded_image_matches_input'], equal)
            self.assertEqual(result['image_match'], 'decoded_pixels')
            self.assertEqual(bool(result['warnings']), not equal)
        self.target.unlink()
        with self.assertRaisesRegex(ValueError, 'no external image'):
            evaluate_prediction(self.pred, self.gt)
        ok, encoded = cv2.imencode('.png', self.image)
        self.annotation['imageData'] = base64.b64encode(encoded).decode('ascii')
        self.write_gt()
        result = evaluate_prediction(self.pred, self.gt)
        self.assertEqual(result['image_match'], 'embedded_decoded_pixels')

    def test_all_artifact_hashes_checked_before_gt_open(self):
        original_read = Path.read_bytes
        for name in self.artifacts.values():
            path = self.pred / name
            data = path.read_bytes()
            path.write_bytes(data + b'tamper')
            def guarded_read(path):
                if path == self.gt:
                    self.fail('GT opened before frozen artifact verification')
                return original_read(path)
            with self.subTest(name=name), patch.object(Path, 'read_bytes', guarded_read):
                with self.assertRaisesRegex(ValueError, '(?i)hash|sha256'):
                    evaluate_prediction(self.pred, self.gt)
            path.write_bytes(data)

    def test_output_overlap_and_existing_directory_rejected(self):
        before = self.snapshot()
        for output in (self.pred, self.pred / 'evaluation', self.root,
                       self.gt, self.target):
            with self.subTest(output=output), self.assertRaises(ValueError):
                evaluate_prediction(self.pred, self.gt, out=output)
        # A missing original source must not be recreated as an evaluation artifact.
        output = self.root / 'new_eval'
        self.manifest['source']['xml'] = {
            'path': str(output / 'metrics.json'), 'sha256': '0' * 64}
        self.freeze()
        with self.assertRaises(ValueError):
            evaluate_prediction(self.pred, self.gt, out=output)
        self.manifest['source'].pop('xml')
        self.freeze()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(output.exists())

    def test_manifest_validation_and_path_escape(self):
        original = json.loads(json.dumps(self.manifest))
        for change in (dict(format_version=2), dict(kind='other'),
                       dict(image_shape=[5, 8]), dict(artifact_sha256={})):
            self.manifest.update(change)
            (self.pred / 'prediction.json').write_text(json.dumps(self.manifest))
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_prediction(self.pred, self.gt)
            self.manifest = json.loads(json.dumps(original))
        self.artifacts = self.manifest['artifacts']
        self.artifacts['input'] = '../original.png'
        self.freeze()
        with self.assertRaises(ValueError):
            evaluate_prediction(self.pred, self.gt)

    def test_hash_missing_for_extra_artifact_is_rejected_before_gt(self):
        self.manifest['artifacts']['extra'] = 'extra.bin'
        (self.pred / 'prediction.json').write_text(json.dumps(self.manifest))
        self.gt.unlink()
        with self.assertRaisesRegex(ValueError, 'sha256'):
            evaluate_prediction(self.pred, self.gt)

    def test_gt_copy_must_match_pixels_not_encoded_file_hash(self):
        copy_dir = self.root / 'image_copy'
        copy_dir.mkdir()
        copy = copy_dir / self.target.name
        ok, encoded = cv2.imencode('.png', self.image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        self.assertTrue(ok)
        copy.write_bytes(encoded.tobytes())
        self.assertNotEqual(digest(copy), digest(self.target))
        self.annotation['imagePath'] = 'image_copy\\original.png'
        self.write_gt()
        self.assertEqual(evaluate_prediction(self.pred, self.gt)['image_match'], 'decoded_pixels')

    def test_comparison_colors_include_uncovered_fp_and_fn(self):
        self.mask[:] = 0
        self.mask[1, 1] = 255  # TP
        self.mask[5, 7] = 255  # FP, despite invalid reference
        self.valid[:] = 0
        save_image(self.pred / self.artifacts['mask'], self.mask)
        save_image(self.pred / self.artifacts['valid'], self.valid)
        self.freeze()
        result = evaluate_prediction(self.pred, self.gt)
        self.assertEqual((result['TP'], result['FP'], result['FN'], result['TN']), (1, 1, 3, 43))
        self.assertEqual(result['gt_coverage_fraction'], 0.)
        comparison = cv2.imdecode(np.frombuffer(
            (self.root / 'prediction_evaluation' / 'comparison.png').read_bytes(), np.uint8),
            cv2.IMREAD_UNCHANGED)
        for point, color in (((1, 1), [0, 255, 0]), ((5, 7), [0, 0, 255]),
                             ((1, 2), [255, 0, 0]), ((0, 0), [0, 0, 0])):
            np.testing.assert_array_equal(comparison[point], color)

    def test_unit_binary_prediction_is_supported(self):
        save_image(self.pred / self.artifacts['mask'], self.mask // 255)
        self.freeze()
        self.assertEqual(evaluate_prediction(self.pred, self.gt)['Dice'], 1.)

    def test_invalid_frozen_mask_values_and_shape(self):
        for key, image in (('mask', np.full((6, 8), 42, np.uint8)),
                           ('valid', np.full((6, 8), 1, np.uint8)),
                           ('valid', np.zeros((5, 8), np.uint8)),
                           ('mask', self.image)):
            with self.subTest(key=key, shape=image.shape):
                save_image(self.pred / self.artifacts[key], image)
                self.freeze()
                with self.assertRaises(ValueError):
                    evaluate_prediction(self.pred, self.gt)
                save_image(self.pred / self.artifacts['mask'], self.mask)
                save_image(self.pred / self.artifacts['valid'], self.valid)

    def test_empty_gt_and_cli(self):
        self.annotation['shapes'] = []
        self.write_gt()
        output = self.root / 'cli_evaluation'
        completed = subprocess.run(
            [sys.executable, '-m', 'reference_anomaly.box_evaluation',
             '--prediction', str(self.pred), '--gt', str(self.gt), '--out', str(output)],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads((output / 'metrics.json').read_text())
        self.assertEqual(result['FP'], 4)
        self.assertEqual(result['gt_coverage_fraction'], 1.)


if __name__ == '__main__':
    unittest.main()