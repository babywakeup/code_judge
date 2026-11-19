"""Synthetic SAM3 integration tests: no weights, CUDA, network or actual GT."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from reference_anomaly.box_prediction import _write, _sha
from reference_anomaly.box_evaluation import _frozen_prediction
from reference_anomaly.sam3_backend import OfficialSam3Backend, resolve_checkpoint
from reference_anomaly.sam3_prediction import run_prediction, select_candidate


class SyntheticBackend:
    provenance = {'backend': 'synthetic_test_double_not_SAM3'}

    def set_image(self, image):
        self.rgb = image.copy()

    def predict(self, *, box_xyxy, points_xy, point_labels):
        mask = self.rgb[..., 0] > 200
        return np.stack((mask, np.ones_like(mask))), np.array([.7, .99])


class Sam3PredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.result = self.root / 'result'
        self.anomaly = self.result / 'anomaly'
        self.anomaly.mkdir(parents=True)
        self.weight = self.root / 'sam3.pt'
        self.weight.write_bytes(b'synthetic checkpoint placeholder, never loaded')
        self.image = np.full((80, 100, 3), 90, np.uint8)
        # Red in BGR; synthetic backend recognizes RED in converted RGB.
        self.image[30:45, 35:50] = (20, 30, 240)
        self.valid = np.full((80, 100), 255, np.uint8)
        self.valid[34:36, 40:42] = 0  # SAM may infer across unknown reference.
        self.score = np.zeros((80, 100), np.float32)
        self.score[30:45, 35:50] = 5
        _write(self.root / 'target.png', self.image)
        _write(self.result / 'input.png', self.image)
        _write(self.anomaly / 'valid.png', self.valid)
        _write(self.anomaly / 'confidence.png', self.valid)
        np.savez_compressed(self.anomaly / 'scores.npz', salience_score=self.score)
        self.manifest = self.result / 'result.json'
        self.manifest.write_text(json.dumps(dict(format_version=1, accepted=True, artifacts={'input': 'input.png'})))
        (self.anomaly / 'anomaly.json').write_text(json.dumps(dict(
            source={'manifest': str(self.manifest)},
            artifacts=dict(scores='scores.npz', valid='valid.png', confidence='confidence.png'))))
        (self.root / 'boxes.xml').write_text('<annotation><object><bndbox><xmin>20</xmin><ymin>20</ymin>'
            '<xmax>65</xmax><ymax>60</ymax></bndbox></object></annotation>')
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps(dict(target='target.png', box='boxes.xml', GT_labelme='never_open_gt.json')))

    def run_prediction(self, **kwargs):
        return run_prediction(self.result, checkpoint=self.weight, config_path=self.config, **kwargs)

    def test_preview_never_constructs_model_or_gt_and_is_not_a_prediction(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == 'never_open_gt.json':
                self.fail('SAM preview opened GT')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded), patch(
                'reference_anomaly.sam3_prediction.OfficialSam3Backend', side_effect=AssertionError('model loaded')):
            path = self.run_prediction(prepare_only=True)
        doc = json.loads(path.read_text())
        self.assertEqual(doc['kind'], 'sam3_prompt_preparation')
        self.assertFalse(doc['model_executed'])
        self.assertNotIn('mask', doc['artifacts'])
        self.assertNotIn('never_open_gt', json.dumps(doc))
        for key, name in doc['artifacts'].items():
            self.assertEqual(_sha(path.parent / name), doc['artifact_sha256'][key])
        with self.assertRaises(ValueError):
            _frozen_prediction(path)  # Cannot evaluate a prompt preview as a model prediction.
        with self.assertRaises(ValueError):
            self.run_prediction(prepare_only=True)  # Existing output is frozen.

    def test_injected_backend_rgb_full_resolution_and_unknown_not_clipped(self):
        backend = SyntheticBackend()
        path = self.run_prediction(backend=backend)
        doc = json.loads(path.read_text())
        self.assertEqual(doc['kind'], 'sam3_prediction')
        self.assertEqual(doc['status'], 'ok')
        self.assertEqual(doc['predicted_pixels'], 225)
        self.assertEqual(doc['predicted_without_reference_pixels'], 4)
        self.assertIn('synthetic_test_double', doc['backend']['backend'])
        _, _, _, _, masks, _ = _frozen_prediction(path)
        self.assertTrue(masks[0][34:36, 40:42].all())
        record = doc['boxes'][0]
        x0, y0, x1, y1 = record['roi_xyxy']
        self.assertEqual(backend.rgb.shape, (y1-y0, x1-x0, 3))
        np.testing.assert_array_equal(backend.rgb[32-y0, 38-x0], [240, 30, 20])
        self.assertEqual(record['prompts'][0]['selected_candidate'], 0)
        self.assertTrue((path.parent / 'candidates.npz').is_file())

    def test_box_mode_uses_no_scores_or_confidence(self):
        (self.anomaly / 'scores.npz').unlink()
        (self.anomaly / 'confidence.png').unlink()
        path = self.run_prediction(mode='box', backend=SyntheticBackend())
        doc = json.loads(path.read_text())
        task = doc['boxes'][0]['prompts'][0]
        self.assertEqual(task['point_labels'], [])
        self.assertEqual(task['selected_candidate'], 1)  # Only quality, no GT-driven selection.
        self.assertNotIn('scores', doc['source'])

    def test_transformers_auto_unions_distinct_instances_and_hashes_bundle(self):
        class InstancesBackend(SyntheticBackend):
            candidate_semantics = 'instances'

            def predict(self, **kwargs):
                first = self.rgb[..., 0] > 200
                second = np.zeros_like(first)
                second[0, 0] = True
                return np.stack((first, second)), np.array([.99, .6])

        for name in ('model.safetensors', 'config.json', 'processor_config.json', 'tokenizer.json'):
            (self.root / name).write_bytes(b'synthetic HF bundle, never loaded')
        with patch('reference_anomaly.sam3_prediction.OfficialSam3Backend',
                   side_effect=AssertionError('official package should not be needed')):
            path = self.run_prediction(mode='box', backend=InstancesBackend())
        doc = json.loads(path.read_text())
        self.assertEqual(doc['config']['backend'], 'transformers')
        self.assertEqual(doc['boxes'][0]['prompts'][0]['selected_candidate'], [0, 1])
        self.assertEqual(doc['predicted_pixels'], 226)  # Both disjoint instances, not best-only.
        self.assertTrue(doc['model_executed'])
        self.assertEqual(doc['source']['checkpoint']['sha256'], _sha(self.root / 'model.safetensors'))
        self.assertIn('hf_tokenizer_json', doc['source'])
        self.assertEqual(doc['transformers_adapter_sha256'], _sha(
            Path(__file__).resolve().parents[1] / 'reference_anomaly' / 'sam3_transformers.py'))

    def test_abstention_does_not_claim_model_execution(self):
        (self.anomaly / 'valid.png').unlink()
        (self.anomaly / 'confidence.png').unlink()
        _write(self.anomaly / 'valid.png', np.zeros_like(self.valid))
        _write(self.anomaly / 'confidence.png', np.zeros_like(self.valid))
        with patch('reference_anomaly.sam3_prediction.OfficialSam3Backend',
                   side_effect=AssertionError('abstention must not load a model')):
            path = self.run_prediction()
        doc = json.loads(path.read_text())
        self.assertEqual(doc['status'], 'abstained')
        self.assertFalse(doc['model_executed'])
        self.assertEqual(doc['predicted_pixels'], 0)

    def test_invalid_options_fail_before_model_loading_or_export(self):
        for options in (dict(threshold=float('nan')), dict(mask_threshold=2),
                        dict(device='invalid'), dict(text=''), dict(mode='anomaly_points', text='defect')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_prediction(prepare_only=True, **options)
        self.assertFalse((self.anomaly / 'sam3_anomaly_points_prepared').exists())

    def test_all_conflicting_candidates_abstain(self):
        class BadBackend(SyntheticBackend):
            def predict(self, **kwargs):
                shape = self.rgb.shape[:2]
                return np.stack((np.zeros(shape, bool), np.ones(shape, bool))), np.array([1., 1.])
        path = self.run_prediction(backend=BadBackend())
        doc = json.loads(path.read_text())
        self.assertEqual(doc['status'], 'abstained')
        self.assertEqual(doc['predicted_pixels'], 0)
        self.assertEqual(doc['boxes'][0]['prompts'][0]['status'], 'no_prompt_consistent_candidate')

    def test_protect_outputs_source_mismatch_and_changed_sources(self):
        with self.assertRaises(ValueError):
            self.run_prediction(prepare_only=True, output_dir=self.root)
        (self.root / 'target.png').unlink()
        _write(self.root / 'target.png', np.zeros_like(self.image))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.run_prediction(prepare_only=True)
        (self.root / 'target.png').unlink()
        _write(self.root / 'target.png', self.image)
        weight = self.weight
        class MutatingBackend(SyntheticBackend):
            def predict(self, **kwargs):
                weight.write_bytes(b'changed during inference')
                return super().predict(**kwargs)
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.run_prediction(backend=MutatingBackend())
        self.assertFalse((self.anomaly / 'sam3_anomaly_points').exists())

    def test_source_mismatch_and_bad_coordinates_fail(self):
        metadata = self.anomaly / 'anomaly.json'
        doc = json.loads(metadata.read_text())
        doc['source']['manifest'] = str(self.root / 'different.json')
        metadata.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            self.run_prediction(prepare_only=True)

    def test_checkpoint_formats_and_cpu_preflight(self):
        self.assertEqual(resolve_checkpoint(self.root), self.weight)
        for name in ('model.safetensors', 'sam3.pt.incomplete', 'missing.pt'):
            with self.assertRaises(ValueError):
                resolve_checkpoint(self.root / name)
        class FakeCuda:
            @staticmethod
            def is_available():
                return False
        class FakeTorch:
            cuda = FakeCuda()
        with patch('reference_anomaly.sam3_backend.importlib.import_module', return_value=FakeTorch()):
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                OfficialSam3Backend(self.weight)

    def test_candidate_validation_and_empty_masks(self):
        masks = np.zeros((2, 4, 5), bool)
        self.assertIsNone(select_candidate(masks, [.9, .8], (4, 5), [], [])[0])
        for bad, quality in ((np.full((2, 4, 5), .7), [.9, .8]),
                             (masks[:, :-1], [.9, .8]), (masks, [float('nan'), .8]),
                             (masks, [.9])):
            with self.assertRaises(ValueError):
                select_candidate(bad, quality, (4, 5), [], [])

    def test_official_adapter_local_checkpoint_and_pixel_prompt_contract(self):
        calls = {}
        class FakeModel:
            inst_interactive_predictor = object()

            def predict_inst(self, state, **kwargs):
                calls['state'] = state
                calls['predict'] = kwargs
                return np.ones((1, 4, 6), bool), np.array([.8]), None

        def build(**kwargs):
            calls['builder'] = kwargs
            return FakeModel()

        class Processor:
            def __init__(self, model, **kwargs):
                calls['processor'] = kwargs

            def set_image(self, image):
                calls['image_size'] = image.size
                return {'original_height': image.height, 'original_width': image.width}

        modules = dict(torch=SimpleNamespace(__version__='test', inference_mode=nullcontext), **{
            'sam3.model_builder': SimpleNamespace(build_sam3_image_model=build, __file__=__file__),
            'sam3.model.sam3_image_processor': SimpleNamespace(Sam3Processor=Processor, __file__=__file__),
        })
        with patch('reference_anomaly.sam3_backend.importlib.import_module', side_effect=modules.__getitem__):
            backend = OfficialSam3Backend(self.weight, device='cpu')
            backend.set_image(np.zeros((4, 6, 3), np.uint8))
            backend.predict(box_xyxy=[1, 1, 5, 3], points_xy=[[3, 2]], point_labels=[1])
            self.assertEqual(calls['builder']['checkpoint_path'], str(self.weight))
            self.assertFalse(calls['builder']['load_from_HF'])
            self.assertTrue(calls['builder']['enable_inst_interactivity'])
            self.assertEqual(calls['processor']['device'], 'cpu')
            self.assertEqual(calls['image_size'], (6, 4))
            self.assertTrue(calls['predict']['normalize_coords'])  # Upstream normalizes pixel input.
            np.testing.assert_array_equal(calls['predict']['point_coords'], [[3, 2]])
            np.testing.assert_array_equal(calls['predict']['point_labels'], [1])
            np.testing.assert_array_equal(calls['predict']['box'], [1, 1, 5, 3])
            backend.predict(box_xyxy=[1, 1, 5, 3], points_xy=[], point_labels=[])
            self.assertIsNone(calls['predict']['point_coords'])
            self.assertIsNone(calls['predict']['point_labels'])


if __name__ == '__main__':
    unittest.main()