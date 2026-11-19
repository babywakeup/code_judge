"""Offline Transformers SAM3 contracts; synthetic inputs and no real model loads.

Only tiny NumPy-backed tensor doubles are used; no Torch import, installed model
or optional inference dependency is required to run the project regressions.
"""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


class TinyTensor:
    def __init__(self, value, dtype=None):
        self.value = np.asarray(value, dtype=dtype)
        self.device = SimpleNamespace(type='cpu')

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape

    def __getitem__(self, key):
        return TinyTensor(self.value[key])

    def __setitem__(self, key, value):
        self.value[key] = value

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype)

    def to(self, device, *args, **kwargs):
        if str(device) != 'cpu':
            raise AssertionError('This synthetic tensor is CPU only')
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def tolist(self):
        return self.value.tolist()

    def unsqueeze(self, axis):
        return TinyTensor(np.expand_dims(self.value, axis))


class TinyTorch:
    __version__ = 'test'
    float32 = np.dtype('float32')
    bool = np.dtype('bool')
    Tensor = TinyTensor
    _grad = True

    @staticmethod
    def zeros(shape, dtype=None):
        return TinyTensor(np.zeros(shape, dtype=dtype or np.float32))

    @staticmethod
    def tensor(value, dtype=None):
        return TinyTensor(value, dtype)

    as_tensor = tensor

    @staticmethod
    @contextmanager
    def inference_mode():
        before = TinyTorch._grad
        TinyTorch._grad = False
        try:
            yield
        finally:
            TinyTorch._grad = before

    no_grad = inference_mode

    @staticmethod
    def is_grad_enabled():
        return TinyTorch._grad

    device = str


torch = TinyTorch


class DeviceBatch(dict):
    """Minimal BatchEncoding analogue retaining evidence of device transfer."""

    def __init__(self, **values):
        super().__init__(values)
        self.transfers = []

    def to(self, device, *args, **kwargs):
        self.transfers.append(str(device))
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device, *args, **kwargs)
        return self


class TransformersSam3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name in ('config.json', 'processor_config.json'):
            (self.root / name).write_text('{}', encoding='utf-8')
        (self.root / 'model.safetensors').write_bytes(b'not real weights')
        self.adapter = importlib.import_module('reference_anomaly.sam3_transformers')
        self.rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        self.rgb[1, 2] = [231, 42, 17]
        self.box = [1, 1, 5, 3]
        self.masks = torch.zeros((2, 4, 6), dtype=torch.bool)
        self.masks[0, 1:3, 1:3] = True
        self.masks[1, 2:4, 4:6] = True
        self.scores = torch.tensor([.65, .95], dtype=torch.float32)

    def fake_stack(self, loading_info=None):
        """No import of transformers, checkpoint deserialization, or network."""
        info = dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
        if loading_info:
            info.update(loading_info)
        self.batches = []

        def encode(**kwargs):
            values = {}
            if 'images' in kwargs:
                values.update(pixel_values=torch.zeros((1, 3, 4, 6)),
                              original_sizes=torch.tensor([[4, 6]]))
            if 'original_sizes' in kwargs:
                values['original_sizes'] = torch.as_tensor(kwargs['original_sizes'])
            for name in ('input_boxes', 'input_boxes_labels', 'input_points', 'input_labels'):
                if name in kwargs:
                    values[name] = torch.as_tensor(kwargs[name])
            batch = DeviceBatch(**values)
            self.batches.append(batch)
            return batch

        self.processor = Mock(side_effect=encode)
        self.processor.post_process_instance_segmentation.return_value = [
            dict(masks=self.masks, scores=self.scores)]
        self.processor.post_process_masks.return_value = [self.masks.unsqueeze(0)]
        self.outputs = SimpleNamespace(
            pred_masks=self.masks[None, None], iou_scores=self.scores[None, None])

        def forward(**kwargs):
            if torch.is_grad_enabled():
                raise AssertionError('Model inference must disable gradients')
            return self.outputs

        self.model = Mock(side_effect=forward)
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.embedding = torch.zeros((1, 2, 2, 3))

        def embeddings(pixel_values):
            if torch.is_grad_enabled():
                raise AssertionError('Embedding inference must disable gradients')
            self.assertEqual(pixel_values.device.type, 'cpu')
            self.assertEqual(pixel_values.dtype, torch.float32)
            return self.embedding

        self.model.get_image_embeddings.side_effect = embeddings
        self.hf = SimpleNamespace(__version__='test')
        for name in ('Sam3Model', 'Sam3TrackerModel'):
            setattr(self.hf, name, SimpleNamespace(
                from_pretrained=Mock(return_value=(self.model, dict(info)))))
        for name in ('Sam3Processor', 'Sam3TrackerProcessor'):
            setattr(self.hf, name, SimpleNamespace(
                from_pretrained=Mock(return_value=self.processor)))
        self.fake_torch = SimpleNamespace(
            __version__=torch.__version__, float32=torch.float32,
            Tensor=torch.Tensor, tensor=torch.tensor, as_tensor=torch.as_tensor,
            device=torch.device, inference_mode=torch.inference_mode,
            no_grad=torch.no_grad, cuda=SimpleNamespace(is_available=lambda: False))
        real_import = importlib.import_module

        def import_module(name, *args, **kwargs):
            if name == 'torch':
                return self.fake_torch
            if name == 'transformers':
                return self.hf
            # PIL and any other ordinary dependencies are imported normally.
            return real_import(name, *args, **kwargs)

        return patch.object(self.adapter.importlib, 'import_module', side_effect=import_module)

    def predict(self, backend, points=None, labels=None):
        return backend.predict(box_xyxy=self.box,
                               points_xy=[] if points is None else points,
                               point_labels=[] if labels is None else labels)

    def assert_candidates(self, masks, quality):
        masks, quality = np.asarray(masks), np.asarray(quality)
        self.assertEqual(masks.shape, (2, 4, 6))
        self.assertTrue(np.isin(masks, [0, 1]).all())
        np.testing.assert_array_equal(masks, self.masks.numpy())
        self.assertEqual(quality.shape, (2,))
        np.testing.assert_allclose(quality, self.scores.numpy())

    def test_checkpoint_directory_pt_alias_and_safetensors_resolve_locally(self):
        # A .pt alias need not itself exist: only the sibling HF bundle is read.
        for path in (self.root, self.root / 'sam3.pt', self.root / 'model.safetensors'):
            with self.subTest(path=path):
                resolved = self.adapter.resolve_transformers_checkpoint(path)
                self.assertIsInstance(resolved, Path)
                self.assertEqual(resolved.resolve(), self.root)

    def test_checkpoint_requires_each_bundle_file_before_loading_models(self):
        for name in ('config.json', 'processor_config.json', 'model.safetensors'):
            with self.subTest(missing=name):
                path = self.root / name
                contents = path.read_bytes()
                path.unlink()
                try:
                    with self.fake_stack():
                        with self.assertRaisesRegex(ValueError, name.replace('.', r'\.')):
                            self.adapter.resolve_transformers_checkpoint(self.root)
                        with self.assertRaises(ValueError):
                            self.adapter.TransformersSam3Backend(self.root, device='cpu')
                        self.hf.Sam3Model.from_pretrained.assert_not_called()
                        self.hf.Sam3TrackerModel.from_pretrained.assert_not_called()
                finally:
                    path.write_bytes(contents)

    def test_cpu_and_auto_load_offline_fp32_correct_classes_and_metadata(self):
        for task, model_name, processor_name in (
                ('image', 'Sam3Model', 'Sam3Processor'),
                ('tracker', 'Sam3TrackerModel', 'Sam3TrackerProcessor')):
            for device in ('cpu', 'auto'):
                with self.subTest(task=task, device=device), self.fake_stack():
                    backend = self.adapter.TransformersSam3Backend(
                        self.root, device=device, task=task)
                    loader = getattr(self.hf, model_name).from_pretrained
                    loader.assert_called_once_with(
                        str(self.root), local_files_only=True, use_safetensors=True,
                        output_loading_info=True, dtype=torch.float32)
                    getattr(self.hf, processor_name).from_pretrained.assert_called_once_with(
                        str(self.root), local_files_only=True)
                    other = 'Sam3TrackerModel' if task == 'image' else 'Sam3Model'
                    getattr(self.hf, other).from_pretrained.assert_not_called()
                    self.model.to.assert_called_once()
                    self.assertEqual(str(self.model.to.call_args.args[0]), 'cpu')
                    self.model.eval.assert_called_once_with()
                    self.assertIsInstance(backend.provenance, dict)
                    metadata = json.dumps(backend.provenance)
                    for expected in (model_name, processor_name, 'cpu', task):
                        self.assertIn(expected, metadata)

    def test_explicit_cuda_without_gpu_fails_without_loading_model(self):
        with self.fake_stack():
            with self.assertRaisesRegex(RuntimeError, '(?i)cuda.*(unavailable|available|not found)'):
                self.adapter.TransformersSam3Backend(self.root, device='cuda')
            self.hf.Sam3Model.from_pretrained.assert_not_called()
            self.hf.Sam3TrackerModel.from_pretrained.assert_not_called()

    def test_partial_or_failed_loading_never_uses_random_parameters(self):
        for task in ('image', 'tracker'):
            for key, value in (
                    ('missing_keys', ['vision_encoder.weight']),
                    ('mismatched_keys', [('vision_encoder.weight', (2, 3), (3, 3))]),
                    ('error_msgs', ['synthetic deserialization failure'])):
                with self.subTest(task=task, diagnostic=key), self.fake_stack({key: value}):
                    with self.assertRaises((ValueError, RuntimeError)):
                        self.adapter.TransformersSam3Backend(self.root, device='cpu', task=task)
                    self.model.assert_not_called()

    def test_unexpected_video_branch_keys_alone_are_allowed(self):
        for task in ('image', 'tracker'):
            with self.subTest(task=task), self.fake_stack(
                    {'unexpected_keys': ['video_tracking_branch.weight']}):
                backend = self.adapter.TransformersSam3Backend(self.root, device='cpu', task=task)
                self.assertIsInstance(backend.provenance, dict)
                self.model.eval.assert_called_once_with()

    def test_image_box_text_postprocessing_preserves_all_instances(self):
        with self.fake_stack():
            backend = self.adapter.TransformersSam3Backend(
                self.root, device='cpu', text='synthetic component',
                threshold=.5, mask_threshold=.5)
            backend.set_image(self.rgb)
            masks, quality = self.predict(backend)
            call = self.processor.call_args.kwargs
            self.assertIsInstance(call['images'], Image.Image)
            self.assertEqual(call['images'].mode, 'RGB')
            np.testing.assert_array_equal(np.asarray(call['images']), self.rgb)
            self.assertEqual(call['text'], 'synthetic component')
            self.assertEqual(call['input_boxes'], [[self.box]])
            self.assertEqual(call['input_boxes_labels'], [[1]])
            self.assertEqual(call['return_tensors'], 'pt')
            self.model.assert_called_once()
            kwargs = self.model.call_args.kwargs
            self.assertNotIn('original_sizes', kwargs)
            self.assertEqual(kwargs['pixel_values'].dtype, torch.float32)
            self.assertEqual(kwargs['pixel_values'].device.type, 'cpu')
            self.assertEqual(self.batches[-1].transfers, ['cpu'])
            self.processor.post_process_instance_segmentation.assert_called_once_with(
                self.outputs, threshold=.5, mask_threshold=.5, target_sizes=[[4, 6]])
            self.assert_candidates(masks, quality)  # Neither best-only nor union here.

    def test_image_without_text_forwards_configured_thresholds(self):
        with self.fake_stack():
            backend = self.adapter.TransformersSam3Backend(
                self.root, device='cpu', threshold=.73, mask_threshold=.61)
            backend.set_image(self.rgb)
            self.predict(backend)
            self.assertIsNone(self.processor.call_args.kwargs.get('text'))
            self.processor.post_process_instance_segmentation.assert_called_once_with(
                self.outputs, threshold=.73, mask_threshold=.61, target_sizes=[[4, 6]])

    def test_image_nonempty_points_fail_clearly_instead_of_being_dropped(self):
        with self.fake_stack():
            backend = self.adapter.TransformersSam3Backend(self.root, device='cpu')
            backend.set_image(self.rgb)
            self.processor.reset_mock()
            with self.assertRaisesRegex(ValueError, '(?i)point'):
                self.predict(backend, points=[[3, 2]], labels=[1])
            self.processor.assert_not_called()
            self.model.assert_not_called()

    def test_tracker_box_point_label_nesting_embeddings_and_candidates(self):
        with self.fake_stack():
            backend = self.adapter.TransformersSam3Backend(self.root, device='auto', task='tracker')
            backend.set_image(self.rgb)
            self.model.get_image_embeddings.assert_called_once()
            masks, quality = self.predict(backend, points=[[2, 1], [4, 3]], labels=[1, 0])
            call = self.processor.call_args.kwargs
            np.testing.assert_array_equal(call['original_sizes'], [[4, 6]])
            self.assertEqual(call['input_boxes'], [[self.box]])
            self.assertEqual(call['input_points'], [[[[2, 1], [4, 3]]]])
            self.assertEqual(call['input_labels'], [[[1, 0]]])
            self.assertEqual(call['return_tensors'], 'pt')
            self.assertEqual(self.batches[-1].transfers, ['cpu'])
            kwargs = self.model.call_args.kwargs
            self.assertIs(kwargs['image_embeddings'], self.embedding)
            self.assertIs(kwargs['multimask_output'], True)
            self.assertEqual(kwargs['input_points'].device.type, 'cpu')
            np.testing.assert_array_equal(kwargs['input_labels'], [[[1, 0]]])
            self.processor.post_process_masks.assert_called_once()
            args = self.processor.post_process_masks.call_args.args
            self.assertIs(args[0], self.outputs.pred_masks)
            np.testing.assert_array_equal(args[1], [[4, 6]])
            self.assert_candidates(masks, quality)

    def test_tracker_box_only_omits_empty_points(self):
        with self.fake_stack():
            backend = self.adapter.TransformersSam3Backend(self.root, device='cpu', task='tracker')
            backend.set_image(self.rgb)
            self.predict(backend)
            call = self.processor.call_args.kwargs
            self.assertEqual(call['input_boxes'], [[self.box]])
            self.assertIsNone(call.get('input_points'))
            self.assertIsNone(call.get('input_labels'))
            self.assertTrue(self.model.call_args.kwargs['multimask_output'])

    def test_tracker_nonempty_text_is_rejected(self):
        with self.fake_stack():
            with self.assertRaisesRegex(ValueError, '(?i)text'):
                backend = self.adapter.TransformersSam3Backend(
                    self.root, device='cpu', task='tracker', text='synthetic component')
                backend.set_image(self.rgb)
                self.predict(backend)
            self.model.assert_not_called()


if __name__ == '__main__':
    unittest.main()