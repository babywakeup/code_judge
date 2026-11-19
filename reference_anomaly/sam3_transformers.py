"""Offline Hugging Face SAM3 adapters: image concepts and interactive points.

No official sam3 package, CUDA installation or remote model code is required.
Importing this module does not import Torch or load any checkpoint.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def resolve_transformers_checkpoint(path: str | Path) -> Path:
    """Resolve a local HF bundle, accepting the legacy sam3.pt sibling alias."""
    directory = Path(path).resolve()
    if directory.suffix in ('.pt', '.safetensors'):
        directory = directory.parent
    if not directory.is_dir():
        raise ValueError(f'Local Transformers model directory not found: {directory}')
    for name in ('config.json', 'processor_config.json', 'model.safetensors'):
        file = directory / name
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f'Transformers SAM3 requires a complete local {name}: {directory}')
    return directory


def bundle_sources(directory: Path) -> dict[str, Path]:
    """Whitelist only model/tokenizer files, not unrelated annotations."""
    names = ('model.safetensors', 'config.json', 'processor_config.json',
             'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'vocab.json', 'merges.txt', 'added_tokens.json', 'preprocessor_config.json')
    return {f'hf_{name.replace(".", "_")}': directory / name for name in names
            if (directory / name).is_file()}


class TransformersSam3Backend:
    """Image: Sam3Model boxes/text; tracker: Sam3TrackerModel boxes/points.

    Image outputs are distinct instances, NOT alternate masks of one instance.
    Tracker outputs are alternative hypotheses. The caller must not conflate
    those semantics. All coordinates here are original crop-local pixels;
    processors perform the normalization required by their respective models.
    """
    def __init__(self, checkpoint: str | Path, *, device='auto', task='image',
                 text: str | None = None, threshold=.5, mask_threshold=.5):
        directory = resolve_transformers_checkpoint(checkpoint)
        if task not in ('image', 'tracker') or device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('task must be image/tracker and device must be auto/cpu/cuda')
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise ValueError('text must be a nonempty string when supplied')
        if task == 'tracker' and text is not None:
            raise ValueError('Tracker point prompts do not support text; use image/box mode for text prompts')
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in (threshold, mask_threshold)):
            raise ValueError('Model confidence and mask thresholds must be in [0,1]')
        try:
            torch = importlib.import_module('torch')
            hf = importlib.import_module('transformers')
        except ImportError as exc:
            raise RuntimeError('Install PyTorch and Transformers with SAM3 support in this interpreter') from exc
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; choose --device cpu or --device auto')
        model_name, processor_name = (('Sam3Model', 'Sam3Processor') if task == 'image'
                                      else ('Sam3TrackerModel', 'Sam3TrackerProcessor'))
        try:
            model_cls, processor_cls = getattr(hf, model_name), getattr(hf, processor_name)
        except (AttributeError, ImportError) as exc:
            raise RuntimeError(f'Installed Transformers lacks {model_name}/{processor_name}') from exc
        self.torch, self.device, self.task = torch, device, task
        self.text, self.threshold, self.mask_threshold = text, threshold, mask_threshold
        # Explicit CPU FP32 also overrides a BF16 dtype recorded in some bundles.
        # No device_map="auto", remote code, Hub request or CUDA-only autocast.
        model, loading = model_cls.from_pretrained(
            str(directory), local_files_only=True, use_safetensors=True,
            output_loading_info=True, dtype=torch.float32,
        )
        failures = {k: loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs') if loading.get(k)}
        if failures:
            raise RuntimeError(f'{model_name} checkpoint did not load completely; refusing random parameters: {failures}')
        self.model = model.to(device).eval()
        self.processor = processor_cls.from_pretrained(str(directory), local_files_only=True)
        self.candidate_semantics = 'instances' if task == 'image' else 'alternatives'
        self.image = None
        self.image_embeddings = None
        self.original_sizes = None
        self.provenance = dict(backend='transformers', model_class=model_name,
                               processor_class=processor_name, task=task, device=device,
                               dtype='float32', transformers_version=str(hf.__version__),
                               torch_version=str(torch.__version__), local_files_only=True,
                               candidate_semantics=self.candidate_semantics, text=text,
                               confidence_threshold=threshold if task == 'image' else None,
                               mask_threshold=mask_threshold if task == 'image' else 0.,
                               missing_keys=[], mismatched_keys=[],
                               ignored_other_branch_key_count=len(loading.get('unexpected_keys', [])))

    def set_image(self, image_rgb: np.ndarray):
        from PIL import Image
        if (image_rgb.dtype != np.uint8 or image_rgb.ndim != 3
                or image_rgb.shape[2] != 3 or not image_rgb.size):
            raise ValueError('SAM3 input must be nonempty uint8 RGB')
        self.image = Image.fromarray(image_rgb)
        self.original_sizes = [[image_rgb.shape[0], image_rgb.shape[1]]]
        if self.task == 'tracker':
            inputs = self.processor(images=self.image, return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                self.image_embeddings = self.model.get_image_embeddings(inputs['pixel_values'])

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)

    def predict(self, *, box_xyxy, points_xy, point_labels):
        if self.image is None:
            raise RuntimeError('set_image must precede predict')
        if len(points_xy) != len(point_labels):
            raise ValueError('Point coordinates and labels must have equal lengths')
        if self.task == 'image':
            if len(points_xy):
                raise ValueError('Sam3Model accepts text/boxes, not points; use the tracker task for point prompts')
            inputs = self.processor(images=self.image, text=self.text,
                                    input_boxes=[[box_xyxy]], input_boxes_labels=[[1]],
                                    return_tensors='pt').to(self.device)
            sizes = inputs['original_sizes'].tolist()
            kwargs = {k: v for k, v in inputs.items() if k != 'original_sizes'}
            with self.torch.inference_mode():
                outputs = self.model(**kwargs)
                result = self.processor.post_process_instance_segmentation(
                    outputs, threshold=self.threshold, mask_threshold=self.mask_threshold,
                    target_sizes=sizes)[0]
            return self._numpy(result['masks']), self._numpy(result['scores'])
        prompts = dict(original_sizes=self.original_sizes, input_boxes=[[box_xyxy]], return_tensors='pt')
        if len(points_xy):
            prompts.update(input_points=[[points_xy]], input_labels=[[point_labels]])
        inputs = self.processor(**prompts).to(self.device)
        kwargs = {k: v for k, v in inputs.items() if k != 'original_sizes'}
        with self.torch.inference_mode():
            outputs = self.model(**kwargs, image_embeddings=self.image_embeddings, multimask_output=True)
            masks = self.processor.post_process_masks(outputs.pred_masks, inputs['original_sizes'])[0]
        # B=1 and one interactive object per prompt call; retain C alternatives.
        return self._numpy(masks)[0], self._numpy(outputs.iou_scores)[0, 0]