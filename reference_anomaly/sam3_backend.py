"""Optional official SAM3 image adapter. Importing this module needs no Torch."""
from __future__ import annotations

from contextlib import nullcontext
import importlib
import importlib.util
from pathlib import Path

import numpy as np


def resolve_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).resolve()
    if checkpoint.is_dir():
        checkpoint /= 'sam3.pt'
    if checkpoint.suffix != '.pt':
        raise ValueError('Official SAM3 requires sam3.pt, not Transformers model.safetensors or .incomplete files')
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f'Complete local SAM3 checkpoint not found: {checkpoint}')
    return checkpoint


def choose_backend(checkpoint: str | Path, backend: str = 'auto') -> str:
    if backend not in ('auto', 'transformers', 'official'):
        raise ValueError('backend must be auto, transformers or official')
    if backend != 'auto':
        return backend
    path = Path(checkpoint).resolve()
    directory = path.parent if path.suffix in ('.pt', '.safetensors') else path
    return 'transformers' if (directory / 'model.safetensors').is_file() else 'official'


def environment_report(checkpoint: str | Path, backend: str = 'auto') -> dict:
    """Report independently verifiable prerequisites; never download weights."""
    selected = choose_backend(checkpoint, backend)
    if selected == 'transformers':
        from .sam3_transformers import resolve_transformers_checkpoint
        path = resolve_transformers_checkpoint(checkpoint) / 'model.safetensors'
    else:
        path = resolve_checkpoint(checkpoint)
    available = importlib.util.find_spec('sam3') is not None
    report = dict(checkpoint=str(path), checkpoint_bytes=path.stat().st_size,
                  selected_backend=selected, official_sam3_installed=available,
                  transformers_installed=importlib.util.find_spec('transformers') is not None,
                  cuda_available=False)
    try:
        torch = importlib.import_module('torch')
        report.update(torch_version=str(torch.__version__), cuda_runtime=torch.version.cuda,
                      cuda_available=bool(torch.cuda.is_available()))
        if report['cuda_available']:
            report['gpu'] = torch.cuda.get_device_name(0)
    except ImportError:
        report['torch_version'] = None
    report['recommended_cuda_path_ready'] = available and report['cuda_available']
    report['automatic_device'] = 'cuda' if report['cuda_available'] else 'cpu'
    return report


class OfficialSam3Backend:
    """SAM3 interactive instance path (not text/concept detection or video).

    Crops are RGB; boxes/points are crop-local pixel coordinates. CPU is only an
    explicit opt-in experiment: upstream officially targets CUDA, so CPU success
    and speed are not guaranteed. No fallback or implicit network downloads.
    """
    def __init__(self, checkpoint: str | Path, *, device: str = 'cuda'):
        path = resolve_checkpoint(checkpoint)
        if device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device must be auto, cpu or cuda')
        try:
            torch = importlib.import_module('torch')
        except ImportError as exc:
            raise RuntimeError('PyTorch is not installed in the selected environment') from exc
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use an NVIDIA CUDA environment; --device cpu is experimental and may be very slow')
        try:
            builder = importlib.import_module('sam3.model_builder')
            processor = importlib.import_module('sam3.model.sam3_image_processor')
        except ImportError as exc:
            raise RuntimeError('Official sam3 package/dependencies are missing; weights alone are not the inference code') from exc
        self.torch, self.device = torch, device
        self.model = builder.build_sam3_image_model(
            checkpoint_path=str(path), load_from_HF=False, device=device,
            eval_mode=True, enable_inst_interactivity=True, compile=False,
        )
        if getattr(self.model, 'inst_interactive_predictor', None) is None:
            raise RuntimeError('Installed SAM3 does not provide the required interactive instance head')
        self.processor = processor.Sam3Processor(self.model, device=device)
        self.state = None
        self.provenance = dict(backend='official_sam3_predict_inst', device=device,
                               torch_version=str(torch.__version__), compile=False,
                               builder_file=str(Path(builder.__file__).resolve()),
                               processor_file=str(Path(processor.__file__).resolve()),
                               load_from_HF=False, enable_inst_interactivity=True)

    def _autocast(self):
        if self.device == 'cpu':
            return nullcontext()
        dtype = self.torch.bfloat16 if self.torch.cuda.is_bf16_supported() else self.torch.float16
        return self.torch.autocast('cuda', dtype=dtype)

    def set_image(self, image_rgb: np.ndarray):
        from PIL import Image
        if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError('SAM3 input must be uint8 RGB')
        # PIL avoids upstream HWC ndarray shape ambiguity in set_image.
        with self.torch.inference_mode(), self._autocast():
            self.state = self.processor.set_image(Image.fromarray(image_rgb))

    def predict(self, *, box_xyxy, points_xy, point_labels):
        if self.state is None:
            raise RuntimeError('set_image must precede predict')
        points = np.asarray(points_xy, np.float32).reshape(-1, 2)
        labels = np.asarray(point_labels, np.int32)
        with self.torch.inference_mode(), self._autocast():
            # Upstream SAM2Transforms: True DIVIDES pixel coordinates by orig_hw
            # before scaling to model resolution. False expects already-normalized
            # coordinates. Do not reverse this counterintuitive flag.
            masks, quality, _ = self.model.predict_inst(
                self.state, box=np.asarray(box_xyxy, np.float32),
                point_coords=points if len(points) else None,
                point_labels=labels if len(points) else None,
                multimask_output=True, return_logits=False, normalize_coords=True,
            )
        return np.asarray(masks), np.asarray(quality)