"""Paired RGB image processing with synchronized label geometry."""

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import BaseImageProcessor
from transformers.image_processing_utils import BatchFeature


class SceneChangeImageProcessor(BaseImageProcessor):
    """Single-modality processor for paired RGB images and segmentation labels."""

    model_input_names = ["pixel_values", "pixel_values_pair"]

    def __init__(self, size=None, patch_size=14, do_rescale=True, rescale_factor=1 / 255,
                 do_normalize=False, image_mean=None, image_std=None,
                 mask_value_map=None, ignore_index=255, **kwargs):
        super().__init__(**kwargs)
        if size is not None:
            if isinstance(size, int):
                size = {"height": size, "width": size}
            if not isinstance(size, dict) or set(size) != {"height", "width"}:
                raise ValueError("size must be an int or a height/width dictionary")
            if any(x < patch_size or x % patch_size for x in size.values()):
                raise ValueError("size dimensions must be positive patch-size multiples")
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self.size = size
        self.patch_size = patch_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = list(image_mean if image_mean is not None else [0.485, 0.456, 0.406])
        self.image_std = list(image_std if image_std is not None else [0.229, 0.224, 0.225])
        if len(self.image_mean) != 3 or len(self.image_std) != 3 or any(x <= 0 for x in self.image_std):
            raise ValueError("image_mean/std must contain three values; std must be positive")
        self.mask_value_map = ({int(k): int(v) for k, v in mask_value_map.items()}
                               if mask_value_map is not None else None)
        self.ignore_index = ignore_index

    @staticmethod
    def _batch(images):
        if isinstance(images, (list, tuple)):
            return list(images)
        if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 4:
            return list(images)
        return [images]

    @staticmethod
    def _image_tensor(image, input_data_format):
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
            input_data_format = "channels_last"
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(np.array(image, copy=True))
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise ValueError("Each RGB image must be PIL, numpy HWC/CHW or torch CHW/HWC")
        if input_data_format is None:
            first, last = image.shape[0] == 3, image.shape[-1] == 3
            if first == last:
                raise ValueError("Ambiguous channel layout; pass input_data_format explicitly")
            input_data_format = "channels_first" if first else "channels_last"
        if input_data_format == "channels_last":
            image = image.permute(2, 0, 1)
        elif input_data_format != "channels_first":
            raise ValueError("input_data_format must be channels_first or channels_last")
        if image.shape[0] != 3:
            raise ValueError("Images must have three RGB channels")
        is_integer = not image.is_floating_point()
        return image.float(), is_integer

    @staticmethod
    def _mask_tensor(mask):
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(np.array(mask, copy=True))
        if not isinstance(mask, torch.Tensor):
            raise ValueError("Mask must be PIL, numpy or torch")
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.ndim != 2 or (mask.is_floating_point() and not torch.equal(mask, mask.round())):
            raise ValueError("Each mask must be an H,W array of integer class IDs")
        return mask.long()

    @staticmethod
    def _center_crop(tensor, size, fill=0):
        height, width = tensor.shape[-2:]
        pad_h, pad_w = max(size[0] - height, 0), max(size[1] - width, 0)
        if pad_h or pad_w:
            tensor = F.pad(tensor, (pad_w // 2, (pad_w + 1) // 2,
                                    pad_h // 2, (pad_h + 1) // 2), value=fill)
        top = int(round((tensor.shape[-2] - size[0]) / 2))
        left = int(round((tensor.shape[-1] - size[1]) / 2))
        return tensor[..., top:top + size[0], left:left + size[1]]

    def preprocess(self, images, images_pair, segmentation_maps=None, return_tensors="pt",
                   input_data_format=None, do_rescale=None, do_normalize=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected processor arguments: {sorted(kwargs)}")
        images, pairs = self._batch(images), self._batch(images_pair)
        if not images or len(images) != len(pairs):
            raise ValueError("Provide equally sized, nonempty image-pair batches")
        if segmentation_maps is None:
            masks = [None] * len(images)
        elif isinstance(segmentation_maps, (np.ndarray, torch.Tensor)) and segmentation_maps.ndim == 3:
            if len(images) == 1 and (segmentation_maps.shape[0] == 1 or segmentation_maps.shape[-1] == 1):
                masks = [segmentation_maps]
            else:
                masks = list(segmentation_maps)
        else:
            masks = self._batch(segmentation_maps)
        if len(masks) != len(images):
            raise ValueError("One segmentation map is required per image pair")
        rescale = self.do_rescale if do_rescale is None else do_rescale
        normalize = self.do_normalize if do_normalize is None else do_normalize
        data = {"pixel_values": [], "pixel_values_pair": []}
        if segmentation_maps is not None:
            data["labels"] = []
        for image, pair, mask in zip(images, pairs, masks):
            x, integer_x = self._image_tensor(image, input_data_format)
            y, integer_y = self._image_tensor(pair, input_data_format)
            if x.shape != y.shape:
                raise ValueError("Each image pair must have equal original dimensions")
            if self.size is None:
                size = tuple(d // self.patch_size * self.patch_size for d in x.shape[-2:])
                if min(size) < self.patch_size:
                    raise ValueError("Input image is smaller than one patch")
            else:
                size = (self.size["height"], self.size["width"])
            for key, tensor, integer in (("pixel_values", x, integer_x), ("pixel_values_pair", y, integer_y)):
                if rescale and (integer or tensor.max() > 1):
                    tensor = tensor * self.rescale_factor
                tensor = self._center_crop(tensor, size)
                if normalize:
                    mean = tensor.new_tensor(self.image_mean)[:, None, None]
                    std = tensor.new_tensor(self.image_std)[:, None, None]
                    tensor = (tensor - mean) / std
                data[key].append(tensor)
            if mask is not None:
                mask = self._mask_tensor(mask)
                if mask.shape != x.shape[-2:]:
                    raise ValueError("Mask dimensions must match the original images")
                if self.mask_value_map is not None:
                    mapped = torch.full_like(mask, self.ignore_index)
                    for source, target in self.mask_value_map.items():
                        mapped[mask == source] = target
                    mask = mapped
                data["labels"].append(self._center_crop(mask, size, fill=self.ignore_index))
        if return_tensors is None:
            return BatchFeature(data=data)
        if str(return_tensors) not in {"pt", "np", "TensorType.PYTORCH", "TensorType.NUMPY"}:
            raise ValueError("Only pt, np or None return_tensors are supported")
        try:
            data = {key: torch.stack(values) for key, values in data.items()}
        except RuntimeError as exc:
            raise ValueError("Batch image sizes differ; configure a fixed size or process separately") from exc
        if str(return_tensors) in {"np", "TensorType.NUMPY"}:
            data = {key: value.cpu().numpy() for key, value in data.items()}
        return BatchFeature(data=data)

    def post_process_semantic_segmentation(self, outputs, target_sizes=None):
        logits = outputs.logits
        if target_sizes is not None and len(target_sizes) != logits.shape[0]:
            raise ValueError("One target size is required per batch item")
        results = []
        for index, item in enumerate(logits):
            if target_sizes is not None:
                item = F.interpolate(item[None], size=tuple(target_sizes[index]),
                                     mode="bilinear", align_corners=False)[0]
            results.append(item.argmax(dim=0))
        return results


SceneChangeProcessor = SceneChangeImageProcessor

__all__ = ["SceneChangeImageProcessor", "SceneChangeProcessor"]
