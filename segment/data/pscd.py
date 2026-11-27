"""PSCD full panoramas, aligned crops and shifted-view crops.

Adapted from the dataset protocol of Robust-Scene-Change-Detection (MIT,
Copyright (c) 2024 Chad Lin). Images remain in the user's local dataset root.
Dataset usage is governed separately by PSCD's terms of use.
"""

from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..models.scene_change import SceneChangeProcessor
from .splits import PSCD_SPLITS


class PSCD(Dataset):
    """Full-image PSCD dataset returning model-ready tensors without a batch axis.

    ``root`` may be the extracted PSCD directory or its immediate parent.
    ``filenames`` overrides the fixed split for local subsets/tests. By default
    labels come from mask_t0, with every nonzero value mapped to class 1.
    The default processor scales RGB to [0,1], center-crops to multiples of 14,
    and performs no mean/std normalization. Use get_raw_sample for full pixels.
    """

    def __init__(self, root=None, mode="train", *, processor=None,
                 filenames: Sequence[str] | None = None, mask_source="t0",
                 hflip_prob=0.0):
        if mode not in PSCD_SPLITS:
            raise ValueError("mode must be train, val or test")
        if mask_source not in {"t0", "t1"}:
            raise ValueError("mask_source must be t0 or t1")
        if not 0 <= hflip_prob <= 1:
            raise ValueError("hflip_prob must be in [0,1]")
        if root is None:
            root = Path(__file__).resolve().parents[2] / "data_" / "PSCD"
        self.root = Path(root).expanduser().resolve()
        if not (self.root / "t0").is_dir() and (self.root / "PSCD" / "t0").is_dir():
            self.root /= "PSCD"
        self.mode = mode
        self.mask_source = mask_source
        self.hflip_prob = hflip_prob
        self.processor = processor if processor is not None else SceneChangeProcessor()
        self.original_filenames = tuple(PSCD_SPLITS[mode] if filenames is None else filenames)
        if not self.original_filenames:
            raise ValueError("Dataset must contain at least one filename")
        if len(set(self.original_filenames)) != len(self.original_filenames):
            raise ValueError("Duplicate filenames are not allowed")
        for name in self.original_filenames:
            if Path(name).name != name or not name.endswith(".png"):
                raise ValueError(f"Expected a PNG basename, got {name!r}")
            for folder in ("t0", "t1", f"mask_{mask_source}"):
                path = self.root / folder / name
                if not path.is_file():
                    raise FileNotFoundError(f"Missing PSCD file: {path}")

    @property
    def filenames(self):
        return self.original_filenames

    def __len__(self):
        return len(self.original_filenames)

    def _index(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return index

    def _load(self, image_index):
        name = self.original_filenames[image_index]
        images = []
        for folder in ("t0", "t1"):
            with Image.open(self.root / folder / name) as image:
                images.append(np.array(image.convert("RGB")))
        with Image.open(self.root / f"mask_{self.mask_source}" / name) as image:
            mask = np.array(image)
        if mask.ndim != 2:
            raise ValueError(f"PSCD mask must be single-channel: {name}")
        if images[0].shape != images[1].shape or mask.shape != images[0].shape[:2]:
            raise ValueError(f"PSCD image/mask dimensions differ: {name}")
        return images[0], images[1], (mask > 0).astype(np.int64)

    def get_raw_sample(self, index):
        """Return uint8 HWC before/after and int64 HW labels, before processing."""
        return self._load(self._index(index))

    def __getitem__(self, index):
        before, after, labels = self.get_raw_sample(index)
        if self.hflip_prob and torch.rand(()).item() < self.hflip_prob:
            before, after, labels = (np.flip(x, axis=1).copy() for x in (before, after, labels))
        batch = self.processor(before, after, segmentation_maps=labels,
                               input_data_format="channels_last", return_tensors="pt")
        return {key: value.squeeze(0) for key, value in batch.items()}


class CroppedPSCD(PSCD):
    """15 overlapping 224-square crops per 224x1024 panorama by default.

    Unlike the reference, optionally resize raw high-resolution panoramas to
    this geometry first. RGB uses bilinear interpolation, masks nearest-neighbor.
    This changes aspect ratio when the source is not already 224x1024.
    """

    def __init__(self, root=None, mode="train", *, crop_num=15,
                 resize_panorama=True, **kwargs):
        if not isinstance(crop_num, int) or not 2 <= crop_num <= 801:
            raise ValueError("crop_num must be an integer between 2 and 801")
        super().__init__(root, mode, **kwargs)
        self.resize_panorama = resize_panorama
        self.crop_num = crop_num
        self.stride = (1024 - 224) // (crop_num - 1)
        self.cropped_length = 224 + (crop_num - 1) * self.stride
        self.cropped_prefix = (1024 - self.cropped_length) // 2

    def _load(self, image_index):
        before, after, labels = super()._load(image_index)
        if before.shape[:2] != (224, 1024) and self.resize_panorama:
            before, after = (
                np.array(Image.fromarray(x).resize((1024, 224), Image.Resampling.BILINEAR))
                for x in (before, after)
            )
            labels = np.array(Image.fromarray(labels.astype(np.uint8)).resize(
                (1024, 224), Image.Resampling.NEAREST)).astype(np.int64)
        return before, after, labels

    @property
    def filenames(self):
        return tuple(f"{Path(name).stem}.{crop}.png"
                     for name in self.original_filenames for crop in range(self.crop_num))

    def __len__(self):
        return len(self.original_filenames) * self.crop_num

    def _crop(self, sample, crop):
        if sample[0].shape[:2] != (224, 1024):
            raise ValueError("CroppedPSCD requires 224x1024 panoramas")
        start = self.cropped_prefix + crop * self.stride
        return tuple(x[:, start:start + 224].copy() for x in sample)

    def get_raw_sample(self, index):
        image_index, crop = divmod(self._index(index), self.crop_num)
        return self._crop(self._load(image_index), crop)


class DiffViewPSCD(CroppedPSCD):
    """Pair t0 crop i with t1 crop i+distance; clear non-overlap t0 labels.

    This reproduces the reference's background-zero policy, not ignore_index.
    Distances of +1/-1 shift the crop window by +57/-57 pixels by default.
    """

    def __init__(self, root=None, mode="train", *, adjacent_distance=1, **kwargs):
        if kwargs.get("mask_source", "t0") != "t0":
            raise ValueError("DiffViewPSCD requires mask_source=t0")
        super().__init__(root, mode, **kwargs)
        if not isinstance(adjacent_distance, int):
            raise ValueError("adjacent_distance must be an integer")
        if abs(adjacent_distance) >= self.crop_num or abs(adjacent_distance) * self.stride >= 224:
            raise ValueError("Shift must leave at least one valid crop and overlapping pixels")
        self.adjacent_distance = adjacent_distance
        self.crops_per_image = self.crop_num - abs(adjacent_distance)
        self.first_crop = max(0, -adjacent_distance)

    def __len__(self):
        return len(self.original_filenames) * self.crops_per_image

    @property
    def filenames(self):
        return tuple(f"{Path(name).stem}.{i}-{i + self.adjacent_distance}.png"
                     for name in self.original_filenames
                     for i in range(self.first_crop, self.first_crop + self.crops_per_image))

    def get_raw_sample(self, index):
        image_index, crop_offset = divmod(self._index(index), self.crops_per_image)
        crop = self.first_crop + crop_offset
        sample = self._load(image_index)
        before, _, labels = self._crop(sample, crop)
        _, after, _ = self._crop(sample, crop + self.adjacent_distance)
        shift = self.adjacent_distance * self.stride
        if shift > 0:
            labels[:, :shift] = 0
        elif shift < 0:
            labels[:, shift:] = 0
        return before, after, labels
