"""VL-CMU-CD paired images and shifted-view samples.

The split IDs and shifted-index protocol follow the MIT-licensed
Robust-Scene-Change-Detection dataset loader. The dataset files themselves are
expected to be provided locally by the user.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..models.scene_change import SceneChangeProcessor


VL_CMU_CD_TRAIN_GROUPS = frozenset(
    "001 002 003 004 005 008 010 011 014 015 016 017 018 019 020 021 022 026 "
    "029 030 031 033 035 037 040 041 042 043 044 046 049 051 052 053 054 055 "
    "057 062 063 065 067 068 070 071 072 073 075 078 079 080 083 084 086 087 "
    "088 089 090 091 096 098 099 102 104 105 108 110 114 115 116 121 122 123 "
    "126 127 128 130 133 136 137 138 140 141 143 146 147 148 151".split()
)
VL_CMU_CD_VAL_GROUPS = frozenset(
    "013 059 074 101 103 109 111 118 124 131 149".split()
)


def _parse_name(name: str) -> tuple[str, str, str]:
    """Return group, sequence and angle from ``<group>_1_<seq>_<angle>.png``."""
    path = Path(name)
    parts = path.stem.split("_")
    if path.name != name or path.suffix.lower() != ".png" or len(parts) != 4:
        raise ValueError(f"Expected a VL-CMU-CD PNG basename, got {name!r}")
    if parts[1] != "1" or not all(part.isdigit() for part in (parts[0], parts[2], parts[3])):
        raise ValueError(f"Expected <group>_1_<seq>_<angle>.png, got {name!r}")
    return parts[0], parts[2], parts[3]


def _resolve_root(root: str | Path | None) -> Path:
    if root is None:
        root = Path(__file__).resolve().parents[2] / "data_" / "VL-CMU-CD2"
    root = Path(root).expanduser().resolve()
    candidates = (root, root / "VL-CMU-CD-binary255")
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in ("train", "test")):
            return candidate
    raise FileNotFoundError(
        f"Could not find VL-CMU-CD train/test directories under {root}. "
        "Expected <root>/{train,test}/{t0,t1,mask} or "
        "<root>/VL-CMU-CD-binary255/{train,test}/{t0,t1,mask}."
    )


class VLCMUCD(Dataset):
    """VL-CMU-CD returning model-ready paired-image and binary-label tensors.

    ``root`` accepts either the extracted ``VL-CMU-CD-binary255`` directory or
    its parent (including the repository's ``data_/VL-CMU-CD2``). Train and
    validation are group-disjoint subsets of ``train``; test uses all of
    ``test``. ``filenames`` optionally selects a local subset of that split.
    Raw masks are binarized with ``mask > 0``; this matches the reference loader
    for the grayscale anti-aliased masks in the binary255 release.
    """

    def __init__(self, root: str | Path | None = None, mode: str = "train", *,
                 processor=None, filenames: Sequence[str] | None = None,
                 hflip_prob: float = 0.0):
        mode = mode.lower()
        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be train, val or test")
        if not 0 <= hflip_prob <= 1:
            raise ValueError("hflip_prob must be in [0,1]")

        self.root = _resolve_root(root)
        self.mode = mode
        self.split_root = self.root / ("test" if mode == "test" else "train")
        for folder in ("t0", "t1", "mask"):
            if not (self.split_root / folder).is_dir():
                raise FileNotFoundError(f"Missing VL-CMU-CD directory: {self.split_root / folder}")

        if filenames is None:
            names = sorted(path.name for path in (self.split_root / "t0").glob("*.png"))
            if mode != "test":
                groups = VL_CMU_CD_TRAIN_GROUPS if mode == "train" else VL_CMU_CD_VAL_GROUPS
                names = [name for name in names if _parse_name(name)[0] in groups]
        else:
            names = sorted(filenames)
        if not names:
            raise ValueError(f"No VL-CMU-CD samples found for mode={mode!r}")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate filenames are not allowed")

        self.original_filenames = tuple(names)
        for name in self.original_filenames:
            _parse_name(name)
            for folder in ("t0", "t1", "mask"):
                path = self.split_root / folder / name
                if not path.is_file():
                    raise FileNotFoundError(f"Missing VL-CMU-CD sample file: {path}")

        self.processor = processor if processor is not None else SceneChangeProcessor()
        self.hflip_prob = hflip_prob
        self._records = tuple((name, name, name) for name in self.original_filenames)
        self._sample_names = self.original_filenames

    @property
    def filenames(self) -> tuple[str, ...]:
        """Sample identifiers, including virtual paired-sequence names."""
        return self._sample_names

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(_parse_name(record[0])[0] for record in self._records)

    @property
    def angles(self) -> tuple[int, ...]:
        return tuple(int(_parse_name(record[0])[2]) for record in self._records)

    @property
    def seq_ids(self) -> tuple[str, ...]:
        return tuple(_parse_name(record[0])[1] for record in self._records)

    def __len__(self) -> int:
        return len(self._records)

    def _index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return index

    def _load_image(self, folder: str, name: str) -> np.ndarray:
        with Image.open(self.split_root / folder / name) as image:
            return np.array(image.convert("RGB"))

    def get_raw_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return uint8 HWC before/after and int64 HW binary labels."""
        before_name, after_name, mask_name = self._records[self._index(index)]
        before = self._load_image("t0", before_name)
        after = self._load_image("t1", after_name)
        with Image.open(self.split_root / "mask" / mask_name) as image:
            mask = np.array(image)
        if mask.ndim != 2:
            raise ValueError(f"VL-CMU-CD mask must be single-channel: {mask_name}")
        if before.shape != after.shape or mask.shape != before.shape[:2]:
            raise ValueError(f"VL-CMU-CD image/mask dimensions differ: {before_name}")
        return before, after, (mask > 0).astype(np.int64)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        before, after, labels = self.get_raw_sample(index)
        if self.hflip_prob and torch.rand(()).item() < self.hflip_prob:
            before, after, labels = (
                np.flip(array, axis=1).copy() for array in (before, after, labels)
            )
        batch = self.processor(
            before,
            after,
            segmentation_maps=labels,
            input_data_format="channels_last",
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in batch.items()}


class DiffViewVLCMUCD(VLCMUCD):
    """Pair shifted sequence indices while retaining group and angle matches.

    This follows ``Diff_VL_CMU_CD``: train/val use an index stride of four,
    test uses one, and labels come from the t0 image. Pairs crossing a group
    boundary or changing camera angle are discarded.
    """

    def __init__(self, root: str | Path | None = None, mode: str = "train", *,
                 adjacent_distance: int = 1, **kwargs):
        if isinstance(adjacent_distance, bool) or not isinstance(adjacent_distance, int):
            raise ValueError("adjacent_distance must be a non-zero integer")
        if adjacent_distance == 0:
            raise ValueError("adjacent_distance must be non-zero")
        super().__init__(root, mode, **kwargs)

        stride = 1 if self.mode == "test" else 4
        shift = adjacent_distance * stride
        records = []
        sample_names = []
        for before_index, before_name in enumerate(self.original_filenames):
            after_index = before_index + shift
            if not 0 <= after_index < len(self.original_filenames):
                continue
            after_name = self.original_filenames[after_index]
            before_group, before_seq, before_angle = _parse_name(before_name)
            after_group, after_seq, after_angle = _parse_name(after_name)
            if before_group != after_group or before_angle != after_angle:
                continue
            records.append((before_name, after_name, before_name))
            sample_names.append(
                f"{before_group}_1_{before_seq}.{after_seq}_{before_angle}.png"
            )
        if not records:
            raise ValueError(
                "No shifted VL-CMU-CD pairs match both group and angle; "
                "check the split, filenames and adjacent_distance."
            )
        self.adjacent_distance = adjacent_distance
        self._records = tuple(records)
        self._sample_names = tuple(sample_names)

    @property
    def seq_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{_parse_name(before)[1]}.{_parse_name(after)[1]}"
            for before, after, _ in self._records
        )
