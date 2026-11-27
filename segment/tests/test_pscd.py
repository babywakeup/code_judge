"""Offline PSCD data protocol tests using generated images."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import Dinov2Config

from segment import SceneChangeConfig, SceneChangeForSemanticSegmentation
from segment.data import PSCD, CroppedPSCD, DiffViewPSCD
from segment.data.splits import PSCD_SPLITS


class PSCDTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "PSCD"
        self.name = "00000000.png"
        columns = np.arange(1024, dtype=np.uint16) % 256
        self.before = np.broadcast_to(columns[None, :, None], (224, 1024, 3)).astype(np.uint8).copy()
        self.after = 255 - self.before
        self.mask = np.full((224, 1024), 255, dtype=np.uint8)
        self.write(self.before, self.after, self.mask)

    def write(self, before, after, mask):
        for folder, array in (("t0", before), ("t1", after), ("mask_t0", mask),
                              ("mask_t1", np.zeros_like(mask))):
            path = self.root / folder
            path.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path / self.name)

    def dataset(self, cls=CroppedPSCD, **kwargs):
        return cls(self.root, filenames=[self.name], **kwargs)

    def test_splits(self):
        self.assertEqual({k: len(v) for k, v in PSCD_SPLITS.items()},
                         {"train": 616, "val": 77, "test": 77})
        groups = [set(v) for v in PSCD_SPLITS.values()]
        self.assertEqual(len(set.union(*groups)), 770)
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(PSCD_SPLITS["val"][0], "00000001.png")
        self.assertEqual(PSCD_SPLITS["test"][0], "00000007.png")

    def test_full_and_nested_root(self):
        dataset = PSCD(self.root.parent, filenames=[self.name])
        self.assertEqual(dataset.root, self.root)
        before, _, labels = dataset.get_raw_sample(0)
        np.testing.assert_array_equal(before, self.before)
        self.assertEqual(labels.dtype, np.int64)
        self.assertTrue(np.all(labels == 1))
        sample = dataset[0]
        self.assertEqual(sample["pixel_values"].shape, (3, 224, 1022))
        self.assertEqual(sample["labels"].dtype, torch.int64)
        self.assertTrue(torch.all(self.dataset(PSCD, mask_source="t1")[0]["labels"] == 0))

    def test_aligned_crops_and_batch(self):
        dataset = self.dataset()
        self.assertEqual(len(dataset), 15)
        self.assertEqual(dataset.stride, 57)
        self.assertEqual(dataset.cropped_prefix, 1)
        self.assertEqual(dataset.filenames[-1], "00000000.14.png")
        for index in (0, 7, 14):
            before, after, labels = dataset.get_raw_sample(index)
            start = 1 + index * 57
            np.testing.assert_array_equal(before, self.before[:, start:start + 224])
            np.testing.assert_array_equal(after, self.after[:, start:start + 224])
            self.assertEqual(labels.shape, (224, 224))
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(batch["pixel_values"].shape, (2, 3, 224, 224))
        self.assertEqual(batch["pixel_values"].dtype, torch.float32)
        self.assertEqual(batch["labels"].shape, (2, 224, 224))
        self.assertLessEqual(batch["pixel_values"].max().item(), 1)
        self.assertGreaterEqual(batch["pixel_values"].min().item(), 0)

    def test_diff_view_both_directions_and_zero(self):
        for distance in (-2, -1, 0, 1, 2):
            dataset = self.dataset(DiffViewPSCD, adjacent_distance=distance)
            self.assertEqual(len(dataset), 15 - abs(distance))
            for index in (0, len(dataset) - 1):
                before, after, labels = dataset.get_raw_sample(index)
                crop = max(0, -distance) + index
                start = 1 + crop * 57
                paired_start = start + distance * 57
                np.testing.assert_array_equal(before, self.before[:, start:start + 224])
                np.testing.assert_array_equal(after, self.after[:, paired_start:paired_start + 224])
                expected = np.ones((224, 224), dtype=np.int64)
                if distance > 0:
                    expected[:, :distance * 57] = 0
                elif distance < 0:
                    expected[:, distance * 57:] = 0
                np.testing.assert_array_equal(labels, expected)

    def test_resize_preserves_binary_labels(self):
        before = np.full((288, 1024, 3), 128, dtype=np.uint8)
        mask = np.zeros((288, 1024), dtype=np.uint8)
        mask[:, 512:] = 255
        self.write(before, before, mask)
        sample = self.dataset().get_raw_sample(14)
        self.assertEqual(sample[0].shape, (224, 224, 3))
        self.assertTrue(set(np.unique(sample[2])) <= {0, 1})
        with self.assertRaises(ValueError):
            self.dataset(resize_panorama=False)[0]

    def test_synchronized_flip(self):
        plain = self.dataset()[0]
        flipped = self.dataset(hflip_prob=1)[0]
        for key in plain:
            torch.testing.assert_close(flipped[key], plain[key].flip(-1))

    def test_validation(self):
        for distance in (-15, -4, 4, 15, 1.5):
            with self.assertRaises(ValueError):
                self.dataset(DiffViewPSCD, adjacent_distance=distance)
        for crop_num in (1, 802, 1.5):
            with self.assertRaises(ValueError):
                self.dataset(crop_num=crop_num)
        for kwargs in ({"mode": "bad"}, {"hflip_prob": 2}, {"mask_source": "bad"}):
            with self.assertRaises(ValueError):
                self.dataset(**kwargs)
        with self.assertRaises(FileNotFoundError):
            PSCD(self.root, filenames=["missing.png"])
        with self.assertRaises(ValueError):
            PSCD(self.root, filenames=["../bad.png"])
        with self.assertRaises(IndexError):
            self.dataset()[15]
        torch.testing.assert_close(self.dataset()[-1]["labels"], self.dataset()[14]["labels"])

    def test_model_loss_and_backward(self):
        torch.set_num_threads(2)
        backbone = Dinov2Config(hidden_size=32, num_hidden_layers=1,
                                num_attention_heads=4, image_size=224)
        config = SceneChangeConfig(backbone_config=backbone, feature_layer=0,
                                   classifier_relu=False, dropout=0)
        model = SceneChangeForSemanticSegmentation(config)
        batch = next(iter(DataLoader(self.dataset(), batch_size=1)))
        output = model(**batch)
        self.assertEqual(output.logits.shape, (1, 2, 224, 224))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.classifier.weight.grad)


if __name__ == "__main__":
    unittest.main()
