"""Offline VL-CMU-CD loader tests using generated paired images."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from segment.data import DiffViewVLCMUCD, VLCMUCD


class VLCMUCDTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.root = self.parent / "VL-CMU-CD-binary255"
        for split in ("train", "test"):
            for folder in ("t0", "t1", "mask"):
                (self.root / split / folder).mkdir(parents=True)

        # Five sequences with four views each exercise the reference train stride of 4.
        for seq in range(5):
            for angle in (0, 90, 180, 270):
                self.write("train", f"001_1_{seq:02}_{angle}.png", seq)
        self.write("train", "013_1_00_0.png", 9)
        # The test stride is 1; the final sample also checks group-boundary filtering.
        self.write("test", "000_1_00_0.png", 1)
        self.write("test", "000_1_01_0.png", 2)
        self.write("test", "006_1_00_0.png", 3)

    def write(self, split, name, sequence):
        before = np.full((28, 28, 3), sequence * 10, dtype=np.uint8)
        after = np.full((28, 28, 3), sequence * 10 + 60, dtype=np.uint8)
        mask = np.zeros((28, 28), dtype=np.uint8)
        mask[:, :4] = 127
        for folder, array in (("t0", before), ("t1", after), ("mask", mask)):
            Image.fromarray(array).save(self.root / split / folder / name)

    def test_fixed_group_splits_and_root_detection(self):
        train = VLCMUCD(self.parent, mode="train")
        val = VLCMUCD(self.root, mode="val")
        test = VLCMUCD(self.parent, mode="test")
        self.assertEqual(len(train), 20)
        self.assertEqual(set(train.group_ids), {"001"})
        self.assertEqual(len(val), 1)
        self.assertEqual(set(val.group_ids), {"013"})
        self.assertEqual(len(test), 3)
        self.assertEqual(set(test.group_ids), {"000", "006"})
        self.assertEqual(train.root, self.root.resolve())

    def test_raw_and_processed_sample(self):
        dataset = VLCMUCD(self.parent, mode="train")
        before, after, labels = dataset.get_raw_sample(0)
        self.assertEqual(before.shape, (28, 28, 3))
        self.assertEqual(after.shape, (28, 28, 3))
        self.assertEqual(labels.dtype, np.int64)
        self.assertEqual(set(np.unique(labels)), {0, 1})

        sample = dataset[0]
        self.assertEqual(sample["pixel_values"].shape, (3, 28, 28))
        self.assertEqual(sample["pixel_values_pair"].shape, (3, 28, 28))
        self.assertEqual(sample["labels"].shape, (28, 28))
        self.assertEqual(sample["labels"].dtype, torch.int64)
        self.assertGreaterEqual(sample["pixel_values"].min().item(), 0)
        self.assertLessEqual(sample["pixel_values"].max().item(), 1)

        batch = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(batch["pixel_values"].shape, (2, 3, 28, 28))
        self.assertEqual(batch["labels"].shape, (2, 28, 28))

    def test_shifted_train_and_test_pairs(self):
        train = DiffViewVLCMUCD(self.parent, mode="train", adjacent_distance=1)
        self.assertEqual(len(train), 16)
        self.assertEqual(train.filenames[0], "001_1_00.01_0.png")
        before, after, labels = train.get_raw_sample(0)
        self.assertTrue(np.all(before == 0))
        self.assertTrue(np.all(after == 70))
        self.assertEqual(labels[:, :4].sum(), 28 * 4)

        test = DiffViewVLCMUCD(self.parent, mode="test", adjacent_distance=1)
        self.assertEqual(len(test), 1)
        self.assertEqual(test.filenames, ("000_1_00.01_0.png",))
        reverse = DiffViewVLCMUCD(self.parent, mode="test", adjacent_distance=-1)
        self.assertEqual(len(reverse), 1)
        before, after, _ = reverse.get_raw_sample(0)
        self.assertTrue(np.all(before == 20))
        self.assertTrue(np.all(after == 70))

    def test_errors_and_indexing(self):
        for mode in ("invalid",):
            with self.assertRaises(ValueError):
                VLCMUCD(self.parent, mode=mode)
        with self.assertRaises(ValueError):
            DiffViewVLCMUCD(self.parent, mode="test", adjacent_distance=0)
        with self.assertRaises(ValueError):
            DiffViewVLCMUCD(self.parent, mode="test", adjacent_distance=1.5)
        with self.assertRaises(IndexError):
            VLCMUCD(self.parent, mode="test")[3]
        with self.assertRaises(FileNotFoundError):
            VLCMUCD(self.parent, mode="test", filenames=["000_1_99_0.png"])
        with self.assertRaises(ValueError):
            VLCMUCD(self.parent, mode="test", filenames=["../invalid.png"])
        torch.testing.assert_close(
            VLCMUCD(self.parent, mode="test")[-1]["labels"],
            VLCMUCD(self.parent, mode="test")[2]["labels"],
        )


if __name__ == "__main__":
    unittest.main()
