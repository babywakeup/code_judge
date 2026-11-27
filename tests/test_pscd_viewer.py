"""Processed RGB triptych and read-only HTTP API contracts."""

import base64
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from PIL import Image
import torch

from apps.pscd_viewer.server import Viewer, make_server


def decode(url):
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
        return np.array(image)


class PSCDViewerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "PSCD"
        columns = np.arange(1024, dtype=np.uint16) % 256
        before = np.broadcast_to(columns[None, :, None], (224, 1024, 3)).astype(np.uint8)
        mask = np.zeros((224, 1024), dtype=np.uint8)
        mask[:, 100:800] = 255
        for folder, array in (("t0", before), ("t1", 255 - before),
                              ("mask_t0", mask), ("mask_t1", 255 - mask)):
            path = self.root / folder
            path.mkdir(parents=True)
            Image.fromarray(array).save(path / "00000000.png")
        self.viewer = Viewer(self.root.parent, filenames=["00000000.png"])

    def test_rgb_matches_actual_processor_output(self):
        result = self.viewer.sample(index=3)
        tensor_sample = self.viewer.dataset("cropped", "test", 1, "t0")[3]
        for key, tensor_key in (("before", "pixel_values"), ("after", "pixel_values_pair")):
            expected = tensor_sample[tensor_key].permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
            np.testing.assert_array_equal(decode(result["images"][key]), expected)
        labels = decode(result["images"]["label"])
        self.assertEqual(labels.shape, (224, 224, 3))
        np.testing.assert_array_equal(labels[..., 0], tensor_sample["labels"].numpy() * 255)
        self.assertEqual(result["filename"], "00000000.3.png")
        self.assertEqual(result["processed_size"], [224, 224])
        self.assertEqual(result["changed_pixels"], int(tensor_sample["labels"].sum()))
        self.assertIs(self.viewer.sample(index=3), result)

    def test_full_shows_patch_center_crop(self):
        result = self.viewer.sample(variant="full")
        self.assertEqual(result["original_size"], [1024, 224])
        self.assertEqual(result["processed_size"], [1022, 224])
        self.assertEqual(result["total"], 1)
        self.assertIsNone(result["crop_before"])
        self.assertEqual(decode(result["images"]["before"]).shape, (224, 1022, 3))

    def test_shifted_label_and_mask_source_overlay(self):
        result = self.viewer.sample(variant="diff", distance="-1")
        self.assertEqual((result["crop_before"], result["crop_after"]), (1, 0))
        self.assertEqual(result["total"], 14)
        label = decode(result["images"]["label"])
        self.assertTrue(np.all(label[:, -57:] == 0))
        alternate = self.viewer.sample(mask="t1")
        foreground = decode(alternate["images"]["label"])[..., 0] > 0
        after = decode(alternate["images"]["after"])
        overlay = decode(alternate["images"]["overlay"])
        np.testing.assert_array_equal(overlay[~foreground], after[~foreground])
        expected = np.rint(after[foreground] * .55 + np.array([255, 75, 110]) * .45).astype(np.uint8)
        np.testing.assert_array_equal(overlay[foreground], expected)

    def test_invalid_parameters_and_bounds(self):
        for params in ({"variant": "invalid"}, {"split": "invalid"},
                       {"distance": "4"}, {"distance": "bad"},
                       {"variant": "diff", "mask": "t1"}):
            with self.assertRaises(ValueError):
                self.viewer.info(**params)
        for index in (-1, 15):
            with self.assertRaises(IndexError):
                self.viewer.sample(index=index)

    def test_high_resolution_resize_indicator(self):
        for folder in ("t0", "t1"):
            Image.new("RGB", (4096, 1152), (120, 80, 30)).save(self.root / folder / "00000000.png")
        for folder in ("mask_t0", "mask_t1"):
            Image.new("L", (4096, 1152), 255).save(self.root / folder / "00000000.png")
        result = self.viewer.sample()
        self.assertTrue(result["resized"])
        self.assertEqual(result["original_size"], [4096, 1152])
        self.assertEqual(result["processed_size"], [224, 224])

    def test_http_resources_and_security(self):
        server = make_server(self.viewer, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        for path, content_type in (("/", "text/html"), ("/app.js", "text/javascript"),
                                   ("/style.css", "text/css"), ("/api/info", "application/json")):
            with urlopen(base + path) as response:
                self.assertIn(content_type, response.headers["Content-Type"])
                self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
                self.assertTrue(response.read())
        with urlopen(base + "/api/sample?variant=diff&distance=1&index=0") as response:
            sample = json.load(response)
        self.assertEqual(sample["crop_after"], 1)
        self.assertEqual(decode(sample["images"]["label"]).shape, (224, 224, 3))
        for path, status in (("/../../segment/LICENSE", 404), ("/image/../../secret", 404),
                             ("/api/sample?index=-1", 400), ("/api/info?root=/etc", 400),
                             ("/api/info?split=test&split=train", 400)):
            with self.assertRaises(HTTPError) as caught:
                urlopen(base + path)
            self.assertEqual(caught.exception.code, status)

    @unittest.skipUnless(shutil.which("node"), "Node.js required for frontend syntax validation")
    def test_frontend_javascript_syntax(self):
        source = Path(__file__).resolve().parents[1] / "apps/pscd_viewer/static/app.js"
        result = subprocess.run([shutil.which("node"), "--check", str(source)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
