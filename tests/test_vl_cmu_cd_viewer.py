"""Processed RGB triptych and read-only HTTP API contracts for VL-CMU-CD."""

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

from apps.vl_cmu_cd_viewer.server import Viewer, make_server
from segment.data import DiffViewVLCMUCD, VLCMUCD


def decode(url):
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
        return np.array(image)


class VLCMUCDViewerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.root = self.parent / "VL-CMU-CD-binary255"
        for split in ("train", "test"):
            for folder in ("t0", "t1", "mask"):
                (self.root / split / folder).mkdir(parents=True)
        for seq in range(5):
            for angle in (0, 90, 180, 270):
                self.write("train", f"001_1_{seq:02}_{angle}.png", seq)
        self.write("train", "013_1_00_0.png", 9)
        self.write("test", "000_1_00_0.png", 1)
        self.write("test", "000_1_01_0.png", 2)
        self.write("test", "006_1_00_0.png", 3)
        self.viewer = Viewer(self.parent)

    def write(self, split, name, sequence):
        before = np.full((28, 28, 3), sequence * 10, dtype=np.uint8)
        after = np.full((28, 28, 3), sequence * 10 + 60, dtype=np.uint8)
        mask = np.zeros((28, 28), dtype=np.uint8)
        mask[:, :4] = 127
        for folder, array in (("t0", before), ("t1", after), ("mask", mask)):
            Image.fromarray(array).save(self.root / split / folder / name)

    def test_processed_images_match_dataset(self):
        result = self.viewer.sample(index=0)
        source = VLCMUCD(self.parent, mode="test")[0]
        for key, source_key in (("before", "pixel_values"), ("after", "pixel_values_pair")):
            expected = source[source_key].permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
            np.testing.assert_array_equal(decode(result["images"][key]), expected)
        label = decode(result["images"]["label"])
        np.testing.assert_array_equal(label[..., 0], source["labels"].numpy() * 255)
        self.assertEqual(result["processed_size"], [28, 28])
        self.assertEqual(result["changed_pixels"], int(source["labels"].sum()))
        self.assertIs(self.viewer.sample(index=0), result)

    def test_info_and_diff_pairs(self):
        info = self.viewer.info(variant="paired", split="test")
        self.assertEqual((info["total"], info["originals"]), (3, 2))
        diff = self.viewer.sample(index=0, variant="diff", split="test", distance="1")
        self.assertEqual(diff["filename"], "000_1_00.01_0.png")
        self.assertEqual((diff["before_file"], diff["after_file"]),
                         ("000_1_00_0.png", "000_1_01_0.png"))
        self.assertEqual(diff["group"], "000")
        self.assertEqual(diff["sequences"], "00.01")
        self.assertEqual(len(DiffViewVLCMUCD(self.parent, mode="test")), 1)
        overlay = decode(diff["images"]["overlay"])
        before = decode(diff["images"]["before"])
        foreground = decode(diff["images"]["label"])[..., 0] > 0
        np.testing.assert_array_equal(overlay[~foreground], before[~foreground])

    def test_http_resources_and_validation(self):
        server = make_server(self.viewer, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        for path, content_type in (("/", "text/html"), ("/app.js", "text/javascript"),
                                   ("/style.css", "text/css"), ("/api/info", "application/json")):
            with urlopen(base + path) as response:
                self.assertIn(content_type, response.headers["Content-Type"])
                self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
                self.assertTrue(response.read())
        with urlopen(base + "/api/sample?variant=diff&split=test&distance=1&index=0") as response:
            sample = json.load(response)
        self.assertEqual(sample["after_file"], "000_1_01_0.png")
        for path, status in (("/../../segment/LICENSE", 404),
                             ("/api/sample?index=-1", 400),
                             ("/api/info?root=/etc", 400),
                             ("/api/info?split=test&split=train", 400),
                             ("/api/info?variant=diff&distance=0", 400)):
            with self.assertRaises(HTTPError) as caught:
                urlopen(base + path)
            self.assertEqual(caught.exception.code, status)

    @unittest.skipUnless(shutil.which("node"), "Node.js required for frontend syntax validation")
    def test_frontend_javascript_syntax(self):
        source = Path(__file__).resolve().parents[1] / "apps/vl_cmu_cd_viewer/static/app.js"
        result = subprocess.run([shutil.which("node"), "--check", str(source)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
