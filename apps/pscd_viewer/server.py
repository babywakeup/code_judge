"""View actual segment.data outputs, without loading models or writing files."""

import argparse
import base64
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit
import webbrowser

import numpy as np
from PIL import Image
import torch

from segment.data import CroppedPSCD, DiffViewPSCD, PSCD

STATIC = Path(__file__).parent / "static"
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data_" / "PSCD"


def png_url(array):
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class Viewer:
    def __init__(self, root=DEFAULT_ROOT, *, filenames=None):
        self.root = Path(root).resolve()
        self.filenames = tuple(filenames) if filenames is not None else None
        self.lock = Lock()
        # Fail early for incorrect roots, rather than serving an unusable page.
        initial = self.dataset("cropped", "test", 1, "t0")
        self.root = initial.root

    @staticmethod
    def options(variant="cropped", split="test", distance="1", mask="t0"):
        if variant not in {"full", "cropped", "diff"}:
            raise ValueError("Unknown dataset variant")
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        if mask not in {"t0", "t1"}:
            raise ValueError("mask must be t0 or t1")
        distance = int(distance)
        if distance not in range(-3, 4):
            raise ValueError("distance must be between -3 and 3")
        if variant == "diff" and mask != "t0":
            raise ValueError("Diff-view labels must use t0 coordinates")
        return variant, split, distance, mask

    @lru_cache(maxsize=12)
    def dataset(self, variant, split, distance, mask):
        cls = {"full": PSCD, "cropped": CroppedPSCD, "diff": DiffViewPSCD}[variant]
        options = dict(mask_source=mask, filenames=self.filenames)
        if variant == "diff":
            options["adjacent_distance"] = distance
        return cls(self.root, mode=split, **options)

    def info(self, **params):
        options = self.options(**params)
        dataset = self.dataset(*options)
        return dict(root=str(dataset.root), variant=options[0], split=options[1],
                    total=len(dataset), originals=len(dataset.original_filenames),
                    crops_per_image=getattr(dataset, "crops_per_image", getattr(dataset, "crop_num", 1)),
                    stride=getattr(dataset, "stride", None), distance=options[2], mask=options[3])

    def sample(self, index="0", **params):
        options = self.options(**params)
        index = int(index)
        with self.lock:
            return self._sample(index, *options)

    @lru_cache(maxsize=6)
    def _sample(self, index, variant, split, distance, mask):
        dataset = self.dataset(variant, split, distance, mask)
        if not 0 <= index < len(dataset):
            raise IndexError("Sample index out of range")
        per_image = getattr(dataset, "crops_per_image", getattr(dataset, "crop_num", 1))
        image_index, crop_offset = divmod(index, per_image)
        original = dataset.original_filenames[image_index]
        crop_before = crop_after = None
        if variant != "full":
            crop_before = crop_offset + getattr(dataset, "first_crop", 0)
            crop_after = crop_before + (distance if variant == "diff" else 0)
        with Image.open(dataset.root / "t0" / original) as image:
            original_size = list(image.size)
        # Deliberately use __getitem__: show exactly the post-processor geometry,
        # not raw PNGs or a separate approximation of the dataset pipeline.
        with torch.no_grad():
            sample = dataset[index]
        before, after = (
            sample[key].permute(1, 2, 0).mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
            for key in ("pixel_values", "pixel_values_pair")
        )
        labels = sample["labels"].numpy()
        foreground = labels == 1
        label_rgb = np.repeat((foreground.astype(np.uint8) * 255)[..., None], 3, axis=-1)
        overlay_base = before if mask == "t0" else after
        overlay = overlay_base.copy()
        overlay[foreground] = np.rint(overlay_base[foreground] * 0.55 + np.array([255, 75, 110]) * 0.45).astype(np.uint8)
        height, width = labels.shape
        suffix = "" if crop_before is None else (
            f".{crop_before}-{crop_after}" if variant == "diff" else f".{crop_before}"
        )
        return dict(index=index, total=len(dataset), original=original,
                    filename=f"{Path(original).stem}{suffix}.png",
                    image_index=image_index, crop_before=crop_before, crop_after=crop_after,
                    variant=variant, split=split, mask=mask, distance=distance,
                    stride=getattr(dataset, "stride", None),
                    original_size=original_size, processed_size=[width, height],
                    resized=variant != "full" and original_size != [1024, 224],
                    changed_pixels=int(foreground.sum()), pixels=int(labels.size),
                    changed_ratio=float(foreground.mean()),
                    images=dict(before=png_url(before), after=png_url(after),
                                label=png_url(label_rgb), overlay=png_url(overlay)))


def make_server(viewer, host="127.0.0.1", port=7877):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path in {"/api/info", "/api/sample"}:
                    query = parse_qs(url.query, keep_blank_values=True)
                    allowed = {"variant", "split", "distance", "mask"}
                    if url.path == "/api/sample":
                        allowed.add("index")
                    if set(query) - allowed or any(len(v) != 1 for v in query.values()):
                        raise ValueError("Unexpected or repeated parameters")
                    params = {key: value[0] for key, value in query.items()}
                    data = viewer.info(**params) if url.path == "/api/info" else viewer.sample(**params)
                    return self.send_json(data)
                routes = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
                if url.path in routes:
                    path = STATIC / routes[url.path]
                    content_type = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[path.suffix]
                    return self.send_body(200, path.read_bytes(), content_type + "; charset=utf-8")
                self.send_json({"error": "Not found"}, status=404)
            except (ValueError, TypeError, IndexError) as exc:
                self.send_json({"error": str(exc)}, status=400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError:
                self.send_json({"error": "Dataset image unavailable"}, status=404)

        def send_body(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, data, status=200):
            self.send_body(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode(),
                           "application/json; charset=utf-8")

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description="PSCD 处理后 Before / After / Label RGB 查看器")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7877)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        viewer = Viewer(args.dataset)
        server = make_server(viewer, args.host, args.port)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    address = f"http://{args.host}:{server.server_port}"
    print(f"PSCD：{viewer.root}\nViewer：{address}（Ctrl+C 退出）", flush=True)
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
