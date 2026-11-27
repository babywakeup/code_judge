"""View the exact processed RGB pairs and labels from ``segment.data``."""

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

from segment.data import DiffViewVLCMUCD, VLCMUCD

STATIC = Path(__file__).parent / "static"
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data_" / "VL-CMU-CD2"


def png_url(array):
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class Viewer:
    """Small cached adapter over the model-ready VL-CMU-CD datasets."""

    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root).expanduser().resolve()
        self.lock = Lock()
        self.root = self.dataset("paired", "test", 1).root

    @staticmethod
    def options(variant="paired", split="test", distance="1"):
        if variant not in {"paired", "diff"}:
            raise ValueError("variant must be paired or diff")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        try:
            distance = int(distance)
        except (TypeError, ValueError) as exc:
            raise ValueError("distance must be an integer") from exc
        if distance not in range(-3, 4):
            raise ValueError("distance must be between -3 and 3")
        if variant == "diff" and distance == 0:
            raise ValueError("Diff view requires a non-zero distance")
        return variant, split, distance

    @lru_cache(maxsize=12)
    def dataset(self, variant, split, distance):
        cls = DiffViewVLCMUCD if variant == "diff" else VLCMUCD
        options = {"adjacent_distance": distance} if variant == "diff" else {}
        return cls(self.root, mode=split, **options)

    def info(self, **params):
        options = self.options(**params)
        dataset = self.dataset(*options)
        groups = dataset.group_ids
        group_starts = [i for i, group in enumerate(groups) if i == 0 or group != groups[i - 1]]
        return dict(root=str(dataset.root), variant=options[0], split=options[1],
                    total=len(dataset), originals=len(group_starts),
                    group_starts=group_starts, distance=options[2],
                    stride=4 if options[1] in {"train", "val"} else 1)

    def sample(self, index="0", **params):
        options = self.options(**params)
        try:
            index = int(index)
        except (TypeError, ValueError) as exc:
            raise ValueError("index must be an integer") from exc
        with self.lock:
            return self._sample(index, *options)

    @lru_cache(maxsize=8)
    def _sample(self, index, variant, split, distance):
        dataset = self.dataset(variant, split, distance)
        if not 0 <= index < len(dataset):
            raise IndexError("Sample index out of range")

        before_name, after_name, _ = dataset._records[index]
        with Image.open(dataset.split_root / "t0" / before_name) as image:
            original_size = list(image.size)
        with torch.no_grad():
            sample = dataset[index]
        before, after = (
            sample[key].permute(1, 2, 0).mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
            for key in ("pixel_values", "pixel_values_pair")
        )
        labels = sample["labels"].numpy()
        foreground = labels == 1
        label_rgb = np.repeat((foreground.astype(np.uint8) * 255)[..., None], 3, axis=-1)
        overlay = before.copy()
        overlay[foreground] = np.rint(
            before[foreground] * 0.55 + np.array([255, 75, 110]) * 0.45
        ).astype(np.uint8)
        height, width = labels.shape
        return dict(
            index=index,
            total=len(dataset),
            filename=dataset.filenames[index],
            before_file=before_name,
            after_file=after_name,
            group=dataset.group_ids[index],
            sequences=dataset.seq_ids[index],
            angle=dataset.angles[index],
            variant=variant,
            split=split,
            distance=distance,
            stride=4 if split in {"train", "val"} else 1,
            original_size=original_size,
            processed_size=[width, height],
            changed_pixels=int(foreground.sum()),
            pixels=int(labels.size),
            changed_ratio=float(foreground.mean()),
            images=dict(before=png_url(before), after=png_url(after),
                        label=png_url(label_rgb), overlay=png_url(overlay)),
        )


def make_server(viewer, host="127.0.0.1", port=7878):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path in {"/api/info", "/api/sample"}:
                    query = parse_qs(url.query, keep_blank_values=True)
                    allowed = {"variant", "split", "distance"}
                    if url.path == "/api/sample":
                        allowed.add("index")
                    if set(query) - allowed or any(len(values) != 1 for values in query.values()):
                        raise ValueError("Unexpected or repeated parameters")
                    params = {key: values[0] for key, values in query.items()}
                    data = viewer.info(**params) if url.path == "/api/info" else viewer.sample(**params)
                    return self.send_json(data)

                routes = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
                if url.path in routes:
                    path = STATIC / routes[url.path]
                    content_type = {
                        ".html": "text/html",
                        ".js": "text/javascript",
                        ".css": "text/css",
                    }[path.suffix]
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
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_body(status, body, "application/json; charset=utf-8")

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description="VL-CMU-CD 处理后 RGB 配对与标签查看器")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7878)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        viewer = Viewer(args.dataset)
        server = make_server(viewer, args.host, args.port)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    address = f"http://{args.host}:{server.server_port}"
    print(f"VL-CMU-CD：{viewer.root}\nViewer：{address}（Ctrl+C 退出）", flush=True)
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
