"""Export normalized JSON/raw inputs for ptm_debug, without running matching."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from periodic_template_matcher.config import MatcherConfig
from periodic_template_matcher._internal.io import workflow
from periodic_template_matcher_cpp import _native


def load_config_inputs(path: Path) -> tuple[MatcherConfig, dict, dict]:
    """Mirror the IO-only prefix of run_config; never call match/save_result."""
    # Current workflow has no load_job. Prefer it if a future version exposes
    # the explicit (MatcherConfig, match-keyword-dict, metadata-dict) contract.
    loader = getattr(workflow, "load_job", None)
    if callable(loader):
        job = loader(path)
        if (not isinstance(job, tuple) or len(job) != 3
                or not isinstance(job[0], MatcherConfig)
                or not isinstance(job[1], dict) or not isinstance(job[2], dict)):
            raise TypeError("load_job must return (MatcherConfig, match kwargs, metadata)")
        return job
    spec = json.loads(path.read_text(encoding="utf-8-sig"))

    def resolve(value: str) -> Path:
        return (path.parent / value).resolve()

    cfg = MatcherConfig.from_options(
        dict(spec.get("options", {})), object_order_axis=workflow.layout_axis(spec["templates"]))
    cfg.label_mode = spec.get("label_mode", cfg.label_mode)
    cfg.validate()
    specs, aliases = workflow.unique_templates(spec["templates"], resolve)
    layouts = [workflow.object_layout(s, cfg.ordered_object_class)
               if cfg.ordered_object_class is not None or "instance_layouts" in s else None
               for s in specs]
    images = [workflow.read_image(resolve(s["image"])) for s in specs]

    def optional_mask(item: dict, key: str):
        return (workflow.read_image(resolve(item[key]), cv2.IMREAD_GRAYSCALE)
                if item.get(key) else None)

    kwargs = dict(
        templates=images,
        masks=[workflow.load_labels(s, resolve, im.shape[:2]) for s, im in zip(specs, images)],
        target=workflow.read_image(resolve(spec["target"])),
        template_valids=[optional_mask(s, "valid_mask") for s in specs],
        annotation_valids=[optional_mask(s, "annotation_valid") for s in specs],
        target_valid=optional_mask(spec, "target_valid_mask"),
        template_valid_heights=[s.get("valid_height") for s in specs],
        target_valid_height=spec.get("target_valid_height"),
        template_objects=[workflow.load_objects(s, resolve, im.shape[:2],
                          cfg.ordered_object_class, cfg.object_cycle)
                          for s, im in zip(specs, images)]
                         if cfg.ordered_object_class is not None else None,
    )
    metadata = dict(config_path=str(path), target=str(resolve(spec["target"])),
                    template_images=[str(resolve(s["image"])) for s in specs],
                    template_names=[s["name"] for s in specs], template_aliases=aliases,
                    template_instance_layouts=layouts, input_template_count=len(spec["templates"]),
                    io_loader="workflow.run_config IO prefix (load_job absent)")
    return cfg, kwargs, metadata


def smoke_inputs(objects: bool = False) -> tuple[MatcherConfig, dict, dict]:
    image = np.random.default_rng(42).integers(0, 256, (96, 112, 3), dtype=np.uint8)
    labels = np.zeros(image.shape[:2], dtype=np.uint16)
    labels[30:50, 30:50] = 1
    cfg = MatcherConfig(angles=[0.0], scales=[1.0], template_workers=1,
                        periodic_completion=False)
    refs = None
    if objects:
        cfg.ordered_object_class = 2
        refs = [[]]
        for index, identity in enumerate(cfg.object_cycle):
            mask = np.zeros(labels.shape, dtype=np.uint8)
            mask[60:80, 10 + index * 30:25 + index * 30] = 1
            labels[mask > 0] = 2
            refs[0].append(dict(identity=identity, mask=mask, shape_index=index))
    cfg.validate()
    return cfg, dict(templates=[image], masks=[labels], target=image.copy(),
                     template_valid_heights=[90], target_valid_height=90,
                     template_objects=refs), dict(kind="synthetic-smoke", seed=42,
                                                   objects=objects, io_loader="synthetic")


def write_fixture(output: Path, cfg: MatcherConfig, kwargs: dict, metadata: dict) -> Path:
    """Normalize at the native boundary, then serialize cv::Mat-compatible bytes."""
    config = asdict(cfg)
    normalized = _native.prepare_inputs(cfg=config, **kwargs)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "fixture.json"
    if manifest.exists():
        raise FileExistsError(f"Refusing to replace existing fixture: {manifest}")
    arrays = []
    depth_codes = {np.dtype("uint8"): 0, np.dtype("int8"): 1, np.dtype("uint16"): 2,
                   np.dtype("int16"): 3, np.dtype("int32"): 4,
                   np.dtype("float32"): 5, np.dtype("float64"): 6}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype == np.bool_:
                value = value.astype(np.uint8)
            if value.ndim not in (2, 3) or min(value.shape) <= 0:
                raise ValueError("Fixture arrays must be nonempty 2D/3D arrays")
            depth = depth_codes[value.dtype.newbyteorder("=")]
            channels = value.shape[2] if value.ndim == 3 else 1
            if not 1 <= channels <= 512:
                raise ValueError("Invalid OpenCV channel count")
            # Windows SDK/debug executable is little-endian; preserve BGR/BGRA.
            raw = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
            name = f"array_{len(arrays):04d}.raw"
            with (output / name).open("xb") as handle:
                handle.write(raw.tobytes(order="C"))
            arrays.append(dict(file=name, bytes=raw.nbytes, dtype=str(raw.dtype),
                               shape=list(raw.shape)))
            return dict(rows=raw.shape[0], cols=raw.shape[1],
                        type=depth + ((channels - 1) << 3), file=name)
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        return value

    inputs = encode(normalized)
    payload = dict(schema="ptm-debug-fixture-v1", config=config, inputs=inputs,
                   metadata=dict(metadata, python_opencv=cv2.__version__,
                                 native_module=str(_native.__file__), byte_order="little",
                                 arrays=arrays))
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smoke", action="store_true")
    source.add_argument("--config", type=Path, help="Existing run_config JSON; paths relative to it")
    parser.add_argument("--smoke-objects", action="store_true", help="Include A/B/C references")
    parser.add_argument("--out", type=Path, required=True, help="New fixture directory")
    args = parser.parse_args()
    if args.smoke_objects and not args.smoke:
        parser.error("--smoke-objects requires --smoke")
    try:
        job = smoke_inputs(args.smoke_objects) if args.smoke else load_config_inputs(args.config.resolve())
        manifest = write_fixture(args.out.resolve(), *job)
        print(json.dumps(dict(fixture=str(manifest), **job[2]), ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Fixture export failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())