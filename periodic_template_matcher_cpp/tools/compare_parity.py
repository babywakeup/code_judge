"""Measure SD-1 array-API parity without building, saving predictions, or tuning tolerances.

Run directly with the selected Python interpreter. The default report is ignored
by the existing package .gitignore. Exit 0 means measurement completed, NOT pass;
exit 1 means an execution error or input mutation. No accuracy/GT claim is made.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from periodic_template_matcher import PeriodicTemplateMatcher as PythonMatcher
from periodic_template_matcher._internal.models import MatchResult
from periodic_template_matcher.utils.regions import ValidRegion
from periodic_template_matcher_cpp import PeriodicTemplateMatcher as CppMatcher, _native
from periodic_template_matcher_cpp.tools.export_fixture import load_config_inputs
from tests.test_rotation_equivariance import rotation, turn

ARRAY_FIELDS = tuple(f.name for f in fields(MatchResult) if f.name != "registration")
assert len(ARRAY_FIELDS) == 14, "Update the report schema if MatchResult changes"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    """Preserve non-finite observations explicitly; emit strict JSON, never NaN."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def fingerprint(cfg, kwargs, metadata):
    """Hash every array's bytes/dtype/shape and all non-array input values."""
    arrays = {}

    def encode(value, path):
        if isinstance(value, np.ndarray):
            digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            record = dict(shape=list(value.shape), dtype=value.dtype.str, sha256=digest)
            arrays[path] = record
            return record
        if isinstance(value, dict):
            return {str(k): encode(v, f"{path}/{k}") for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v, f"{path}/{i}") for i, v in enumerate(value)]
        return json_safe(value)

    payload = encode(dict(config=asdict(cfg), kwargs=kwargs, metadata=metadata), "input")
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), array_count=len(arrays), arrays=arrays)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(config_path):
    spec = json.loads(config_path.read_text(encoding="utf-8-sig"))
    paths = {config_path}
    keys = ("image", "labelme", "labels", "valid_mask", "annotation_valid",
            "target", "target_valid_mask")
    for item in [spec, *spec["templates"]]:
        for key in keys:
            if item.get(key):
                paths.add((config_path.parent / item[key]).resolve())
    return {str(p): file_hash(p) for p in sorted(paths)}


def rotate_inputs(cfg, kwargs, metadata, k):
    """Use the actual quarter-turn fixture, with explicit rotated original-pixel ROI.

    Baseline kwargs/config are never normalized in place. Height constraints must
    become masks before rotation, otherwise they incorrectly clip the new top.
    Identity names stay attached to references; only cycle numbering reverses.
    """
    images = kwargs["templates"]
    heights = kwargs.get("template_valid_heights") or [None] * len(images)
    valids = kwargs.get("template_valids") or [None] * len(images)
    rois = [ValidRegion.from_config(cfg, h).build_mask(im, mask)
            for im, h, mask in zip(images, heights, valids)]
    target_roi = ValidRegion.from_config(cfg, kwargs.get("target_valid_height")).build_mask(
        kwargs["target"], kwargs.get("target_valid"))

    def rotate_arrays(value):
        if isinstance(value, np.ndarray):
            return turn(value, k)
        if isinstance(value, dict):
            return {key: rotate_arrays(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [rotate_arrays(item) for item in value]
        return deepcopy(value)

    turned = rotate_arrays(kwargs)
    turned.update(template_valids=[turn(v, k) for v in rois],
                  target_valid=turn(target_roi, k), template_valid_heights=None,
                  target_valid_height=None)
    linear = rotation(images[0].shape, k)[:2, :2]
    old_axis = 0 if cfg.object_order_axis == "x" else 1
    direction = linear[:, old_axis]
    new_axis = "y" if cfg.object_order_axis == "x" else "x"
    reverse = bool(direction[1 - old_axis] < 0)
    cycle = list(reversed(cfg.object_cycle)) if reverse else list(cfg.object_cycle)
    turned_cfg = replace(deepcopy(cfg), valid_height=None, object_order_axis=new_axis,
                         object_cycle=cycle,
                         class_periodicity={c: ["y" if a == "x" else "x" for a in axes]
                                            for c, axes in cfg.class_periodicity.items()},
                         period_vectors=[(linear @ np.asarray(v)).tolist()
                                         for v in cfg.period_vectors])
    if turned.get("template_objects") is not None:
        for row in turned["template_objects"]:
            for ref in row:
                # load_objects carries a bounding-box center on the old axis.
                if "center_x" in ref or "center_y" in ref:
                    ref.pop("center_x", None)
                    ref.pop("center_y", None)
                    ys, xs = np.nonzero(ref["mask"])
                    for axis, coordinates in (("x", xs), ("y", ys)):
                        ref[f"center_{axis}"] = float((coordinates.min() + coordinates.max()) / 2)
    turned_metadata = deepcopy(metadata)
    for layout in turned_metadata.get("template_instance_layouts", []) or []:
        if isinstance(layout, dict):
            if "order_axis" in layout:
                layout["order_axis"] = new_axis
            if reverse and "order" in layout:
                layout["order"] = list(reversed(layout["order"]))
    turned_metadata.update(rotation="cw90" if k == -1 else "ccw90",
                           quarter_turn_k=k, cycle_reversed=reverse,
                           roi_policy="original height AND mask, then turn; rotated heights cleared")
    turned_cfg.validate()
    return turned_cfg, turned, turned_metadata


def array_diff(a, b):
    result = dict(cpp_present=a is not None, python_present=b is not None)
    for name, value in (("cpp", a), ("python", b)):
        if value is not None:
            result[name] = dict(shape=list(value.shape), dtype=str(value.dtype), elements=value.size)
    if a is None or b is None:
        return dict(result, status="exact_equal" if a is b else "presence_mismatch",
                    unequal_count=0 if a is b else None, max_abs_diff=None)
    if a.shape != b.shape:
        return dict(result, status="shape_mismatch", unequal_count=None, max_abs_diff=None)
    unequal = a != b
    count = int(np.count_nonzero(unequal))
    # Cast before subtraction to avoid uint8/uint16 wraparound and bool subtraction.
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    finite = np.isfinite(delta)
    max_diff = float(delta[finite].max()) if finite.any() else None
    dtype_equal = a.dtype == b.dtype
    result.update(status="exact_equal" if count == 0 and dtype_equal else "different",
                  dtype_equal=dtype_equal, unequal_count=count,
                  unequal_pixel_count=int(np.count_nonzero(np.any(unequal, axis=2)))
                                      if a.ndim == 3 else count,
                  max_abs_diff=max_diff, nonfinite_difference_count=int((~finite).sum()))
    if a.ndim == 2:
        aa, bb = a != 0, b != 0
        union = int(np.count_nonzero(aa | bb))
        intersection = int(np.count_nonzero(aa & bb))
        result.update(nonzero_iou=intersection / union if union else None,
                      nonzero_union_pixels=union, both_nonzero_empty=union == 0)
    return result


def iou_by_value(a, b, values):
    if a is None or b is None or a.shape != b.shape:
        return dict(status="not_comparable", entries=[])
    entries = []
    for value, name in values:
        aa, bb = a == value, b == value
        union = int(np.count_nonzero(aa | bb))
        intersection = int(np.count_nonzero(aa & bb))
        entries.append(dict(value=int(value), name=name, cpp_pixels=int(aa.sum()),
                            python_pixels=int(bb.sum()), intersection=intersection, union=union,
                            iou=intersection / union if union else None, both_empty=union == 0))
    return dict(status="measured", entries=entries)


def control_point_diff(a, b, shape):
    """Compare original-pixel transforms at corners, edge midpoints and center."""
    if a is None or b is None:
        return dict(status="both_absent" if a is b else "presence_mismatch",
                    max_displacement_px=None)
    h, w = shape[:2]
    points = np.array([[x, y, 1.] for y in (0., (h - 1) / 2, h - 1.)
                       for x in (0., (w - 1) / 2, w - 1.)])
    aa, bb = np.asarray(a, np.float64), np.asarray(b, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pa, pb = points @ aa.T, points @ bb.T
        pa, pb = pa[:, :2] / pa[:, 2:], pb[:, :2] / pb[:, 2:]
        distances = np.linalg.norm(pa - pb, axis=1)
    return dict(status="measured" if np.isfinite(distances).all() else "nonfinite",
                cpp_matrix=aa.tolist(), python_matrix=bb.tolist(),
                matrix_max_abs_diff=float(np.max(np.abs(aa - bb))),
                source_control_points=points[:, :2].tolist(), cpp_points=pa.tolist(),
                python_points=pb.tolist(), displacement_px=distances.tolist(),
                max_displacement_px=float(distances.max()))


def registration_summary(result):
    reg = result.registration
    return dict(status=reg.get("status"), accepted=reg.get("accepted"),
                best_template_index=reg.get("best_template_index"), ranking=reg.get("ranking"),
                scores=[m.get("match_score") for m in reg.get("matches", [])],
                ordered_objects=reg.get("ordered_objects"),
                base_registration_status=reg.get("base_registration_status"),
                prediction_scope=reg.get("prediction_scope"))


def numeric_difference(a, b):
    return None if a is None or b is None else float(a) - float(b)


def compare_results(cpp, python, cfg, kwargs):
    ca, pa = registration_summary(cpp), registration_summary(python)
    arrays = {name: array_diff(getattr(cpp, name), getattr(python, name)) for name in ARRAY_FIELDS}
    cm, pm = cpp.registration.get("matches", []), python.registration.get("matches", [])
    matches = []
    for i in range(max(len(cm), len(pm))):
        c, p = cm[i] if i < len(cm) else {}, pm[i] if i < len(pm) else {}
        matches.append(dict(template_index=i, cpp_status=c.get("status"), python_status=p.get("status"),
                            cpp_score=c.get("match_score"), python_score=p.get("match_score"),
                            score_diff=numeric_difference(c.get("match_score"), p.get("match_score")),
                            control_points=control_point_diff(c.get("matrix_template_to_target"),
                                p.get("matrix_template_to_target"), kwargs["templates"][i].shape)))
    best_c, best_p = ca["best_template_index"], pa["best_template_index"]
    selected = (control_point_diff(cpp.matrix_template_to_target, python.matrix_template_to_target,
                                  kwargs["templates"][best_c].shape)
                if best_c is not None and best_c == best_p else
                dict(status="not_comparable_different_or_absent_selected_template", max_displacement_px=None))
    objects_c = (ca["ordered_objects"] or {}).get("objects", [])
    objects_p = (pa["ordered_objects"] or {}).get("objects", [])
    objects = []
    for identity in cfg.object_cycle:
        cs = [o for o in objects_c if o.get("identity") == identity]
        ps = [o for o in objects_p if o.get("identity") == identity]
        item = dict(identity=identity, cpp=cs, python=ps)
        if len(cs) == len(ps) == 1:
            c, p = cs[0], ps[0]
            ti = c.get("template_index")
            if ti is not None and ti == p.get("template_index"):
                item["control_points"] = control_point_diff(c.get("matrix_template_to_target"),
                    p.get("matrix_template_to_target"), kwargs["templates"][ti].shape)
        objects.append(item)
    class_values = set(int(v) for v in cfg.class_periodicity)
    for result in (cpp, python):
        if result.mapped_labels is not None:
            class_values.update(int(v) for v in np.unique(result.mapped_labels))
    class_ious = {name: iou_by_value(getattr(cpp, name), getattr(python, name),
                                   [(i, str(i)) for i in sorted(class_values)])
                 for name in ("mapped_labels", "base_labels")}
    identity_ious = iou_by_value(cpp.object_identities, python.object_identities,
                                [(i + 1, name) for i, name in enumerate(cfg.object_cycle)])
    same_summary = json_safe(ca) == json_safe(pa)
    return dict(status="exact_equal" if same_summary and all(
                    a["status"] == "exact_equal" for a in arrays.values()) else "different",
                status_definition="exact equality of 14 arrays (including dtype) and registration summary; no tolerance/GT",
                cpp=ca, python=pa, status_equal=ca["status"] == pa["status"],
                accepted_equal=ca["accepted"] == pa["accepted"],
                ranking_equal=ca["ranking"] == pa["ranking"],
                best_template_equal=best_c == best_p, registration_summary_equal=same_summary,
                selected_score_diff=numeric_difference(cpp.match_score, python.match_score),
                selected_control_points=selected, template_matches=matches, objects=objects,
                arrays=arrays, class_iou=class_ious, identity_iou=identity_ious)


def measure_case(name, cfg, kwargs, metadata):
    started = time.perf_counter()
    record = dict(name=name, started_at_utc=utc_now(), config=asdict(cfg), metadata=metadata,
                  input_hash_before=fingerprint(cfg, kwargs, metadata), engines={})
    results = {}
    for engine, matcher in (("cpp", CppMatcher), ("python", PythonMatcher)):
        before = fingerprint(cfg, kwargs, metadata)
        begin = time.perf_counter()
        run = dict(started_at_utc=utc_now(), input_hash_before=before)
        try:
            # The IO loader calls it target; both public array APIs call it input.
            options = {k: v for k, v in kwargs.items() if k not in ("templates", "masks", "target")}
            results[engine] = matcher(cfg).match(kwargs["templates"], kwargs["masks"],
                                                kwargs["target"], **options)
            run["execution_status"] = "completed"
        except Exception as error:
            run.update(execution_status="error", error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
        run.update(elapsed_seconds=time.perf_counter() - begin, finished_at_utc=utc_now(),
                   input_hash_after=fingerprint(cfg, kwargs, metadata))
        run["input_unchanged"] = run["input_hash_after"] == before
        record["engines"][engine] = run
        print(f"{name} {engine}: {run['execution_status']} {run['elapsed_seconds']:.6f}s", flush=True)
    if len(results) == 2:
        try:
            record["comparison"] = compare_results(results["cpp"], results["python"], cfg, kwargs)
            record["status"] = record["comparison"]["status"]
        except Exception as error:
            record.update(status="error", comparison_error=str(error), traceback=traceback.format_exc())
    else:
        record["status"] = "error"
    record.update(input_hash_after=fingerprint(cfg, kwargs, metadata), finished_at_utc=utc_now(),
                  elapsed_seconds=time.perf_counter() - started)
    record["input_unchanged"] = record["input_hash_before"] == record["input_hash_after"]
    if not record["input_unchanged"] or not all(r["input_unchanged"] for r in record["engines"].values()):
        record["status"] = "input_mutated"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT.parent / "periodic_template_matcher/config.sd1.labelme.opencv.json")
    parser.add_argument("--out", type=Path, default=ROOT / "build/parity-report.json")
    args = parser.parse_args()
    start = time.perf_counter()
    native_path = Path(_native.__file__).resolve()
    report = dict(schema="ptm-array-parity-v1", started_at_utc=utc_now(),
                  versions=dict(python=sys.version, python_executable=sys.executable,
                                platform=platform.platform(), numpy=np.__version__,
                                python_opencv=cv2.__version__, cpp_opencv=_native.opencv_version,
                                opencv_versions_equal=cv2.__version__ == _native.opencv_version,
                                native_module=str(native_path), native_sha256_before=file_hash(native_path)),
                  policy=dict(build_performed=False, tolerance=None, ground_truth_used=False,
                              exit_zero_means="measurement completed, not parity pass"), cases=[])
    failed = False
    try:
        config_path = args.config.resolve()
        report["source_hashes_before"] = source_hashes(config_path)
        cfg, kwargs, metadata = load_config_inputs(config_path)
        report["base_input_hash_before"] = fingerprint(cfg, kwargs, metadata)
        # Only one rotated job/result pair lives at a time (SD-1 is full resolution).
        report["cases"].append(measure_case("base", cfg, kwargs, metadata))
        for k, name in ((-1, "cw90"), (1, "ccw90")):
            try:
                rotated = rotate_inputs(cfg, kwargs, metadata, k)
                report["cases"].append(measure_case(name, *rotated))
                del rotated
            except Exception as error:
                report["cases"].append(dict(name=name, status="error", error=str(error),
                                            traceback=traceback.format_exc()))
        report["base_input_hash_after"] = fingerprint(cfg, kwargs, metadata)
        report["base_input_unchanged"] = report["base_input_hash_before"] == report["base_input_hash_after"]
        report["source_hashes_after"] = source_hashes(config_path)
        report["source_files_unchanged"] = report["source_hashes_before"] == report["source_hashes_after"]
        failed = (not report["base_input_unchanged"] or not report["source_files_unchanged"]
                  or any(c["status"] in ("error", "input_mutated") for c in report["cases"]))
    except Exception as error:
        failed = True
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    report["versions"]["native_sha256_after"] = file_hash(native_path)
    report["versions"]["native_file_unchanged"] = (
        report["versions"]["native_sha256_before"] == report["versions"]["native_sha256_after"])
    report.update(status="measurement_error" if failed else "measurement_completed",
                  finished_at_utc=utc_now(), elapsed_seconds=time.perf_counter() - start,
                  case_status_counts={s: sum(c["status"] == s for c in report["cases"])
                                      for s in sorted({c["status"] for c in report["cases"]})})
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(report=str(output), status=report["status"],
                          elapsed_seconds=report["elapsed_seconds"],
                          case_status_counts=report["case_status_counts"]), ensure_ascii=False), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())