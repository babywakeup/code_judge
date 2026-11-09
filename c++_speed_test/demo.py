"""Python orchestrates input, correctness checks, and timing; C++ computes SSD."""
import argparse
from statistics import median
from time import perf_counter
import numpy as np
from native_matcher import match_cpp, match_python


def timed(function, image, template, repeats):
    times = []
    for _ in range(repeats):
        start = perf_counter()
        result = function(image, template)
        times.append(perf_counter() - start)
    return result, median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--template-size", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not (0 < args.template_size <= args.size <= 2048) or args.repeats < 1:
        parser.error("Require 0 < template-size <= size <= 2048 and repeats >= 1")
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (args.size, args.size), dtype=np.uint8)
    template = rng.integers(0, 256, (args.template_size, args.template_size), dtype=np.uint8)
    x, y = (args.size - args.template_size) // 2, (args.size - args.template_size) // 3
    image[y:y + args.template_size, x:x + args.template_size] = template
    match_cpp(image, template)  # exclude initial library loading from steady-state timing
    py_result, py_time = timed(match_python, image, template, args.repeats)
    cpp_result, cpp_time = timed(match_cpp, image, template, args.repeats)
    if py_result != cpp_result or cpp_result != (x, y, 0):
        raise AssertionError((py_result, cpp_result, (x, y, 0)))
    print(f"Correctness: PASS; (x, y, SSD) = {cpp_result}")
    print(f"Image: {image.shape}, template: {template.shape}, repeats: {args.repeats}")
    print(f"Python median: {py_time * 1000:.3f} ms")
    print(f"C++ median (including Python wrapper): {cpp_time * 1000:.3f} ms")
    print(f"Speedup vs Python pixel loops: {py_time / cpp_time:.2f}x")


if __name__ == "__main__":
    main()
