"""Small cross-language correctness check, including ABI and input boundaries."""
import numpy as np
from native_matcher import match_cpp, match_python


def main():
    rng = np.random.default_rng(7)
    cases = [
        (np.zeros((4, 5), np.uint8), np.zeros((2, 2), np.uint8)),
        (np.full((3, 4), 255, np.uint8), np.zeros((3, 4), np.uint8)),
        (rng.integers(0, 256, (8, 10), dtype=np.uint8)[:, ::2],
         rng.integers(0, 256, (2, 3), dtype=np.uint8)),
    ]
    for _ in range(20):
        image = rng.integers(0, 256, (7, 9), dtype=np.uint8)
        template = rng.integers(0, 256, (3, 4), dtype=np.uint8)
        cases.append((image, template))
    for image, template in cases:
        actual, expected = match_cpp(image, template), match_python(image, template)
        if actual != expected:
            raise AssertionError((actual, expected))
    invalid = [
        (np.zeros((0, 2), np.uint8), np.zeros((1, 1), np.uint8)),
        (np.zeros((2, 2), np.float32), np.zeros((1, 1), np.uint8)),
        (np.zeros((2, 2), np.uint8), np.zeros((3, 1), np.uint8)),
        (np.zeros((2, 2, 3), np.uint8), np.zeros((1, 1), np.uint8)),
    ]
    for image, template in invalid:
        try:
            match_cpp(image, template)
        except ValueError:
            continue
        raise AssertionError("Invalid input was accepted")
    print(f"PASS: {len(cases)} matching comparisons and {len(invalid)} invalid inputs")


if __name__ == "__main__":
    main()
