"""Shared deterministic fixtures; test modules do not import one another."""
import cv2
import numpy as np


def texture(seed=2):
    rng = np.random.default_rng(seed)
    image = np.full((160, 200, 3), 65, np.uint8)
    for _ in range(65):
        center = tuple(int(v) for v in rng.integers([8, 8], [192, 152]))
        color = tuple(int(v) for v in rng.integers(70, 240, 3))
        cv2.circle(image, center, int(rng.integers(2, 9)), color, -1)
    return cv2.GaussianBlur(image, (3, 3), .5)