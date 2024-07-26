from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import GrayImage, Image

import numpy as np
import cv2 as cv


# Convolution kernel used for the Laplacian of Gaussian operation
log_kernel = 2 * np.array([
    [0, 1, 1, 2, 2, 2, 1, 1, 0],
    [1, 2, 4, 5, 5, 5, 4, 2, 1],
    [1, 4, 5, 3, 0, 3, 5, 4, 1],
    [2, 5, 3, -12, -24, -12, 3, 5, 2],
    [2, 5, 0, -24, -40, -24, 0, 5, 2],
    [2, 5, 3, -12, -24, -12, 3, 5, 2],
    [1, 4, 5, 3, 0, 3, 5, 4, 1],
    [1, 2, 4, 5, 5, 5, 4, 2, 1],
    [0, 1, 1, 2, 2, 2, 1, 1, 0]
])

# Kernel used for morphological erosion
structuring_element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))


def remove_small_components(bin_src_img: GrayImage, min_area: int, bin_dst_img: GrayImage) -> GrayImage:
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(bin_src_img, connectivity=8)

    keep_mask = np.zeros_like(labels, dtype=bool)
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            keep_mask |= (labels == i)

    bin_dst_img[keep_mask] = 255
    return bin_dst_img


def preprocess(input_img: Image) -> GrayImage:
    # conversion in grayscale (out-of-place)
    x = cv.cvtColor(input_img, cv.COLOR_RGB2GRAY)

    # Laplacian of Gaussian and treshold binarization (in-place)
    cv.filter2D(x, -1, log_kernel, dst=x)
    cv.threshold(x, 240, 255, cv.THRESH_BINARY, dst=x)

    # blob filtering (out-of-place)
    y = np.zeros_like(x)
    remove_small_components(x, 20, bin_dst_img=y)

    # morphological erosion (in-place)
    cv.erode(y, structuring_element, iterations=2, dst=y)
    return y


#OPTIMIZE: preprocessing should reuse old arrays instead of allocate new ones
def preprocess_outofplace(input_img: Image) -> GrayImage:
    gray_img = cv.cvtColor(input_img, cv.COLOR_RGB2GRAY)
    edge = cv.filter2D(gray_img, -1, log_kernel)
    binary = cv.threshold(edge, 240, 255, cv.THRESH_BINARY)[1]
    blobfilter = remove_small_components(binary, 20)
    eroded = cv.erode(blobfilter, structuring_element, iterations=2)
    return eroded
