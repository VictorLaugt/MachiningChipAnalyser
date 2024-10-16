from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import GrayImage

import numpy as np
import cv2 as cv


# Convolution kernel used for the Laplacian of Gaussian operation
log_kernel = np.array([
    [  0,   2,   2,   4,   4,   4,   2,   2,   0],
    [  2,   4,   8,  10,  10,  10,   8,   4,   2],
    [  2,   8,  10,   6,   0,   6,  10,   8,   2],
    [  4,  10,   6, -24, -48, -24,   6,  10,   4],
    [  4,  10,   0, -48, -80, -48,   0,  10,   4],
    [  4,  10,   6, -24, -48, -24,   6,  10,   4],
    [  2,   8,  10,   6,   0,   6,  10,   8,   2],
    [  2,   4,   8,  10,  10,  10,   8,   4,   2],
    [  0,   2,   2,   4,   4,   4,   2,   2,   0]
])

# Kernel used for morphological erosion
structuring_element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))


def remove_small_components(bin_src_img: GrayImage, min_area: int, bin_dst_img: GrayImage) -> None:
    """Filter the connected components of a binary image according to their area.

    Parameters
    ----------
    bin_src_img: (h, w)-array of uint8
        Binary image in which to analyze connected components.
    min_area: int
        Minimum area a connected component must have to be written in bin_dst_img.
    bin_dst_img: (h, w)-array of uint8
        All connected component from bin_src_img whose area >= min_area are written in bin_dst_img.
    """
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(bin_src_img, connectivity=8)

    keep_mask = np.zeros_like(labels, dtype=bool)
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            keep_mask |= (labels == i)

    bin_dst_img[keep_mask] = 255


def preprocess(input_img: GrayImage) -> GrayImage:
    """Preprocess an input machining image to produce a binary image which can
    then be used to extract features of the chip.

    Parameters
    ----------
    input_img: (h, w)-array or (h, w, 3)-array of uint8
        Input machining image.

    Returns
    -------
    binary_img: (h, w)-array of uint8
        Preprocessed machining image.
    """
    x = np.empty_like(input_img)

    # Laplacian of Gaussian and treshold binarization (in-place)
    cv.filter2D(input_img, -1, log_kernel, dst=x)
    cv.threshold(x, 240, 255, cv.THRESH_BINARY, dst=x)

    # blob filtering (out-of-place)
    y = np.zeros_like(x)
    remove_small_components(x, 20, bin_dst_img=y)

    # morphological erosion (in-place)
    cv.erode(y, structuring_element, iterations=2, dst=y)

    return y
