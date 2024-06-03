"""
Normalization
LoG edge extraction (with different matrix coefficients for contact and spikes detection)
Threshold binarization
Morphological opening and closing (for contact detection only)
"""

from pathlib import Path

import numpy as np
import cv2 as cv

import image_loader
import utils


dir_path = Path("imgs", "vertical")
# dir_path = Path("imgs", "diagonal")
loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)



log_kernel = np.array([
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


# contact detection
p1 = utils.Pipeline()
p1.add("norm", lambda img: cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
p1.add("contact_edge", lambda img: cv.filter2D(img, -1, 2 * log_kernel))
p1.add("contact_binary", lambda img: cv.threshold(img, 245, 255, cv.THRESH_BINARY)[1])
p1.add("contact_open", lambda img: cv.morphologyEx(img, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))))
p1.add("contact_close", lambda img: cv.morphologyEx(img, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=5))
p1.run(loader)


# spike detection
p2 = utils.Pipeline()
p2.add("norm", lambda img: cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
p2.add("spikes_edge", lambda img: cv.filter2D(img, -1, (1/3) * log_kernel))
p2.add("spikes_binary", lambda img: cv.threshold(img, 245, 255, cv.THRESH_BINARY)[1])
p2.run(loader)


p1.show_samples(20)
p1.show_videos()

p2.show_samples(20)
p2.show_videos()
