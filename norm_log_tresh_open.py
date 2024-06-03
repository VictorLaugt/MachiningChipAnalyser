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
import video
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

norm = [cv.normalize(img, None, 0, 255, cv.NORM_MINMAX) for img in loader]

# contact detection
log_kernel_contact = 2 * log_kernel
contact_edge = [cv.filter2D(img, -1, log_kernel_contact) for img in norm]

contact_binary = [cv.threshold(img, 245, 255, cv.THRESH_BINARY)[1] for img in contact_edge]

contact_open = [
    cv.morphologyEx(
        img,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)),
    )
    for img in contact_binary
]

contact_close = [
    cv.morphologyEx(
        img,
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)),
        iterations=5
    )
    for img in contact_open
]


# spike detection
log_kernel_spikes = (1/3) * log_kernel
spikes_edge = [cv.filter2D(img, -1, log_kernel_spikes) for img in norm]

spikes_binary = [cv.threshold(img, 245, 255, cv.THRESH_BINARY)[1] for img in spikes_edge]


# image display
sample_idx = 20
cv.imshow("original", loader[sample_idx])

cv.imshow("contact norm edge", contact_edge[sample_idx])
cv.imshow("contact binary", contact_binary[sample_idx])
cv.imshow("contact open", contact_open[sample_idx])
cv.imshow("contact close", contact_close[sample_idx])

cv.imshow("spikes norm edge", spikes_edge[sample_idx])
cv.imshow("spikes binary", spikes_binary[sample_idx])

while cv.waitKey(30) != 113:
    pass
cv.destroyAllWindows()


# video display
video.create_from_gray(loader, "original.avi")

video.create_from_gray(contact_edge, "contact_edge.avi")
video.create_from_gray(contact_binary, "contact_binary.avi")
video.create_from_gray(contact_open, "contact_open.avi")
video.create_from_gray(contact_close, "contact_close.avi")

video.create_from_gray(spikes_edge, "spikes_edge.avi")
video.create_from_gray(spikes_binary, "spikes_binary.avi")


video.play("original.avi")
video.play("contact_edge.avi")
video.play("contact_binary.avi")
video.play("contact_open.avi")
video.play("contact_close.avi")

video.play("spikes_edge.avi")
video.play("spikes_binary.avi")
