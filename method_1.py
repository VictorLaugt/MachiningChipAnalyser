"""
LoG edge extraction -> Threshold binarization (-> Morphological opening)
"""

from pathlib import Path

import numpy as np
import cv2 as cv

import image_loader
import video


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

log_kernel_contact = (2) * log_kernel
log_kernel_spikes = (1/3) * log_kernel

edge_contact = [cv.filter2D(img, -1, log_kernel_contact) for img in loader]
edge_spikes = [cv.filter2D(img, -1, log_kernel_spikes) for img in loader]

norm_edge_contact = [cv.normalize(img, None, 0, 255, cv.NORM_MINMAX) for img in edge_contact]
norm_edge_spikes = [cv.normalize(img, None, 0, 255, cv.NORM_MINMAX) for img in edge_spikes]

binary_contact = [cv.threshold(img, 240, 255, cv.THRESH_BINARY)[1] for img in norm_edge_contact]
binary_spikes = [cv.threshold(img, 240, 255, cv.THRESH_BINARY)[1] for img in norm_edge_spikes]

clean_contact = [
    cv.morphologyEx(
        img,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    )
    for img in binary_contact
]


# image display
sample_idx = 20
cv.imshow("original", loader[sample_idx])
cv.imshow("LoG contact", norm_edge_contact[sample_idx])
cv.imshow("LoG spikes", norm_edge_spikes[sample_idx])
cv.imshow("contact", clean_contact[sample_idx])
cv.imshow("spikes", binary_spikes[sample_idx])
while cv.waitKey(30) != 113:
    pass
cv.destroyAllWindows()


# video display
video.create_from_gray(loader, "original.avi")
video.create_from_gray(norm_edge_contact, "meth1_norm_edge_contact.avi")
video.create_from_gray(norm_edge_spikes, "meth1_norm_edge_spikes.avi")
video.create_from_gray(clean_contact, "meth1_clean_contact.avi")
video.create_from_gray(binary_spikes, "meth1_binary_spikes.avi")

video.play("original.avi")
video.play("meth1_norm_edge_contact.avi")
video.play("meth1_norm_edge_spikes.avi")
video.play("meth1_clean_contact.avi")
video.play("meth1_binary_spikes.avi")
