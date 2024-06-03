"""
LoG edge extraction
Threshold binarization
Morphological erosion
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
    [1, 2, 4, 5, 5, 5, 4, 2, 1],
    [1, 4, 5, 3, 0, 3, 5, 4, 1],
    [0, 1, 1, 2, 2, 2, 1, 1, 0]
])


#  detection
log_kernel_ = 2 * log_kernel

edge = [cv.filter2D(img, -1, log_kernel_) for img in loader]

binary = [cv.threshold(img, 240, 255, cv.THRESH_BINARY)[1] for img in edge]

morph = [cv.erode(img, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))) for img in binary]


# image display
sample_idx = 20
cv.imshow("original", loader[sample_idx])

cv.imshow("edge", edge[sample_idx])
cv.imshow("binary", binary[sample_idx])
cv.imshow("morph", morph[sample_idx])
cv.imwrite("morph.png", morph[sample_idx])
while cv.waitKey(30) != 113:
    pass
cv.destroyAllWindows()


# video display
video.create_from_gray(loader, "original.avi")

video.create_from_gray(edge, "edge.avi")
video.create_from_gray(binary, "binary.avi")
video.create_from_gray(morph, "morph.avi")

video.play("original.avi")
video.play("edge.avi")
video.play("binary.avi")
video.play("morph.avi")
