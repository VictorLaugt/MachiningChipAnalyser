"""
Threshold binarization -> Morphological edge extraction
"""

from pathlib import Path

import numpy as np
import cv2 as cv

import image_loader
import video


dir_path = Path("imgs", "vertical")
# dir_path = Path("imgs", "diagonal")
loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)


binary = [cv.threshold(img, 45, 255, cv.THRESH_BINARY)[1] for img in loader]

structure = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
# edge = [cv.morphologyEx(img, cv.MORPH_GRADIENT, structure) for img in binary]
eroded = [cv.erode(img, structure, iterations=1) for img in binary]
edge = [cv.absdiff(bin_img, eroded_img) for bin_img, eroded_img in zip(binary, eroded)]


# image display
sample_idx = 43
cv.imshow("original", loader[sample_idx])
cv.imshow("binary", binary[sample_idx])
cv.imshow("edge", edge[sample_idx])
while cv.waitKey(30) != 113:
    pass
cv.destroyAllWindows()


# video display
video.create_from_gray(loader, "original.avi")
video.create_from_gray(binary, "meth2_binary.avi")
video.create_from_gray(edge, "meth2_edge.avi")

video.play("original.avi")
video.play("meth2_binary.avi")
video.play("meth2_edge.avi")
