"""
Normalization
Threshold binarization
Morphological edge extraction (subtraction of eroded image from binary image)
"""

from pathlib import Path

import numpy as np
import cv2 as cv

import image_loader
import video


dir_path = Path("imgs", "vertical")
# dir_path = Path("imgs", "diagonal")
loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)

norm = [cv.normalize(img, None, 0, 255, cv.NORM_MINMAX) for img in loader]

binary = [cv.threshold(img, 15, 255, cv.THRESH_BINARY)[1] for img in norm]

structure = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
# edge = [cv.morphologyEx(img, cv.MORPH_GRADIENT, structure) for img in binary]
eroded = [cv.erode(img, structure, iterations=1) for img in binary]
edge = [cv.absdiff(bin_img, eroded_img) for bin_img, eroded_img in zip(binary, eroded)]


# image display
# sample_idx = 43  # cas pathologique
sample_idx = 20
cv.imshow("original", loader[sample_idx])
cv.imshow("norm", norm[sample_idx])
cv.imshow("binary", binary[sample_idx])
cv.imshow("edge", edge[sample_idx])
while cv.waitKey(30) != 113:
    pass
cv.destroyAllWindows()


# video display
video.create_from_gray(loader, "original.avi")
video.create_from_gray(norm, "norm.avi")
video.create_from_gray(binary, "binary.avi")
video.create_from_gray(edge, "edge.avi")

video.play("original.avi")
video.play("norm.avi")
video.play("binary.avi")
video.play("edge.avi")
