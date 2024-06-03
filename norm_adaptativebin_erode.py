"""
Gaussian blur
Normalization
Binarisation using an adaptative threshold
Moprhological erosion
"""

from pathlib import Path

import numpy as np
import cv2 as cv

import image_loader
import video


dir_path = Path("imgs", "vertical")
# dir_path = Path("imgs", "diagonal")
loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)

blur = [cv.GaussianBlur(img, (9, 9), 0) for img in loader]

norm = [cv.normalize(img, None, 0, 255, cv.NORM_MINMAX) for img in blur]

# binary = [cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2) for img in norm]
binary = [cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2) for img in norm]

morph = [cv.erode(img, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))) for img in binary]
# morph = [cv.morphologyEx(img, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)), iterations=2) for img in binary]


# image display
# sample_idx = 43  # cas pathologique
sample_idx = 20
cv.imshow("original", loader[sample_idx])
cv.imshow("blur", blur[sample_idx])
cv.imshow("norm", norm[sample_idx])
cv.imshow("binary", binary[sample_idx])
cv.imshow("morph", morph[sample_idx])
while cv.waitKey(30) != 113:
    pass
cv.destroyAllWindows()


# video display
video.create_from_gray(loader, "original.avi")
video.create_from_gray(blur, "blur.avi")
video.create_from_gray(norm, "norm.avi")
video.create_from_gray(binary, "binary.avi")
video.create_from_gray(morph, "morph.avi")

video.play("original.avi")
video.play("blur.avi")
video.play("norm.avi")
video.play("binary.avi")
video.play("morph.avi")
