"""
Canny edge detection
"""

from pathlib import Path

import numpy as np
import cv2 as cv

import image_loader
import video


dir_path = Path("imgs", "vertical")
# dir_path = Path("imgs", "diagonal")
loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)

edge = [cv.Canny(img, 100, 200) for img in loader]


# image display
sample_idx = 43
cv.imshow("original", loader[sample_idx])
cv.imshow("edge", edge[sample_idx])
while cv.waitKey(30) != 113:
    pass
cv.destroyAllWindows()


# video display
video.create_from_gray(loader, "original.avi")
video.create_from_gray(edge, "edge.avi")

video.play("original.avi")
video.play("edge.avi")
