"""
Normalization
Threshold binarization
Morphological edge extraction (subtraction of eroded image from binary image)
"""

from pathlib import Path

import cv2 as cv

import image_loader
import utils


dir_path = Path("imgs", "vertical")
# dir_path = Path("imgs", "diagonal")
loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)

p = utils.Pipeline()

p.add("norm", lambda img: cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
p.add("binary", lambda img: cv.threshold(img, 15, 255, cv.THRESH_BINARY)[1])
p.add("edge", lambda img: cv.morphologyEx(img, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))))


p.run(loader)
p.show_samples(20)
p.show_videos()
