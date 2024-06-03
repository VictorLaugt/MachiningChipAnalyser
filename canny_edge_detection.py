"""
Canny edge detection
"""

from pathlib import Path

import numpy as np
import cv2 as cv

import image_loader
import utils


dir_path = Path("imgs", "vertical")
# dir_path = Path("imgs", "diagonal")
loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)

p = utils.Pipeline()
p.add("edge", lambda img: cv.Canny(img, 100, 200))


p.run(loader)
p.show_samples(43)
p.show_videos()
