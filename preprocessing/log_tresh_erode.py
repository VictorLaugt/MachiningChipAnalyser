"""
LoG edge extraction
Threshold binarization
Morphological erosion
"""

import numpy as np
import cv2 as cv

import utils
import connected_components


pipeline = utils.Pipeline()

log_kernel = 2 * np.array([
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

pipeline.add("edge", lambda img: cv.filter2D(img, -1, log_kernel))
pipeline.add("binary", lambda img: cv.threshold(img, 240, 255, cv.THRESH_BINARY)[1])
pipeline.add("clean", lambda img: connected_components.remove_small_components(img, min_area=20))
# pipeline.add("clean", lambda img: connected_components.remove_small_components(img, min_area=45))
pipeline.add("erode", lambda img: cv.erode(img, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=2))

# pipeline.add("morph", lambda img: cv.erode(img, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))))


if __name__ == '__main__':
    from pathlib import Path
    import image_loader

    dir_path = Path("imgs", "vertical")
    # dir_path = Path("imgs", "diagonal")
    loader = image_loader.ImageLoaderColorConverter(dir_path, cv.COLOR_RGB2GRAY)

    pipeline.run(loader)
    pipeline.show_samples(20)
    pipeline.show_videos()
