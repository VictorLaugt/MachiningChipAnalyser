"""
LoG edge extraction
Threshold binarization
Morphological erosion
"""

import numpy as np
import cv2 as cv

import dag_process
import connected_components


processing = dag_process.DagProcess()

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

processing.add("gray", lambda img: cv.cvtColor(img, cv.COLOR_RGB2GRAY))
processing.add("edge", lambda img: cv.filter2D(img, -1, log_kernel))
processing.add("binary", lambda img: cv.threshold(img, 240, 255, cv.THRESH_BINARY)[1])
processing.add("blobfilter", lambda img: connected_components.remove_small_components(img, min_area=20))
# processing.add("blobfilter", lambda img: connected_components.remove_small_components(img, min_area=45))
processing.add("morph", lambda img: cv.erode(img, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=2))
# processing.add("morph", lambda img: cv.morphologyEx(img, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=2))

# processing.add("morph", lambda img: cv.erode(img, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))))


if __name__ == '__main__':
    from pathlib import Path
    import image_loader

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "log_thresh_blobfilter_erode")
    loader = image_loader.ImageLoader(input_dir)

    processing.run(loader, output_dir)
    # processing.compare_frames(14, ("input", "morph"))
    processing.show_frame(14)
    # processing.compare_videos(("input", "morph"))
    processing.show_video()
