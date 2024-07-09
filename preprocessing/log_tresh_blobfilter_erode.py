"""
LoG edge extraction
Threshold binarization
Morphological erosion
"""

import numpy as np
import cv2 as cv

import dag_process_vizualiser
import connected_components


processing = dag_process_vizualiser.DagProcessVizualiser()

log_kernel = 2 * np.array([
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

processing.add("gray", lambda img: cv.cvtColor(img, cv.COLOR_RGB2GRAY))
processing.add("edge", lambda img: cv.filter2D(img, -1, log_kernel))
processing.add("binary", lambda img: cv.threshold(img, 240, 255, cv.THRESH_BINARY)[1])
processing.add("blobfilter", lambda img: connected_components.remove_small_components(img, min_area=20))
# processing.add("blobfilter", lambda img: connected_components.remove_small_components(img, min_area=45))
processing.add("morph", lambda img: cv.erode(img, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=2))
# processing.add("morph", lambda img: cv.morphologyEx(img, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=2))


if __name__ == '__main__':
    import os
    from pathlib import Path
    import image_loader

    input_dir_str = os.environ.get("INPUT_DIR")
    output_dir_str = os.environ.get("OUTPUT_DIR")

    if input_dir_str is not None:
        input_dir = Path(input_dir_str)
    else:
        input_dir = Path("imgs", "vertical")

    if output_dir_str is not None:
        output_dir = Path(output_dir_str)
    else:
        output_dir = Path("results", "log_thresh_blobfilter_erode")

    loader = image_loader.ImageLoader(input_dir)

    processing.run(loader)
    processing.show_frames(min(14, len(loader)-1))
    for step_name in ("input", "edge", "binary", "blobfilter", "morph"):
        processing.save_frame_comp(output_dir, 14, (step_name,))
    # processing.save_frame_comp(
    #     output_dir,
    #     14,
    #     ("edge", "binary", "blobfilter", "morph"),
    #     horizontal=False
    # )

    # processing.show_videos()
    # processing.save_videos(output_dir)

