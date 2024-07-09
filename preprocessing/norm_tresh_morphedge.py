"""
Normalization
Threshold binarization
Morphological edge extraction (subtraction of eroded image from binary image)
"""

import cv2 as cv

import dag_process_vizualiser


processing = dag_process_vizualiser.DagProcessVizualiser()

processing.add("gray", lambda img: cv.cvtColor(img, cv.COLOR_RGB2GRAY))
processing.add("norm", lambda img: cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
processing.add("binary", lambda img: cv.threshold(img, 15, 255, cv.THRESH_BINARY)[1])
processing.add("edge", lambda img: cv.morphologyEx(img, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))))


if __name__ == '__main__':
    import os
    from pathlib import Path
    import image_loader

    input_dir_str = os.environ.get("INPUT_DIR")
    if input_dir_str is not None:
        input_dir = Path(os.environ["INPUT_DIR"])
    else:
        input_dir = Path("imgs", "vertical")

    output_dir = Path("results", "norm_tresh_morphedge")
    loader = image_loader.ImageLoader(input_dir)

    processing.run(loader)
    processing.show_frames(min(20, len(loader)-1))
    processing.show_videos()
    processing.save_videos(output_dir)
