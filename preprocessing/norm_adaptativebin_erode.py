"""
Gaussian blur
Normalization
Binarisation using an adaptative threshold
Moprhological erosion
"""

import cv2 as cv

import dag_process


processing = dag_process.DagProcessVizualiser()

processing.add("gray", lambda img: cv.cvtColor(img, cv.COLOR_RGB2GRAY))
processing.add("blur", lambda img: cv.GaussianBlur(img, (9, 9), 0))
processing.add("norm", lambda img: cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))
# processing.add("binary", lambda img: cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2))
processing.add("binary", lambda img: cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2))
processing.add("morph", lambda img: cv.erode(img, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))))
# processing.add("morph", lambda img: cv.morphologyEx(img, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)), iterations=2))


if __name__ == '__main__':
    from pathlib import Path
    import image_loader

    input_dir = Path("imgs", "vertical")
    output_dir = Path("results", "norm_adaptativebin_erode")
    loader = image_loader.ImageLoader(input_dir)

    processing.run(loader)
    processing.show_frames(20)
    processing.show_videos()
    processing.save_videos(output_dir)

