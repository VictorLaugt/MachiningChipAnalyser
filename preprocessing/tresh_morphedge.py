"""
Threshold binarization
Morphological edge extraction (subtraction of eroded image from binary image)
"""

import cv2 as cv

import dag_process


processing = dag_process.DagProcessVizualiser()

processing.add("gray", lambda img: cv.cvtColor(img, cv.COLOR_RGB2GRAY))
processing.add("binary", lambda img: cv.threshold(img, 45, 255, cv.THRESH_BINARY)[1])
# processing.add("binary", lambda img: cv.threshold(img, 50, 255, cv.THRESH_BINARY)[1])
processing.add("edge", lambda img: cv.morphologyEx(img, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))))


if __name__ == '__main__':
    from pathlib import Path
    import image_loader

    input_dir = Path("imgs", "vertical")
    output_dir = Path("results", "tresh_morphedge")
    loader = image_loader.ImageLoader(input_dir)

    processing.run(loader)
    processing.show_frames(40)
    processing.show_videos()
    processing.save_videos(output_dir)
