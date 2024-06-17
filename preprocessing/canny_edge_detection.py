"""
Canny edge detection
"""

import cv2 as cv

import dag_process


processing = dag_process.DagProcess()
processing.add("edge", lambda img: cv.Canny(img, 100, 200))


if __name__ == '__main__':
    from pathlib import Path
    import image_loader

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "canny_edge_detection")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    processing.show_frame(43)
    processing.show_video()
