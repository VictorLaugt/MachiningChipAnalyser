"""
Canny edge detection
"""

import cv2 as cv

import dag_process


processing = dag_process.DagProcess()
processing.add("gray", lambda img: cv.cvtColor(img, cv.COLOR_RGB2GRAY))
processing.add("edge", lambda img: cv.Canny(img, 100, 200))


if __name__ == '__main__':
    from pathlib import Path
    import image_loader

    input_dir = Path("imgs", "vertical")
    output_dir = Path("results", "canny_edge_detection")
    loader = image_loader.ImageLoader(input_dir)

    processing.run(loader)
    processing.show_frames(43)
    processing.show_videos()
    processing.save_videos(output_dir)
