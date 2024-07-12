from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geometry import PointArray
    from chip_extraction import MainFeatures

import geometry
from shape_detection.chip_extraction import extract_main_features
import concave_hull  # pip install concave_hull

import numpy as np
import cv2 as cv


def compute_chip_concave_hull(chip_pts: PointArray) -> PointArray:
    """Return the points of a concave hull of the chip."""
    chup_hull_idx = concave_hull.concave_hull_indexes(
        chip_pts.reshape(-1, 2),
        concavity=1.2,
        length_threshold=0.0
    )
    return chip_pts[chup_hull_idx].reshape(-1, 1, 2)


def extract_chip_features(binary_img: np.ndarray) -> tuple[MainFeatures, PointArray]:
    """Detect and return the chip features from the preprocessed binary image."""
    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    chip_hull_pts = compute_chip_concave_hull(chip_pts)

    return main_ft, chip_hull_pts


def render_chip_features(render: np.ndarray, main_ft: MainFeatures, chip_hull_pts: PointArray) -> None:
    """Draw a representation of features `main_ft` and `chip_ft` on image `render`."""
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    # dark_green = (0, 85, 0)
    # blue = (255, 0, 0)

    geometry.draw_line(render, main_ft.base_line, color=red, thickness=3)
    geometry.draw_line(render, main_ft.tool_line, color=red, thickness=3)

    for a, b in zip(chip_hull_pts[:-1], chip_hull_pts[1:]):
        cv.line(render, tuple(a[0]), tuple(b[0]), color=yellow, thickness=1)
        # cv.circle(render, tuple(a[0]), 6, color=green, thickness=-1)


class ChipFeatureCollector:
    def __init__(self):
        pass

    def extract_and_render(self, binary_img: np.ndarray, background: np.ndarray|None=None) -> np.ndarray:
        main_ft, chip_hull_pts = extract_chip_features(binary_img)
        if background is None:
            ft_repr = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
            ft_repr[binary_img > 0] = (255, 255, 255)
            render_chip_features(ft_repr, main_ft, chip_hull_pts)
        else:
            ft_repr = background.copy()
            render_chip_features(ft_repr, main_ft, chip_hull_pts)
        return ft_repr


if __name__ == '__main__':
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode

    collector = ChipFeatureCollector()

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    # processing.add("chipcurve", collector.extract_and_render, ("morph", "input"))
    processing.add("chipcurve", collector.extract_and_render)

    input_dir_str = os.environ.get("INPUT_DIR")
    if input_dir_str is not None:
        input_dir = Path(os.environ["INPUT_DIR"])
    else:
        input_dir = Path("imgs", "vertical")

    output_dir = Path("results", "chipcurve")
    loader = image_loader.ImageLoader(input_dir)

    processing.run(loader)
    processing.show_frame_comp(min(15, len(loader)-1), ("chipcurve", "input"))
    processing.show_video_comp(("chipcurve", "input"))
