from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from geometry import Line, Point, PointArray
    from shape_detection.chip_extraction import MainFeatures

import geometry
from shape_detection.constrained_hull_polynomial import (
    compute_chip_convex_hull,
    extract_chip_curve_points
)
from shape_detection.chip_extraction import extract_main_features

from dataclasses import dataclass

import numpy as np
import cv2 as cv
import skimage as ski
from filterpy.kalman import KalmanFilter


@dataclass
class InsideFeatures:
    thickness: Sequence[float]
    inside_contour_pts: Sequence[Point]
    chip_curve_pts: PointArray


def rasterized_line(p0: Point, p1: Point, img_height: int, img_width: int) -> tuple[int, np.ndarray[int], np.ndarray[int]]:
    line_x, line_y = ski.draw.line(*p0, *p1)
    inside_mask = (0 <= line_x) & (line_x < img_width) & (0 <= line_y) & (line_y < img_height)
    raster_x, raster_y = line_x[inside_mask], line_y[inside_mask]
    return raster_x, raster_y


def compute_bisectors(
            chip_curve_pts: PointArray,
            indirect_chip_rotation: bool
        ) -> np.ndarray[float]:
    """Return the unit vectors bisecting the edges of the chip curve."""
    pts = chip_curve_pts.reshape(-1, 2)
    bisectors = np.zeros_like(pts, dtype=np.float32)

    u = pts[:-2] - pts[1:-1]
    v = pts[2:] - pts[1:-1]
    w = v * ((np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1))).reshape(-1, 1)

    # numerical instability correction if the angle between u and v is greater than pi/2
    stable = (np.sum(u*v, axis=1) > 0)
    unstable = ~stable

    bisectors[1:-1][stable] = u[stable] + w[stable]
    normal = u[unstable] - w[unstable]

    normal_first = pts[0] - pts[1]
    normal_last = pts[-2] - pts[-1]
    if indirect_chip_rotation:
        bisectors[1:-1][unstable] = np.column_stack((-normal[:, 1], normal[:, 0]))
        bisectors[0] = (-normal_first[1], normal_first[0])
        bisectors[-1] = (-normal_last[1], normal_last[0])
    else:
        bisectors[1:-1][unstable] = np.column_stack((normal[:, 1], -normal[:, 0]))
        bisectors[0] = (normal_first[1], -normal_first[0])
        bisectors[-1] = (normal_last[1], -normal_last[0])

    return bisectors / np.linalg.norm(bisectors, axis=1).reshape(-1, 1)


def find_inside_contour(
    binary_img: np.ndarray,
    chip_curve_pts: PointArray,
    indirect_rotation: bool,
    thickness_majorant: int
        ) -> InsideFeatures:
    h, w = binary_img.shape
    thickness = []
    inside_contour_pts = []

    bisectors = compute_bisectors(chip_curve_pts, indirect_rotation)
    for i in range(len(chip_curve_pts)-1):
        a, b = chip_curve_pts[i, 0, :], chip_curve_pts[i+1, 0, :]
        ua, ub = bisectors[i], bisectors[i+1]
        c, d = (a + thickness_majorant*ua).astype(np.int32), (b + thickness_majorant*ub).astype(np.int32)

        out_raster_x, out_raster_y = rasterized_line(a, b, h, w)
        in_raster_x, in_raster_y = rasterized_line(c, d, h, w)
        out_length, in_length = len(out_raster_x), len(in_raster_x)
        for i in range(out_length):
            j = i * in_length // out_length
            p0, p1 = (out_raster_x[i], out_raster_y[i]), (in_raster_x[j], in_raster_y[j])
            ray_x, ray_y = rasterized_line(p0, p1, h, w)

            selected_idx = np.nonzero(binary_img[ray_y, ray_x])[0]
            if len(selected_idx) > 0:
                selected_x, selected_y = ray_x[selected_idx], ray_y[selected_idx]

                distances = np.linalg.norm((selected_x - p0[0], selected_y - p0[1]), axis=0)
                furthest_idx = np.argmax(distances)

                thickness.append(distances[furthest_idx])
                inside_contour_pts.append((selected_x[furthest_idx], selected_y[furthest_idx]))

    return InsideFeatures(thickness, inside_contour_pts, chip_curve_pts)


def extract_chip_inside_contour(binary_img: np.ndarray) -> tuple[MainFeatures, InsideFeatures]:
    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    chip_hull_pts = compute_chip_convex_hull(main_ft, chip_pts)
    chip_curve_pts = extract_chip_curve_points(main_ft, chip_hull_pts)

    chip_binary_img = np.zeros_like(binary_img)
    chip_binary_img[chip_pts[:, 0, 1], chip_pts[:, 0, 0]] = 255

    inside_ft = find_inside_contour(
        chip_binary_img,
        chip_curve_pts,
        main_ft.indirect_rotation,
        thickness_majorant=125
    )

    return main_ft, inside_ft


def render_inside_features(render: np.ndarray, main_ft: MainFeatures, inside_ft: InsideFeatures) -> None:
    """Draw a representation of features `main_ft` and `inside_ft` on image `render`."""
    for x, y in inside_ft.inside_contour_pts:
        render[y, x] = (0, 0, 255)  # red
    for pt in inside_ft.chip_curve_pts.reshape(-1, 2):
        cv.circle(render, pt, 3, (0, 255, 0), -1)



if __name__ == '__main__':
    import os
    from pathlib import Path

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode
    import inside_feature_collector

    # ---- environment variables
    input_dir_str = os.environ.get("INPUT_DIR")
    output_dir_str = os.environ.get("OUTPUT_DIR")
    scale_str = os.environ.get("SCALE_UM")

    if input_dir_str is not None:
        input_dir = Path(input_dir_str)
    else:
        input_dir = Path("imgs", "vertical")

    if output_dir_str is not None:
        output_dir = Path(output_dir_str)
    else:
        output_dir = Path("results", "chipvurve")

    if scale_str is not None:
        scale_um = float(scale_str)
    else:
        scale_um = 3.5


    # ---- processing
    collector = inside_feature_collector.MedianFilterCollector(scale_um)
    # collector = inside_feature_collector.KalmanFilterCollector(scale_um)

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()

    processing.add("chipinside", collector.extract_and_render)

    loader = image_loader.ImageLoader(input_dir)
    processing.run(loader)


    # ---- visualization
    processing.show_frame_comp(min(15, len(loader)-1), ("chipinside", "morph"))
    processing.save_frame_comp(output_dir, min(15, len(loader)-1), ("chipinside",))
    processing.show_video_comp(("chipinside", "morph"))
    collector.show_thickness_animated_graph()
