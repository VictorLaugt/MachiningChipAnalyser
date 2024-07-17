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
import connected_components

from dataclasses import dataclass

import numpy as np
import cv2 as cv
import skimage as ski


@dataclass
class InsideFeatures:
    chip_curve_pts: PointArray

    noised_inside_contour_pts: Sequence[Point]
    noised_thickness: Sequence[float]

    inside_contour_pts: Sequence[Point]
    thickness: Sequence[float]



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
    chip_bin_img: np.ndarray,
    chip_curve_pts: PointArray,
    indirect_rotation: bool,
    thickness_majorant: int
        ) -> tuple[Sequence[Point], Sequence[float]]:
    h, w = chip_bin_img.shape
    thickness = []
    inside_contour_pts = []

    bisectors = compute_bisectors(chip_curve_pts, indirect_rotation)
    out_curve = chip_curve_pts.reshape(-1, 2)
    in_curve = (out_curve + thickness_majorant*bisectors).astype(np.int32)

    for block_idx in range(len(out_curve)-1):
        out_edge_x, out_edge_y = rasterized_line(
            out_curve[block_idx], out_curve[block_idx+1], h, w
        )
        in_edge_x, in_edge_y = rasterized_line(
            in_curve[block_idx], in_curve[block_idx+1], h, w
        )
        out_edge_length, in_edge_length = len(out_edge_x), len(in_edge_x)

        for i in range(out_edge_length):
            j = i * in_edge_length // out_edge_length

            out_x, out_y = out_edge_x[i], out_edge_y[i]
            in_x, in_y = in_edge_x[j], in_edge_y[j]
            ray_x, ray_y = rasterized_line((out_x, out_y), (in_x, in_y), h, w)

            selected_idx = np.nonzero(chip_bin_img[ray_y, ray_x])[0]
            if len(selected_idx) > 0:
                selected_x, selected_y = ray_x[selected_idx], ray_y[selected_idx]
                distances = np.linalg.norm((selected_x - out_x, selected_y - out_y), axis=0)
                innermost_idx = np.argmax(distances)

                thickness.append(distances[innermost_idx])
                inside_contour_pts.append((selected_x[innermost_idx], selected_y[innermost_idx]))

    return inside_contour_pts, thickness


NEIGHBORHOOD_MASKS = (
    cv.circle(np.zeros((9, 9), dtype=np.uint8), (4, 4), 1, 1, -1),
    cv.circle(np.zeros((9, 9), dtype=np.uint8), (4, 4), 2, 1, -1),
    cv.circle(np.zeros((9, 9), dtype=np.uint8), (4, 4), 3, 1, -1),
    cv.circle(np.zeros((9, 9), dtype=np.uint8), (4, 4), 4, 1, -1),
)


def clean_inside_contour(
    chip_bin_img: np.ndarray,
    inside_contour_pts: Sequence[Point],
    thickness: Sequence[float],
    min_neighbors_count: int
        ) -> tuple[Sequence[Point], Sequence[float]]:

    print()
    h, w = chip_bin_img.shape
    inside_contour_bin_img = np.zeros((h+8, w+8), dtype=np.uint8)
    for x, y in inside_contour_pts:
        inside_contour_bin_img[y, x] = 1
    max_thickness = max(thickness)

    filtered_inside_contour_pts = []
    filtered_thickness = []

    for (x, y), t in zip(inside_contour_pts, thickness):
        # mask_index = round(3 * t / thickness_majorant)
        mask_index = round(3 * t / max_thickness)
        print(f"thickness = {t}, mask index = {mask_index}")
        mask = NEIGHBORHOOD_MASKS[mask_index]
        count = np.sum(inside_contour_bin_img[y-4:y+5, x-4:x+5] * mask)
        if count > min_neighbors_count:
            filtered_inside_contour_pts.append((x, y))
            filtered_thickness.append(t)

    return filtered_inside_contour_pts, filtered_thickness


def extract_chip_inside_contour(binary_img: np.ndarray) -> tuple[MainFeatures, InsideFeatures]:
    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    chip_hull_pts = compute_chip_convex_hull(main_ft, chip_pts)
    chip_curve_pts = extract_chip_curve_points(main_ft, chip_hull_pts)

    chip_binary_img = np.zeros_like(binary_img)
    chip_binary_img[chip_pts[:, 0, 1], chip_pts[:, 0, 0]] = 255
    # TODO: try to replace this connected component filter by a more generic process
    # chip_binary_img = connected_components.remove_small_components(chip_binary_img, min_area=20)

    noised_inside_contour_pts, noised_thickness = find_inside_contour(
        chip_binary_img,
        chip_curve_pts,
        main_ft.indirect_rotation,
        thickness_majorant=125
    )
    clean_inside_contour_pts, clean_thickness = clean_inside_contour(
        chip_binary_img,
        noised_inside_contour_pts,
        noised_thickness,
        min_neighbors_count=2
    )

    inside_ft = InsideFeatures(
        chip_curve_pts,
        noised_inside_contour_pts,
        noised_thickness,
        clean_inside_contour_pts,
        clean_thickness
    )
    return main_ft, inside_ft


def render_inside_features(render: np.ndarray, main_ft: MainFeatures, inside_ft: InsideFeatures) -> None:
    """Draw a representation of features `main_ft` and `inside_ft` on image `render`."""
    for x, y in inside_ft.noised_inside_contour_pts:
        render[y, x] = (0, 0, 255)  # red
    for x, y in inside_ft.inside_contour_pts:
        render[y, x] = (0, 255, 0)  # green
    for pt in inside_ft.chip_curve_pts.reshape(-1, 2):
        cv.circle(render, pt, 3, (255, 0, 0), -1)



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
        output_dir = Path("results", "chipcurve")

    if scale_str is not None:
        scale_um = float(scale_str)
    else:
        scale_um = 3.5


    # ---- processing
    collector = inside_feature_collector.CollectorMedian(scale_um)
    # collector = inside_feature_collector.CollectorWavelet(scale_um)
    # collector = inside_feature_collector.CollectorKalman(scale_um)

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("chipinside", collector.extract_and_render)

    loader = image_loader.ImageLoader(input_dir)
    processing.run(loader)


    # ---- visualization
    processing.show_frame_comp(min(15, len(loader)-1), ("chipinside", "morph"))
    processing.save_frame_comp(output_dir, min(15, len(loader)-1), ("chipinside",))
    processing.show_video_comp(("chipinside", "morph"))
    collector.show_thickness_animated_graph()
    # collector.show_thickness_graph(14)
    collector.save_measures(output_dir.joinpath("thickness.csv"), 14)
