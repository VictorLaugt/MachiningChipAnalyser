from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from shape_detection.chip_inside_contour import ChipInsideFeatures

import geometry
from shape_detection.chip_extraction import extract_main_features
from shape_detection.constrained_hull_polynomial import (
    compute_chip_convex_hull,
    extract_chip_curve_points
)
from shape_detection.chip_inside_contour import (
    create_edge_lines,
    compute_distance_edge_points,
    find_inside_contour
)

import numpy as np
import cv2 as cv



def render_chip_inside(binary_img: np.ndarray) -> tuple[ChipInsideFeatures, np.ndarray]:
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    ft_repr = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
    # ft_repr[np.nonzero(binary_img)] = (255, 255, 255)

    main_ft = extract_main_features(binary_img)

    contours, _ = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pts = np.vstack(contours)
    chip_pts = geometry.under_lines(pts, (main_ft.base_line, main_ft.tool_line), (10, 10))

    chip_hull_pts = compute_chip_convex_hull(main_ft, chip_pts)
    chip_curve_pts = extract_chip_curve_points(main_ft, chip_hull_pts)

    edge_lines = create_edge_lines(chip_curve_pts)
    dist_edge_pt = compute_distance_edge_points(chip_pts, edge_lines)
    nearest_edge_idx = np.argmin(dist_edge_pt, axis=0)
    inside_ft = find_inside_contour(chip_pts, edge_lines, nearest_edge_idx, max_thickness=125.)

    for pt in chip_curve_pts.reshape(-1, 2):
        cv.circle(ft_repr, pt, 3, color=green, thickness=-1)
    for edge in edge_lines:
        geometry.draw_line(ft_repr, edge, yellow, 1)
    for x, y in inside_ft.inside_pts:
        ft_repr[y, x] = red

    return inside_ft, ft_repr


if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt

    img = cv.imread(str(Path('experiments', 'preprocessed_machining_image.png')))
    binary_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    inside_ft, inside_render = render_chip_inside(binary_img)

    cv.imshow('inside', inside_render)
    while cv.waitKey(0) != 113:
        pass
    cv.destroyAllWindows()


    from scipy.signal import savgol_filter
    window_size_sg = 15
    poly_order = 2
    smoothed_thickness = savgol_filter(inside_ft.thickness, window_size_sg, poly_order)

    plt.figure(figsize=(14, 9))
    plt.plot(inside_ft.thickness)
    plt.plot(smoothed_thickness)
    plt.grid()
    plt.show()
