import sys

import geometry
from shape_detection.chip_extraction import extract_chip_points

import numpy as np
from numpy.polynomial import Polynomial
import cv2 as cv


def draw_chip_curve(mask, hull_points):
    margin = 5

    h, w = mask.shape
    pts = hull_points.reshape(-1, 2)

    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i+1]
        if (
                margin < x2 < w-margin and margin < y2 < h-margin and
                margin < x1 < w-margin and margin < y1 < h-margin
        ):
            cv.line(mask, (x1, y1), (x2, y2), 255, 5)


def filter_chip_curve_points(curve_points, tool_angle, tool_chip_max_angle, is_clockwise):
    print()

    pi_2 = np.pi/2
    if tool_angle > np.pi:
        tool_angle -= 2*np.pi

    if is_clockwise:
        curve_surface_vectors = curve_points[:-1, 0, :] - curve_points[1:, 0, :]
    else:
        curve_surface_vectors = curve_points[1:, 0, :] - curve_points[:-1, 0, :]

    curve_segment_lengths = np.linalg.norm(curve_surface_vectors, axis=-1)
    curve_vector_angles = np.arccos(curve_surface_vectors[:, 0] / curve_segment_lengths)

    mask = np.zeros(len(curve_points), dtype=bool)
    mask[1:] = np.abs(pi_2 + tool_angle - curve_vector_angles) < tool_chip_max_angle

    print(f"{curve_points = }")
    print(f"{curve_vector_angles = }")

    return curve_points[mask]


def extract_chip_curve(binary_img):
    # TODO: make this part more generic so it can adapt to the chip orientation
    h, w = binary_img.shape
    border_up = (0, 0, -1)
    border_left = (0, -1, 0)
    border_down = (h, 0, 1)
    border_right = (w, 1,  0)

    tool_opposite_border = border_left
    base_opposite_border = border_up
    base_border = border_down

    indirect_rotation = True

    chip_pts, base_line, tool_line, base_angle, tool_angle = extract_chip_points(binary_img)
    tool_base_inter = geometry.intersect_line(tool_line, base_line)

    # compute the chip convex hull and constrain it to cross anchor points
    highest_idx, _ = geometry.line_furthest_point(chip_pts, base_line)
    chip_highest = chip_pts[highest_idx, 0, :]
    anchor_1 = geometry.orthogonal_projection(*chip_highest, tool_opposite_border)
    anchor_2 = geometry.orthogonal_projection(*anchor_1, base_border)
    anchors = np.array([anchor_1, anchor_2, tool_base_inter], dtype=np.int32).reshape(-1, 1, 2)
    chip_hull_pts = cv.convexHull(np.vstack((chip_pts, anchors)), clockwise=indirect_rotation)

    first_pt_idx = np.where(
        (chip_hull_pts[:, 0, 0] == anchors[0, 0, 0]) &
        (chip_hull_pts[:, 0, 1] == anchors[0, 0, 1])
    )[0][0]
    chip_hull_pts = np.roll(chip_hull_pts, -first_pt_idx, axis=0)

    # extract the chip curve points from the convex hull
    _, base_distance = geometry.line_nearest_point(chip_hull_pts, base_line)
    _, tool_distance = geometry.line_nearest_point(chip_hull_pts, tool_line)
    chip_curve_pts = geometry.under_lines(
        chip_hull_pts,
        (base_line, tool_line, base_opposite_border, tool_opposite_border),
        (base_distance+20, tool_distance+5, 15, 15)
    )
    key_pts = filter_chip_curve_points(chip_curve_pts, tool_angle, np.pi/4, indirect_rotation)
    # key_pts = chip_curve_pts

    # Fit a polynomial to the key points
    x, y = geometry.rotate(key_pts[:, 0, 0], key_pts[:, 0, 1], -tool_angle)
    if len(key_pts) < 2:
        print("Warning !: Chip curve not found", file=sys.stderr)
        polynomial = None
    elif len(key_pts) == 2:
        polynomial = Polynomial.fit(x, y, 1)
    else:
        polynomial = Polynomial.fit(x, y, 2)

    return polynomial, chip_hull_pts, key_pts, base_line, tool_line, tool_angle


def render_chip_curve(binary_img, render=None):
    h, w = binary_img.shape

    if render is None:
        render = np.zeros_like(binary_img)
    else:
        render = render.copy()

    polynomial, hull_points, key_points, base_line, tool_line, tool_angle = extract_chip_curve(binary_img)
    if polynomial is not None:
        x = np.arange(0, w, 1, dtype=np.int32)
        y = polynomial(x)
        x, y = geometry.rotate(x, y, tool_angle)
        x, y = x.astype(np.int32), y.astype(np.int32)
        inside_mask = (2 <= x) & (x < w-2) & (2 <= y) & (y < h-2)
        x, y = x[inside_mask], y[inside_mask]
        render[y, x] = render[y, x+1] = render[y, x-1] = render[y+1, x] = render[y-1, x] = 127

    for pt in hull_points.reshape(-1, 2):
        cv.circle(render, pt, 4, 127 // 2, -1)
    for kpt in key_points.reshape(-1, 2):
        cv.circle(render, kpt, 4, 127, -1)

    geometry.draw_line(render, base_line, color=127, thickness=2)
    geometry.draw_line(render, tool_line, color=127, thickness=2)

    return render


if __name__ == '__main__':
    from pathlib import Path

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("chipcurve", render_chip_curve, ("morph", "morph"))

    # input_dir = Path("imgs", "vertical")
    input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "chipcurve")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    processing.compare_frames(15, ("input", "chipcurve"))
    processing.compare_videos(("chipcurve", "input"))


# if __name__ == '__main__':
#     img = cv.cvtColor(cv.imread('preprocessed.png'), cv.COLOR_RGB2GRAY)
#     extracted = render_chip_curve(img, img, render=img)

#     cv.imshow('extracted', extracted)
#     while cv.waitKey(30) != 113:
#         pass
#     cv.destroyAllWindows()
