import geometry
from shape_detection.chip_extraction import extract_chip_points

import numpy as np
import cv2 as cv

from scipy import interpolate


# def filter_chip_curve(hull_points, maximum_angle, lateral_margin):
#     """Filter the points of hull to only keep the chip curve"""
#     if len(hull_points) < 3:
#         return hull_points[:]

#     points = hull_points.reshape(-1, 2)
#     for i in range(2, len(points)):
#         a, b, c = points[i-2], points[i-1], points[i]

#         if c[0] < lateral_margin:  # image lateral border reached
#             return hull_points[:i]

#         ab, bc = (b - a), (c - b)
#         angle = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))
#         if angle > maximum_angle:  # sharp angle detected (i.e end of chip reached)
#             return hull_points[:i]

#     return hull_points[:]


def interpolate_curve(curve_points, image_shape):
    h, w = image_shape

    x, y = curve_points[:, 0], curve_points[:, 1]
    x, y = np.r_[x, x[0]], np.r_[y, y[0]]

    tck, u = interpolate.splprep([x, y], s=0, per=True)

    xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
    xi, yi = xi.astype(np.int32), yi.astype(np.int32)

    boundaries = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    return xi[boundaries], yi[boundaries]


def interpolate_chip_hull(binary_img):
    points, base_line, tool_line = extract_chip_points(binary_img)
    hull_points = cv.convexHull(points, clockwise=True)
    hull_interpolation = interpolate_curve(hull_points.reshape(-1, 2), binary_img.shape)
    return hull_interpolation, base_line, tool_line


def render_chip_interpolation(binary_img, render=None):
    if render is None:
        render = np.zeros_like(binary_img)
    else:
        render = render.copy()

    hull_interpolation, base_line, tool_line = interpolate_chip_hull(binary_img)
    x_interpolate, y_interpolate = hull_interpolation

    render[y_interpolate, x_interpolate] = 255
    geometry.draw_line(render, *base_line, color=127, thickness=1)
    geometry.draw_line(render, *tool_line, color=127, thickness=1)

    return render


if __name__ == '__main__':
    import image_loader
    from pathlib import Path

    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("chipcurve", render_chip_interpolation)

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "hull_interpolation")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    processing.compare_frames(20, ("input", "chipcurve"))
    processing.show_video()
    processing.compare_videos(("input", "chipcurve"))
