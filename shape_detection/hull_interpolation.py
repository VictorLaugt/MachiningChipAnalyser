from pathlib import Path

import numpy as np
import cv2 as cv

from scipy import interpolate


def above_line(points, a, b, c, min_distance):
    """Filter the points to keep those which verify a*x + b*y + c >= min_distance"""
    x, y = points[..., 0], points[..., 1]
    return points[a*x + b*y + c >= min_distance]


# def filter_curve(hull_points, maximum_angle, lateral_margin):
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


def extract_chip_curve(binary):
    extracted_shape = binary.copy()
    contours, _hierarchy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    points = np.vstack(contours)  # ~ (n, 1, 2)

    # MOCK: remove the tool and the base
    points = above_line(points, a=0, b=-1, c=385, min_distance=5)  # above the base
    points = above_line(points, a=-1, b=0, c=967, min_distance=5)  # at the left of the tool

    hull_points = cv.convexHull(points, clockwise=True)  # ~ (p, 1, 2)

    x_interpolate, y_interpolate = interpolate_curve(hull_points.reshape(-1, 2), binary.shape)

    # display the convex hull and its interpolation
    for pt in hull_points.reshape(-1, 2):
        cv.circle(extracted_shape, pt, 5, 127, -1)
    extracted_shape[y_interpolate, x_interpolate] = 127

    return extracted_shape


if __name__ == '__main__':
    import utils
    import image_loader

    import preprocessing.log_tresh_blobfilter_erode

    shape_detection = utils.DagProcess()
    shape_detection.add("chipcurve", extract_chip_curve)

    processing = preprocessing.log_tresh_blobfilter_erode.processing.then(shape_detection)

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "hull_interpolation")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    processing.compare_frames(20, ("input", "chipcurve"))
    processing.show_video()
    processing.compare_videos(("input", "chipcurve"))
