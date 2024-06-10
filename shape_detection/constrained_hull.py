import sys
from pathlib import Path

import numpy as np
import cv2 as cv


def above_line(points, a, b, c, min_distance):
    """Filter the points to keep those which verify a*x + b*y + c >= min_distance"""
    x, y = points[:, :, 0], points[:, :, 1]
    mask = (a*x + b*y + c >= min_distance).flatten()
    return points[mask]


def draw_lines(img, points, color, thickness):
    points = points.reshape(-1, 2)
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        cv.line(img, start_point, end_point, color, thickness)


def extract_chip_curve(precise, rough):
    h, w = precise.shape

    contours, _hierarchy = cv.findContours(precise, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    points = np.vstack(contours)  # ~ (n, 1, 2)

    # MOCK: remove the tool and the base
    points = above_line(points, a=0, b=-1, c=385, min_distance=5)  # above the base
    points = above_line(points, a=-1, b=0, c=967, min_distance=5)  # at the left of the tool

    # compute the convex hull and constrain it to cross two anchor points
    y_min = points[points[:, :, 1].argmin(), 0, 1]
    anchors = np.array([
        [[0, y_min]],
        [[0, h-1]]
    ])
    points = np.vstack((points, anchors))
    hull_points = cv.convexHull(points, clockwise=True)  # ~ (p, 1, 2)

    # remove points of the convex hull near the tool, the base, and the image borders
    hull_points = above_line(hull_points, a=0, b=-1, c=385, min_distance=20)
    hull_points = above_line(hull_points, a=-1, b=0, c=967, min_distance=10)
    hull_points = above_line(hull_points, a=0, b=1, c=0, min_distance=5)
    hull_points = above_line(hull_points, a=1, b=0, c=0, min_distance=5)

    mask = np.zeros((h, w), dtype=np.uint8)
    draw_lines(mask, hull_points, 255, 5)
    y, x = np.nonzero(rough & mask)
    chip_curve_points = np.stack((x, y), axis=1).reshape(-1, 1, 2)

    if len(chip_curve_points) >= 5:
        ellipse = cv.fitEllipse(chip_curve_points)
    else:
        print("Warning !: Not enough point to fit an ellipse", file=sys.stderr)

    # display
    to_display = np.zeros((h, w), dtype=np.uint8)
    to_display[y, x] = 255
    # for pt in hull_points.reshape(-1, 2):
    #     cv.circle(to_display, pt, 5, 127, -1)
    if len(chip_curve_points) >= 5:
        cv.ellipse(to_display, ellipse, 127, 1)
    else:
        draw_lines(to_display, chip_curve_points, 127, 1)

    return to_display


if __name__ == '__main__':
    import image_loader

    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("chipcurve", extract_chip_curve, ("erode", "clean"))

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "chipcurve")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    processing.compare_frames(20, ("input", "chipcurve"))
    processing.show_video()
    processing.compare_videos(("input", "chipcurve"))
