from pathlib import Path

import numpy as np
import cv2 as cv


def above_line(points, a, b, c, min_distance):
    """Filter the points to keep those which verify a*x + b*y + c >= min_distance"""
    x, y = points[:, :, 0], points[:, :, 1]
    mask = (a*x + b*y + c >= min_distance).flatten()
    return points[mask]


def draw_lines(img, points):
    points = points.reshape(-1, 2)
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        cv.line(img, start_point, end_point, 127, 1)


def extract_chip_curve(binary):
    h, w = binary.shape

    extracted_shape = binary.copy()
    contours, _hierarchy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
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

    if len(hull_points) >= 5:
        ellipse = cv.fitEllipse(hull_points)

    # display
    for pt in hull_points.reshape(-1, 2):
        cv.circle(extracted_shape, pt, 5, 127, -1)
    if len(hull_points) >= 5:
        cv.ellipse(extracted_shape, ellipse, 127, 1)
    else:
        draw_lines(extracted_shape, hull_points)

    return extracted_shape


if __name__ == '__main__':
    import utils
    import image_loader

    import preprocessing.log_tresh_blobfilter_erode

    shape_detection = utils.DagProcess()
    shape_detection.add("chipcurve", extract_chip_curve)

    pipeline = preprocessing.log_tresh_blobfilter_erode.pipeline.then(shape_detection)

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "chipcurve")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    pipeline.run(loader, output_dir)
    pipeline.compare_frames(20, ("input", "chipcurve"))
    pipeline.show_video()
    pipeline.compare_videos(("input", "chipcurve"))
