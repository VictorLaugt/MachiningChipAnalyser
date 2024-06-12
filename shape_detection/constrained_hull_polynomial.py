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


def extract_chip_curve(precise, rough):
    h, w = precise.shape
    border_line_up = (0, 0, -1)
    border_line_left = (0, -1, 0)
    # border_line_down = (h, 0, 1)
    # border_line_right = (w, 1,  0)

    points, base_line, tool_line, base_angle, tool_angle = extract_chip_points(precise)

    # compute the convex hull and constrain it to cross an anchor point
    y_min = points[points[:, :, 1].argmin(), 0, 1]
    anchor = np.array([[[0, y_min]], [[0, h-1]]])
    hull_points = cv.convexHull(np.vstack((points, anchor)))
    first_point_index = np.where(
        (hull_points[:, 0, 0] == anchor[0, 0, 0]) &
        (hull_points[:, 0, 1] == anchor[0, 0, 1])
    )[0][0]
    hull_points = np.roll(hull_points, -first_point_index, axis=0)

    # filter points from the convex hull
    _, base_distance = geometry.line_nearest_point(hull_points, base_line)
    _, tool_distance = geometry.line_nearest_point(hull_points, tool_line)
    key_points = geometry.under_lines(
        hull_points[2:],
        (base_line, tool_line, border_line_up, border_line_left),
        (base_distance+20, tool_distance+5, 15, 15)
    )
    # key_points = geometry.under_lines(
    #     hull_points[2:],
    #     (base_line, tool_line, border_line_up),
    #     (20, 20, 15)
    # )


    # extract points of the chip curve
    # chip_curve_mask = np.zeros((h, w), dtype=np.uint8)
    # draw_chip_curve(chip_curve_mask, key_points)
    # y, x = np.nonzero(rough & chip_curve_mask)
    # key_points = np.stack((x, y), axis=1).reshape(-1, 1, 2)

    x, y = geometry.rotate(key_points[:, 0, 0], key_points[:, 0, 1], -tool_angle)

    # Fit a polynomial to the key points
    if len(key_points) < 2:
        print("Warning !: Chip curve not found", file=sys.stderr)
        polynomial = None
    elif len(key_points) == 2:
        polynomial = Polynomial.fit(x, y, 1)
    else:
        polynomial = Polynomial.fit(x, y, 2)

    return polynomial, hull_points, key_points, base_line, tool_line, tool_angle


def render_chip_curve(precise, rough, render=None):
    h, w = precise.shape

    if render is None:
        render = np.zeros_like(precise)
    else:
        render = render.copy()

    polynomial, hull_points, key_points, base_line, tool_line, tool_angle = extract_chip_curve(precise, rough)
    if polynomial is not None:
        x = np.arange(0, w, 1, dtype=np.int32)
        y = polynomial(x)
        x, y = geometry.rotate(x, y, tool_angle)
        x, y = x.astype(np.int32), y.astype(np.int32)
        inside_mask = (0 <= x) & (x < w) & (0 <= y) & (y < h)
        x, y = x[inside_mask], y[inside_mask]
        render[y, x] = 127

    geometry.draw_line(render, base_line, color=127, thickness=1)
    geometry.draw_line(render, tool_line, color=127, thickness=1)

    for pt in hull_points.reshape(-1, 2):
        cv.circle(render, pt, 7, 127 // 2, -1)
    for kpt in key_points.reshape(-1, 2):
        cv.circle(render, kpt, 7, 127, -1)

    return render


if __name__ == '__main__':
    from pathlib import Path

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("chipcurve", render_chip_curve, ("morph", "blobfilter", "morph"))

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
