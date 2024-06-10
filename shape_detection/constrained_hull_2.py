import sys

from shape_detection.point_extraction import extract_points, draw_line

import numpy as np
import cv2 as cv


def above_line(points, a, b, c, min_distance):
    """Filter the points to keep those which verify a*x + b*y + c >= min_distance"""
    x, y = points[:, :, 0], points[:, :, 1]
    mask = (a*x + b*y + c >= min_distance).flatten()
    return points[mask]


def draw_chip_curve_mask(mask, hull_points):
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

    # MOCK: remove the tool and the base
    points, (rho0, xn0, yn0), (rho1, xn1, yn1) = extract_points(precise)

    # compute the convex hull and constrain it to cross an anchor point
    x_min = points[points[:, :, 0].argmin(), 0, 0]
    anchor = np.array([[[x_min, h-1]]])
    hull_points = cv.convexHull(np.vstack((points, anchor)))
    # FIXME: roll the array to force the anchor to be the first point of the hull
    x, y = hull_points[:, 0, 0], hull_points[:, 0, 1]

    # remove points of the convex hull near the tool and the base
    mask = (
        (xn0*x + yn0*y - rho0 + 15 <= 0) &
        (xn1*x + yn1*y - rho1 + 15 <= 0)
    ).flatten()
    filtered_hull_points = hull_points[mask]

    # extract points near the hull
    mask = np.zeros((h, w), dtype=np.uint8)
    draw_chip_curve_mask(mask, filtered_hull_points)

    y, x = np.nonzero(rough & mask)
    chip_curve_points = np.stack((x, y), axis=1).reshape(-1, 1, 2)

    if len(chip_curve_points) >= 5:
        ellipse = cv.fitEllipse(chip_curve_points)
    else:
        print("Warning !: Not enough point to fit an ellipse", file=sys.stderr)

    # display
    to_display = np.zeros((h, w), dtype=np.uint8)
    to_display[y, x] = 255
    draw_line(to_display, rho0, xn0, yn0, 127, 1)
    draw_line(to_display, rho1, xn1, yn1, 127, 1)
    # cv.drawContours(to_display, (hull_points,), 0, 127, 0)
    if len(chip_curve_points) >= 5:
        cv.ellipse(to_display, ellipse, 127, 1)

    return to_display


if __name__ == '__main__':
    from pathlib import Path

    import image_loader
    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("chipcurve", extract_chip_curve, ("erode", "clean"))

    # input_dir = Path("imgs", "vertical")
    input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "chipcurve")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    # processing.show_frame(25)
    processing.compare_frames(25, ("erode", "chipcurve"))
    # processing.show_video()
    processing.compare_videos(("erode", "chipcurve"))


# if __name__ == '__main__':
#     img = cv.cvtColor(cv.imread('preprocessed.png'), cv.COLOR_RGB2GRAY)
#     extracted = extract_chip_curve(img, img)

#     cv.imshow('extracted', extracted)
#     while cv.waitKey(30) != 113:
#         pass
#     cv.destroyAllWindows()
