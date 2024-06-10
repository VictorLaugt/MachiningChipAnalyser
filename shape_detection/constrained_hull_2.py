import sys
from pathlib import Path

import numpy as np
import cv2 as cv


def above_line(points, a, b, c, min_distance):
    """Filter the points to keep those which verify a*x + b*y + c >= min_distance"""
    x, y = points[:, :, 0], points[:, :, 1]
    mask = (a*x + b*y + c >= min_distance).flatten()
    return points[mask]


def draw_chip_curve_mask(mask, hull_points):
    h, w = mask.shape
    pts = hull_points.reshape(-1, 2)

    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i+1]
        if (
                10 < x2 < w-10 and 10 < y2 < h-10 and
                10 < x1 < w-10 and 10 < y1 < h-10
        ):
            cv.line(mask, (x1, y1), (x2, y2), 255, 5)


def extract_chip_curve(precise, rough):
    h, w = precise.shape

    contours, _hierarchy = cv.findContours(precise, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    points = np.vstack(contours)  # ~ (n, 1, 2)

    # MOCK: remove the tool and the base
    points = above_line(points, a=0, b=-1, c=385, min_distance=5)  # above the base
    points = above_line(points, a=-1, b=0, c=967, min_distance=5)  # at the left of the tool

    # compute the convex hull and constrain it to cross an anchor point
    x_min = points[points[:, :, 0].argmin(), 0, 0]
    anchor = np.array([[[x_min, h-1]]])
    points = np.vstack((points, anchor))
    hull_points = cv.convexHull(points)  # ~ (p, 1, 2)

    # remove points of the convex hull near the tool and the base
    hull_points = above_line(hull_points, a=0, b=-1, c=385, min_distance=20)
    hull_points = above_line(hull_points, a=-1, b=0, c=967, min_distance=10)

    # extract points near the hull
    mask = np.zeros((h, w), dtype=np.uint8)
    draw_chip_curve_mask(mask, hull_points)

    y, x = np.nonzero(rough & mask)
    # y, x = np.nonzero(precise & mask) # TODO: test what is the best between extracting from rough or precise
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
    # processing.show_frame(25)
    processing.compare_frames(25, ("erode", "chipcurve"), horizontal=True)
    # processing.show_video()
    processing.compare_videos(("erode", "chipcurve"), horizontal=True)


# if __name__ == '__main__':
#     img = cv.cvtColor(cv.imread('preprocessed.png'), cv.COLOR_RGB2GRAY)
#     extracted = extract_chip_curve(img, img)

#     cv.imshow('extracted', extracted)
#     while cv.waitKey(30) != 113:
#         pass
#     cv.destroyAllWindows()
