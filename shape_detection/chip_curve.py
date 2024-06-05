from pathlib import Path

import numpy as np
import cv2 as cv


def above_line(points, a, b, c, min_distance):
    """Filter the points to keep those which verify a*x + b*y + c >= min_distance"""
    x, y = points[..., 0], points[..., 1]
    return points[a*x + b*y + c >= min_distance]


def filter_curve(hull_points, maximum_angle, lateral_margin):
    """Filter the points of hull to only keep the chip curve"""
    if len(hull_points) < 3:
        return hull_points[:]

    points = hull_points.reshape(-1, 2)
    for i in range(2, len(points)):
        a, b, c = points[i-2], points[i-1], points[i]

        if c[0] < lateral_margin:  # image lateral border reached
            return hull_points[:i]

        ab, bc = (b - a), (c - b)
        angle = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))
        if angle > maximum_angle:  # sharp angle detected (i.e end of chip reached)
            return hull_points[:i]

    return hull_points[:]


def extract_chip_curve(binary):
    extracted_shape = binary.copy()
    contours, _hierarchy = cv.findContours(binary[:, :967], cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    points = np.vstack(contours)  # ~ (n, 1, 2)

    # MOCK: remove the tool and the base
    points = above_line(points, a=0, b=-1, c=385, min_distance=5)  # above the base
    points = above_line(points, a=-1, b=0, c=967, min_distance=5)  # at the left of the tool
    origin = np.array([[967, 385]], dtype=points.dtype)

    hull_points = cv.convexHull(points, clockwise=True)  # ~ (p, 1, 2)
    start_index = np.argmin(np.linalg.norm(hull_points - origin, axis=-1))
    hull_points = np.roll(hull_points, -start_index, axis=0)

    curve_points = filter_curve(hull_points, maximum_angle=np.pi/4, lateral_margin=20)

    cv.polylines(extracted_shape, (curve_points,), isClosed=False, color=127, thickness=0)


    # cv.drawContours(extracted_shape, (hull_points,), 0, 127, 0)

    return extracted_shape


if __name__ == '__main__':
    import utils
    import image_loader

    from preprocessing.log_tresh_erode import pipeline as preprocessing

    # binary = cv.cvtColor(cv.imread('morph2.png'), cv.COLOR_RGB2GRAY)

    pipeline = utils.Pipeline()
    pipeline.add("chip_curve", extract_chip_curve)

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "chip_curve")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    preprocessing.run(loader)

    pipeline.run(preprocessing.get_output(), output_dir)
    pipeline.show_samples(20)
    pipeline.show_videos()




# cv.imshow('img', binary)
# cv.imshow('extracted_contours', extracted_shape)
# while cv.waitKey(30) != 113:
#     pass
# cv.destroyAllWindows()

