import sys

import numpy as np
import cv2 as cv

import geometry


# ---- detection
# def segment_by_angles(lines, k):
#     """Group lines based on angles with k-means."""
#     angles = 2 * lines[:, 0, 1].reshape(-1, 1)
#     angle_coordinates = np.hstack((np.cos(angles), np.sin(angles)))

#     # criteria = (type, max_iter, epsilon)
#     criteria_type = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER)
#     criteria = (criteria_type, 10, 1.0)
#     flags = cv.KMEANS_RANDOM_CENTERS
#     _compactness, labels, _centers = cv.kmeans(angle_coordinates, k, None, criteria, 10, flags)

#     return labels.flatten()

def best_base_line(lines):
    """Return the best horizontal line."""
    for rho, theta in lines[:, 0, :]:
        if np.abs(theta - np.pi/2) < 0.2:
            return rho, theta

def best_tool_line(lines):
    """Return the best tool line."""
    for rho, theta in lines[:, 0, :]:
        if 0 <= theta <= np.pi/8 or np.pi - 0.2 <= theta <= np.pi:
            return rho, theta

def locate_base_and_tool(binary_img):
    """Compute line parameters for base and tool."""
    lines = cv.HoughLines(binary_img, 1, np.pi/180, 100)
    if lines is None or len(lines) < 2:
        raise ValueError("Warning !: line not found")

    rho_base, theta_base = geometry.positive_rho(*best_base_line(lines))
    rho_tool, theta_tool = geometry.positive_rho(*best_tool_line(lines))

    xn_base, yn_base = np.cos(theta_base), np.sin(theta_base)
    xn_tool, yn_tool = np.cos(theta_tool), np.sin(theta_tool)

    return (rho_base, xn_base, yn_base), (rho_tool, xn_tool, yn_tool)

def extract_points(binary_img):
    """Return coordinates of points between base and tool, and line parameters
    for base and tool.
    """
    (rho0, xn0, yn0), (rho1, xn1, yn1) = locate_base_and_tool(binary_img)

    contours, _hierarchy = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    points = np.vstack(contours)
    x, y = points[:, 0, 0], points[:, 0, 1]

    mask = (
        (xn0*x + yn0*y - rho0 + 10 <= 0) &
        (xn1*x + yn1*y - rho1 + 10 <= 0)
    ).flatten()
    filtered_points = points[mask]

    return filtered_points, (rho0, xn0, yn0), (rho1, xn1, yn1)


# ---- rendering
def draw_line(img, rho, xn, yn, color, thickness):
    """Draw on img the line whose polar parameters are
    (rho, theta), with xn = cos(theta), yn = sin(theta).
    """
    x0, y0 = rho * xn, rho * yn
    x1, y1 = int(x0 - 2000 * yn), int(y0 + 2000 * xn)
    x2, y2 = int(x0 + 2000 * yn), int(y0 - 2000 * xn)
    cv.line(img, (x1, y1), (x2, y2), color, thickness)

def render_point_extraction(binary_img):
    render = np.zeros_like(binary_img)

    points, line0, line1 = extract_points(binary_img)

    x, y = points[:, 0, 0], points[:, 0, 1]
    render[y, x] = 255

    draw_line(render, *line0, color=127, thickness=1)
    draw_line(render, *line1, color=127, thickness=1)

    return render


if __name__ == '__main__':
    import image_loader
    from pathlib import Path

    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("pointextraction", render_point_extraction)

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "lines")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    processing.show_frame(21)
    processing.compare_videos(("input", "pointextraction"))
