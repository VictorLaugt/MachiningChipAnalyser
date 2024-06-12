import numpy as np
import cv2 as cv

import geometry


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

    return (rho_base, xn_base, yn_base), (rho_tool, xn_tool, yn_tool), theta_base, theta_tool


def extract_chip_points(binary_img):
    """Return coordinates of points between base and tool, and line parameters
    for base and tool.
    """
    base_line, tool_line, theta_base, theta_tool = locate_base_and_tool(binary_img)

    contours, _hierarchy = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    points = np.vstack(contours)
    filtered_points = geometry.under_lines(points, (base_line, tool_line), (10, 10))

    return filtered_points, base_line, tool_line, theta_base, theta_tool


def render_chip_extraction(binary_img, render=None):
    if render is None:
        render = np.zeros_like(binary_img)
    else:
        render = render.copy()

    points, base_line, tool_line, _base_angle, _tool_angle = extract_chip_points(binary_img)

    x, y = points[:, 0, 0], points[:, 0, 1]
    render[y, x] = 255

    geometry.draw_line(render, base_line, color=127, thickness=1)
    geometry.draw_line(render, tool_line, color=127, thickness=1)

    return render


if __name__ == '__main__':
    import image_loader
    from pathlib import Path

    import preprocessing.log_tresh_blobfilter_erode

    processing = preprocessing.log_tresh_blobfilter_erode.processing.copy()
    processing.add("chipextraction", render_chip_extraction)

    input_dir = Path("imgs", "vertical")
    # input_dir = Path("imgs", "diagonal")
    output_dir = Path("results", "lines")
    loader = image_loader.ImageLoaderColorConverter(input_dir, cv.COLOR_RGB2GRAY)

    processing.run(loader, output_dir)
    processing.show_frame(21)
    processing.compare_videos(("input", "chipextraction"))
