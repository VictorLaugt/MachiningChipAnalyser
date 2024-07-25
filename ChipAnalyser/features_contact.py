from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from type_hints import ColorImage, OpenCVIntArray
    from chip_extract import MainFeatures


import sys
import matplotlib.pyplot as plt

import geometry
import colors

import numpy as np
from numpy.polynomial import Polynomial
import cv2 as cv


class ContactFeatures:
    __slots__ = (
        "contact_point",  # type: FloatPt
        "key_pts",        # type: OpenCVIntArray
        "polynomial"      # type: Polynomial
    )


def extract_key_points(main_ft: MainFeatures, curve_points: OpenCVIntArray, tool_chip_max_angle: float) -> OpenCVIntArray:
    """Return key points from the chip curve which can then be use to fit a polynomial."""
    if main_ft.tool_angle > np.pi:
        tool_angle = main_ft.tool_angle - 2*np.pi
    else:
        tool_angle = main_ft.tool_angle

    if main_ft.indirect_rotation:
        curve_surface_vectors = curve_points[:-1, 0, :] - curve_points[1:, 0, :]
    else:
        curve_surface_vectors = curve_points[1:, 0, :] - curve_points[:-1, 0, :]

    curve_segment_lengths = np.linalg.norm(curve_surface_vectors, axis=-1)
    curve_vector_angles = np.arccos(curve_surface_vectors[:, 0] / curve_segment_lengths)

    mask = np.ones(len(curve_points), dtype=bool)
    mask[1:] = np.abs(np.pi/2 + tool_angle - curve_vector_angles) < tool_chip_max_angle

    return curve_points[mask]


def fit_polynomial(main_ft: MainFeatures, key_pts: OpenCVIntArray) -> Polynomial:
    """Return a polynomial of degree 2 which fits the key points."""
    rot_x, rot_y = geometry.rotate(key_pts[:, 0, 0], key_pts[:, 0, 1], -main_ft.tool_angle)

    if len(key_pts) < 2:
        print("Warning !: Cannot fit the chip curve", file=sys.stderr)
        polynomial = None
    elif len(key_pts) == 2:
        polynomial = Polynomial.fit(rot_x, rot_y, 1)
    else:
        polynomial = Polynomial.fit(rot_x, rot_y, 2)

    return polynomial


def chip_tool_contact_point(main_ft: MainFeatures, polynomial: Polynomial) -> tuple[float, float]:
    """Return the contact point between the tool and the chip curve."""
    # abscissa of the contact point, rotated in the polynomial basis
    xi, yi = main_ft.tool_base_intersection
    cos, sin = np.cos(main_ft.tool_angle), -np.sin(main_ft.tool_angle)
    rot_xc = xi*cos - yi*sin

    # ordinate of the contact point, rotated in the polynomial basis
    rot_yc = polynomial(rot_xc)

    # contact point, rotated back in the image basis
    return geometry.rotate(rot_xc, rot_yc, main_ft.tool_angle)


def extract_contact_features(main_ft: MainFeatures, outside_segments: OpenCVIntArray) -> ContactFeatures:
    contact_ft = ContactFeatures()

    contact_ft.key_pts = extract_key_points(main_ft, outside_segments, np.pi/4)
    contact_ft.polynomial = fit_polynomial(main_ft, contact_ft.key_pts)
    contact_ft.contact = chip_tool_contact_point(main_ft, contact_ft.polynomial)

    return contact_ft


def render_contact_features(render: ColorImage, main_ft: MainFeatures, contact_ft: ContactFeatures) -> ColorImage:
    contact_line = geometry.parallel(main_ft.base_line, *contact_ft.contact_point)
    if contact_ft.polynomial is not None:
        # FIXME: clarify these type castings
        x = np.arange(0, render.shape[1], 1, dtype=np.int32)
        y = contact_ft.polynomial(x)
        x, y = geometry.rotate(x, y, main_ft.tool_angle)
        x, y = x.astype(np.int32), y.astype(np.int32)
        for i in range(len(x)-1):
            cv.line(render, (x[i], y[i]), (x[i+1], y[i+1]), colors.BLUE, thickness=2)

    geometry.draw_line(render, main_ft.base_line, colors.RED, thickness=3)
    geometry.draw_line(render, main_ft.tool_line, colors.RED, thickness=3)
    geometry.draw_line(render, contact_line, colors.YELLOW, thickness=1)

    for kpt in contact_ft.key_pts.reshape(-1, 2):
        cv.circle(render, kpt, 6, colors.GREEN, thickness=-1)


# class ChipFeatureCollector:
#     def __init__(self, scale: float=1.0):
#         self.scale = scale
#         self.chip_features: list[ContactFeatures] = []
#         self.main_features: list[MainFeatures] = []
#         self.contact_lengths: list[float] = []

#     def collect(self, main_ft: MainFeatures, chip_ft: ContactFeatures) -> None:
#         xi, yi = main_ft.tool_base_intersection
#         xc, yc = chip_ft.contact_point
#         self.main_features.append(main_ft)
#         self.chip_features.append(chip_ft)
#         self.contact_lengths.append(self.scale * np.linalg.norm((xc-xi, yc-yi)))

#     def extract_and_render(self, binary_img: np.ndarray, background: Optional[np.ndarray]=None) -> np.ndarray:
#         main_ft, chip_ft = extract_contact_features(binary_img)
#         self.collect(main_ft, chip_ft)
#         if background is None:
#             ft_repr = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
#             render_chip_features(ft_repr, main_ft, chip_ft)
#             ft_repr[binary_img > 0] = (255, 255, 255)
#         else:
#             ft_repr = background.copy()
#             render_chip_features(ft_repr, main_ft, chip_ft)
#         return ft_repr

#     def show_contact_length_graph(self) -> None:
#         plt.figure(figsize=(10, 5))
#         plt.plot(range(1, len(self.contact_lengths)+1), self.contact_lengths, 'x-')
#         plt.xlabel('frame')
#         plt.ylabel('contact length (µm)')
#         plt.grid()
#         plt.show()
