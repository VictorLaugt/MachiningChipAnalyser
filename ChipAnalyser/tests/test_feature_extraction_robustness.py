from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import IntPt, Line

import unittest

from dataclasses import dataclass
import numpy as np

from features_main import extract_main_features
from features_contact import extract_contact_features
from features_tip import locate_tool_tip


class TestRobustMainFeatures(unittest.TestCase):
    def test_extract_main_features_robustness(self):
        empty_img = np.zeros((500, 900), dtype=np.uint8)
        main_ft = extract_main_features(empty_img)
        self.assertIsNone(main_ft)


class TestRobustContactFeatures(unittest.TestCase):
    @dataclass
    class MainFeaturesMock:
        indirect_rotation: bool
        tool_angle: bool
        tool_base_inter_pt: IntPt

    def test_extract_contact_features_robustness(self):
        main_ft = self.MainFeaturesMock(True, np.pi/6, (50, 50))
        out_curve = np.empty((0, 2), dtype=int)
        contact_ft = extract_contact_features(main_ft, out_curve)

        self.assertEqual(len(contact_ft.key_pts), 0)
        self.assertTrue(np.isnan(contact_ft.polynomial.coef[0]))
        xc, yc = contact_ft.contact_pt
        self.assertTrue(np.isnan(xc) and np.isnan(yc))


class TestRobustTipFeatures(unittest.TestCase):
    @dataclass
    class MainFeaturesMock:
        base_line: Line
        tool_line: Line

    @staticmethod
    def random_line():
        rho = 500 * np.random.random()
        theta = 2 * np.pi * np.random.random()
        return (rho, np.cos(theta), np.sin(theta))

    def test_extract_tip_features_robustness_no_main_features(self):
        batch_size = 10
        empty_images = [np.random.randint(0, 256, (500, 900), dtype=np.uint8)] * batch_size
        main_features = [None] * batch_size
        tip_ft = locate_tool_tip(empty_images, main_features)
        self.assertIsNone(tip_ft)

    def test_extract_tip_features_robustness_invalid_images(self):
        batch_size = 10
        empty_images = [np.zeros((500, 900), dtype=np.uint8)] * batch_size
        main_features = [
            self.MainFeaturesMock(self.random_line(), self.random_line())
            for _ in range(batch_size)
        ]
        tip_ft = locate_tool_tip(empty_images, main_features)
        self.assertIsNone(tip_ft)


if __name__ == '__main__':
    unittest.main()
