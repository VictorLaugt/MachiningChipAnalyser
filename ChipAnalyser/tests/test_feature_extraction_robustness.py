from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from type_hints import IntPt

import unittest

from dataclasses import dataclass
import numpy as np

import features_main
import features_contact


class TestRobustMainFeatures(unittest.TestCase):
    def test_extract_main_features_robustness(self):
        empty_img = np.zeros((500, 900), dtype=np.uint8)
        main_ft = features_main.extract_main_features(empty_img)
        self.assertTrue(main_ft is features_main.FAILURE)


class TestRobustContactFeatures(unittest.TestCase):
    @dataclass
    class MainFeaturesMock:
        indirect_rotation: bool
        tool_angle: bool
        tool_base_inter_pt: IntPt

    def test_extract_contact_features_robustness(self):
        main_ft = self.MainFeaturesMock(True, np.pi/6, (50, 50))
        out_curve = np.empty((0, 2), dtype=int)
        contact_ft = features_contact.extract_contact_features(main_ft, out_curve)

        self.assertEqual(len(contact_ft.key_pts), 0)
        self.assertTrue(np.isnan(contact_ft.polynomial.coef[0]))
        xc, yc = contact_ft.contact_pt
        self.assertTrue(np.isnan(xc) and np.isnan(yc))


if __name__ == '__main__':
    unittest.main()
