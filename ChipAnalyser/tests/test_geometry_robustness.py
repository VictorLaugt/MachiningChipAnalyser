import unittest
import numpy as np

import geometry


class TestUnderAboveNan(unittest.TestCase):
    def test_above_nan(self):
        pts = np.random.randint(0, 1000, (500, 2))
        filtered_pts = geometry.above_lines(pts, (geometry.NAN_LINE,), (0,))
        self.assertEqual(filtered_pts.shape, (0, 2))

    def test_under_nan(self):
        pts = np.random.randint(0, 1000, (500, 2))
        filtered_pts = geometry.under_lines(pts, (geometry.NAN_LINE,), (0,))
        self.assertEqual(filtered_pts.shape, (0, 2))


class TestRotateNan(unittest.TestCase):
    @staticmethod
    def random_angle():
        return 2 * np.pi * np.random.random()

    def test_rotate_nan_y_pt(self):
        x, y = np.random.randint(0, 1000), np.nan
        rot_x, rot_y = geometry.rotate(x, y, self.random_angle())
        self.assertTrue(np.isnan(rot_x))
        self.assertTrue(np.isnan(rot_y))

    def test_rotate_nan_x_pt(self):
        x, y = np.nan, np.random.randint(0, 1000)
        rot_x, rot_y = geometry.rotate(x, y, self.random_angle())
        self.assertTrue(np.isnan(rot_x))
        self.assertTrue(np.isnan(rot_y))

    def test_rotate_nan_xy_pt(self):
        x, y = np.nan, np.nan
        rot_x, rot_y = geometry.rotate(x, y, self.random_angle())
        self.assertTrue(np.isnan(rot_x))
        self.assertTrue(np.isnan(rot_y))

    def test_rotate_nan_x_arr(self):
        x, y = 1000 * np.random.random(500), 1000 * np.random.random(500)
        nan_indices = np.random.randint(0, len(x), (len(x) // 4,))
        x[nan_indices] = np.nan

        nan_mask = np.zeros_like(x, dtype=bool)
        nan_mask[nan_indices] = True

        rot_x, rot_y = geometry.rotate(x, y, self.random_angle())
        self.assertTrue(np.all(np.isnan(rot_x[nan_mask])))
        self.assertTrue(np.all(np.isnan(rot_y[nan_mask])))
        self.assertTrue(np.all(~np.isnan(rot_x[~nan_mask])))
        self.assertTrue(np.all(~np.isnan(rot_y[~nan_mask])))

    def test_rotate_nan_y_arr(self):
        x, y = 1000 * np.random.random(500), 1000 * np.random.random(500)
        nan_indices = np.random.randint(0, len(y), (len(y) // 4,))
        y[nan_indices] = np.nan

        nan_mask = np.zeros_like(y, dtype=bool)
        nan_mask[nan_indices] = True

        rot_x, rot_y = geometry.rotate(x, y, self.random_angle())
        self.assertTrue(np.all(np.isnan(rot_x[nan_mask])))
        self.assertTrue(np.all(np.isnan(rot_y[nan_mask])))
        self.assertTrue(np.all(~np.isnan(rot_x[~nan_mask])))
        self.assertTrue(np.all(~np.isnan(rot_y[~nan_mask])))

    def test_rotate_nan_x_arr(self):
        x, y = 1000 * np.random.random(500), 1000 * np.random.random(500)
        nan_indices = np.random.randint(0, len(x), (len(x) // 4,))
        x[nan_indices] = np.nan
        y[nan_indices] = np.nan

        nan_mask = np.zeros_like(x, dtype=bool)
        nan_mask[nan_indices] = True

        rot_x, rot_y = geometry.rotate(x, y, self.random_angle())
        self.assertTrue(np.all(np.isnan(rot_x[nan_mask])))
        self.assertTrue(np.all(np.isnan(rot_y[nan_mask])))
        self.assertTrue(np.all(~np.isnan(rot_x[~nan_mask])))
        self.assertTrue(np.all(~np.isnan(rot_y[~nan_mask])))


class IntersectLine(unittest.TestCase):
    @staticmethod
    def random_line():
        rho = 500 * np.random.random()
        theta = 2 * np.pi * np.random.random()
        return (rho, np.cos(theta), np.sin(theta))

    def test_nan_intersect_line(self):
        with self.assertRaises(ValueError):
            geometry.intersect_line(geometry.NAN_LINE, self.random_line())

    def test_line_intersect_nan(self):
        with self.assertRaises(ValueError):
            geometry.intersect_line(self.random_line(), geometry.NAN_LINE)

    def test_nan_intersect_nan(self):
        with self.assertRaises(ValueError):
            geometry.intersect_line(geometry.NAN_LINE, geometry.NAN_LINE)

    def test_intersect_parallels(self):
        with self.assertRaises(ZeroDivisionError):
            geometry.intersect_line((10., 1., 0.), (100., -1., 0.))

    def test_intersect_parallels_safe(self):
        self.assertIsNone(
            geometry.intersect_line_safe((10., 1., 0.), (100., -1., 0.))
        )


if __name__ == '__main__':
    unittest.main()
