import unittest

import numpy as np

from gui import log_interp_clamped


class TestWeightingInterpolation(unittest.TestCase):
    def test_log_interp_midpoint(self):
        xs = np.asarray([200.0, 210.0], dtype=float)
        ys = np.asarray([1e-3, 1e-1], dtype=float)
        out = log_interp_clamped(205.0, xs, ys)
        self.assertAlmostEqual(out, 1e-2, places=12)

    def test_log_interp_clamps_to_endpoints(self):
        xs = np.asarray([200.0, 210.0, 220.0], dtype=float)
        ys = np.asarray([1e-3, 1e-2, 1e-1], dtype=float)
        self.assertAlmostEqual(log_interp_clamped(190.0, xs, ys), 1e-3, places=12)
        self.assertAlmostEqual(log_interp_clamped(225.0, xs, ys), 1e-1, places=12)

    def test_log_interp_zero_cutoff(self):
        xs = np.asarray([200.0, 210.0], dtype=float)
        ys = np.asarray([1e-3, 1e-2], dtype=float)
        self.assertEqual(log_interp_clamped(400.0, xs, ys, zero_at_or_above=400.0), 0.0)
        self.assertEqual(log_interp_clamped(450.0, xs, ys, zero_at_or_above=400.0), 0.0)

    def test_log_interp_handles_empty(self):
        self.assertEqual(log_interp_clamped(222.0, np.asarray([]), np.asarray([])), 0.0)


if __name__ == "__main__":
    unittest.main()
