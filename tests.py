import unittest

import numpy as np

from main import find_nearby_maxima


class Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.array([[40, 40, 39, 39, 38],
                               [40, 41, 39, 39, 39],
                               [30, 30, 30, 32, 32],
                               [33, 33, 30, 32, 35],
                               [30, 30, 30, 33, 36]], dtype=np.uint8)

    def test(self):
        seeds = ((1, 0), (2, 0), (0, 4), (2, 4), (3, 4), (4, 4))
        correct = ((1, 1), (1, 1), (1, 1), (1, 1), (4, 4), (4, 4))
        self.assertEqual(correct, tuple(find_nearby_maxima(self.image, seeds)))

    def test_floats(self):
        rng = np.random.default_rng(2021)
        image = rng.random((10, 10))
        seeds = ((0, 4), (2, 4), (3, 4), (4, 4))
        correct = ((1, 4), (1, 4), (1, 4), (4, 3))
        self.assertEqual(correct, tuple(find_nearby_maxima(image, seeds)))

    def test_nans(self):
        image_nans = self.image.astype(float)
        image_nans[:, 2] = np.nan
        seeds = ((1, 0), (2, 0), (0, 4), (2, 4), (3, 4), (4, 4))
        correct = ((1, 1), (1, 1), (1, 3), (1, 3), (4, 4), (4, 4))
        self.assertEqual(correct, tuple(find_nearby_maxima(image_nans, seeds)))
