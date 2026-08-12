import unittest
import numpy as np

from dynasid.geo import haversine_km, pairwise_haversine_km


class GeoTests(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(haversine_km(40.7, -74.0, 40.7, -74.0), 0.0)

    def test_known_latitude_degree(self):
        self.assertAlmostEqual(haversine_km(0, 0, 1, 0), 111.195, places=2)

    def test_pairwise_symmetry(self):
        matrix = pairwise_haversine_km([0, 1, 2], [0, 0, 0])
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), 0)


if __name__ == "__main__":
    unittest.main()
