from unittest import TestCase
import numpy as np
import networkx as nx

import cnvm.collective_variables as cv


class TestOpinionShares(TestCase):
    def test_default(self):
        shares = cv.OpinionShares(3)

        x = np.array([[0, 1, 0, 1, 2, 2, 1, 1, 1, 0]])
        self.assertTrue(np.allclose(shares(x), np.array([[3, 5, 2]])))

        x = np.array([[0, 1, 0, 1, 0, 0, 0],
                      [1, 1, 1, 2, 1, 2, 2]])
        c = np.array([[5, 2, 0], [0, 4, 3]])
        self.assertTrue(np.allclose(shares(x), c))

        x = np.array([[1, 1, 1, 2, 1, 2, 2]])
        self.assertTrue(np.allclose(shares(x), np.array([[0, 4, 3]])))

        x = np.array([[]])
        self.assertTrue(np.allclose(shares(x), np.array([[0, 0, 0]])))

    def test_normalized(self):
        shares = cv.OpinionShares(3, normalize=True)

        x = np.array([[0, 1, 0, 1, 2, 2, 1, 1, 1, 0]])
        self.assertTrue(np.allclose(shares(x), np.array([[0.3, 0.5, 0.2]])))

        x = np.array([[0, 1, 0, 1, 0]])
        self.assertTrue(np.allclose(shares(x), np.array([[0.6, 0.4, 0]])))

    def test_weights(self):
        weights = np.array([1, 0, 0.5, 0, 1])
        shares = cv.OpinionShares(3, weights=weights)

        x = np.array([[0, 1, 0, 2, 1]])
        self.assertTrue(np.allclose(shares(x), np.array([[1.5, 1, 0]])))

        shares = cv.OpinionShares(3, weights=weights, normalize=True)
        x = np.array([[0, 0, 1, 1, 2]])
        self.assertTrue(np.allclose(shares(x), np.array([[1 / 2.5, 0.5 / 2.5, 1 / 2.5]])))

    def test_idx_to_return(self):
        x = np.array([[0, 1, 0, 1, 2, 2, 1, 1, 1, 0]])

        shares = cv.OpinionShares(3, idx_to_return=1)
        self.assertTrue(np.allclose(shares(x), np.array([[5]])))

        shares = cv.OpinionShares(3, idx_to_return=np.array([0, 1]))
        self.assertTrue(np.allclose(shares(x), np.array([[3, 5]])))

        shares = cv.OpinionShares(3, idx_to_return=np.array([2, 0, 1]))
        self.assertTrue(np.allclose(shares(x), np.array([[2, 3, 5]])))


class TestOpinionSharesByDegree(TestCase):
    def setUp(self):
        edges = [(0, 1), (0, 2), (0, 3), (0, 4),
                 (1, 2), (2, 3), (2, 4)]
        self.network = nx.Graph(edges)
        self.degrees = [4, 2, 4, 2, 2]

    def test_default(self):
        shares = cv.OpinionSharesByDegree(3, self.network)
        x = np.array([[0, 1, 0, 2, 1],
                      [1, 1, 1, 0, 0]])
        c = np.array([[0, 2, 1, 2, 0, 0],
                      [2, 1, 0, 0, 2, 0]])
        self.assertTrue(np.allclose(shares(x), c))

    def test_normalize(self):
        shares = cv.OpinionSharesByDegree(3, self.network, normalize=True)
        x = np.array([[0, 1, 0, 2, 1],
                      [1, 1, 1, 0, 0]])
        c = np.array([[0, 2 / 3, 1 / 3, 1, 0, 0],
                      [2 / 3, 1 / 3, 0, 0, 1, 0]])
        self.assertTrue(np.allclose(shares(x), c))

    def test_idx_to_return(self):
        shares = cv.OpinionSharesByDegree(3, self.network, idx_to_return=0)
        x = np.array([[0, 1, 0, 2, 1],
                      [1, 1, 1, 0, 0]])
        c = np.array([[0, 2],
                      [2, 0]])
        self.assertTrue(np.allclose(shares(x), c))

        shares = cv.OpinionSharesByDegree(3, self.network, idx_to_return=np.array([2, 0]))
        x = np.array([[0, 1, 0, 2, 1],
                      [1, 1, 1, 0, 0]])
        c = np.array([[1, 0, 0, 2],
                      [0, 2, 0, 0]])
        self.assertTrue(np.allclose(shares(x), c))


class TestCompositeCollectiveVariable(TestCase):
    def setUp(self):
        weights1 = np.array([0, 1, 0, 1, 1])
        self.shares1 = cv.OpinionShares(num_opinions=2, weights=weights1, idx_to_return=0)
        weights2 = np.array([1, 1, 0, 0, 0])
        self.shares2 = cv.OpinionShares(num_opinions=2, weights=weights2, idx_to_return=0)

    def test_composite_cv(self):
        composite = cv.CompositeCollectiveVariable([self.shares1, self.shares2])
        x = np.array([[0, 1, 0, 0, 1],
                      [1, 1, 0, 0, 0]])
        c = np.array([[1, 1],
                      [2, 0]])
        self.assertTrue(np.allclose(composite(x), c))
