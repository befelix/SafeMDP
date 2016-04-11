from __future__ import division, print_function, absolute_import

import unittest
import GPy
import numpy as np
import networkx as nx
from numpy.testing import *

from .utilities import (DifferenceKernel, max_out_degree, reachable_set,
                        grid_world_graph)


class DifferenceKernelTest(unittest.TestCase):

    @staticmethod
    def _check(gp, x1, x2):
        """Compare the gp difference predictions on X1 and X2.

        Parameters
        ----------
        gp: GPy.core.GP
        x1: np.array
        x2: np.array
        """
        n = x1.shape[0]

        # Difference prediction with library
        a = np.hstack((np.eye(n), -np.eye(n)))
        m1, v1 = gp.predict_noiseless(np.vstack((x1, x2)), full_cov=True)
        m1 = a.dot(m1)
        v1 = np.linalg.multi_dot((a, v1, a.T))

        # Predict diagonal
        m2, v2 = gp.predict_noiseless(np.hstack((x1, x2)),
                                      kern=DifferenceKernel(gp.kern),
                                      full_cov=False)

        assert_allclose(m1, m2)
        assert_allclose(np.diag(v1), v2.squeeze())

        # Predict full covariance
        m2, v2 = gp.predict_noiseless(np.hstack((x1, x2)),
                                      kern=DifferenceKernel(gp.kern),
                                      full_cov=True)

        assert_allclose(m1, m2)
        assert_allclose(v1, v2, atol=1e-12)

    def test_1d(self):
        """Test the difference kernel for a 1D input."""
        # Create some GP model
        kernel = GPy.kern.RBF(input_dim=1, lengthscale=0.05)
        likelihood = GPy.likelihoods.Gaussian(variance=0.005 ** 2)
        x = np.linspace(0, 1, 5)[:, None]
        y = x ** 2
        gp = GPy.core.GP(x, y, kernel, likelihood)

        # Create test points
        n = 10
        x1 = np.linspace(0, 1, n)[:, None]
        x2 = x1 + np.linspace(0, 0.1, n)[::-1, None]

        self._check(gp, x1, x2)

    def test_2d(self):
        """Test the difference kernel for a 2D input."""

        # Create some GP model
        kernel = GPy.kern.RBF(input_dim=2, lengthscale=0.05)
        likelihood = GPy.likelihoods.Gaussian(variance=0.005 ** 2)
        x = np.hstack((np.linspace(0, 1, 5)[:, None],
                       np.linspace(0.5, 1.5, 5)[:, None]))
        y = x[:, [0]] ** 2 + x[:, [1]] ** 2
        gp = GPy.core.GP(x, y, kernel, likelihood)

        # Create test points
        n = 10

        x1 = np.hstack((np.linspace(0, 1, n)[:, None],
                        np.linspace(0.5, 1.5, n)[:, None]))
        x2 = x1 + np.hstack((np.linspace(0, 0.1, n)[::-1, None],
                             np.linspace(0., 0.1, n)[::-1, None]))

        self._check(gp, x1, x2)


class MaxOutDegreeTest(unittest.TestCase):
    def test_all(self):
        """Test the max_out_degree function."""
        graph = nx.DiGraph()
        graph.add_edges_from(((0, 1),
                              (1, 2),
                              (2, 3),
                              (3, 1)))
        assert_(max_out_degree(graph), 1)

        graph.add_edge(0, 2)
        assert_(max_out_degree(graph), 2)

        graph.add_edge(2, 3)
        assert_(max_out_degree(graph), 2)

        graph.add_edge(3, 2)
        assert_(max_out_degree(graph), 2)

        graph.add_edge(3, 1)
        assert_(max_out_degree(graph), 3)


class ReachableSetTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ReachableSetTest, self).__init__(*args, **kwargs)
        #             3
        #             ^
        #             |
        # 0 --> 1 --> 2 --> 0
        #       ^
        #       |
        #       4
        self.graph = nx.DiGraph()
        self.graph.add_edges_from([(0, 1),
                                   (1, 2),
                                   (2, 0),
                                   (2, 3),
                                   (4, 1)], action=1)

        self.safe_set = np.ones((self.graph.number_of_nodes(),
                                 max_out_degree(self.graph) + 1),
                                dtype=np.bool)
        self.true = np.array([1, 1, 1, 1, 0], dtype=np.bool)

    def setUp(self):
        self.safe_set[:] = True

    def _check(self, graph=None):
        if graph is None:
            graph = self.graph
        reach = reachable_set(graph, [0], self.safe_set)
        assert_equal(reach, self.true)

    def test_all_safe(self):
        """Test reachable set if everything is safe"""
        self.true[:] = [1, 1, 1, 1, 0]
        self._check()

    def test_all_safe_inverse(self):
        """Test reachable set on inverse graph if everything is safe"""
        self.true[:] = [1, 1, 1, 0, 1]
        self._check(graph=self.graph.reverse())

    def test_unsafe1(self):
        """Test safety aspect"""
        self.safe_set[1] = False
        self.true[:] = [1, 0, 0, 0, 0]
        self._check()

    def test_unsafe2(self):
        """Test safety aspect"""
        self.safe_set[2] = False
        self.true[:] = [1, 1, 0, 0, 0]
        self._check()

    def test_unsafe3(self):
        """Test safety aspect"""
        self.safe_set[3] = False
        self.true[:] = [1, 1, 1, 0, 0]
        self._check()

    def test_unsafe4(self):
        """Test safety aspect"""
        self.safe_set[4] = False
        self.true[:] = [1, 1, 1, 1, 0]
        self._check()

    def test_out(self):
        """Test writing the output"""
        self.safe_set[3] = False
        self.true[:] = [1, 1, 1, 0, 0]
        out = np.zeros_like(self.true)
        reachable_set(self.graph, [0], self.safe_set, out=out)
        assert_equal(out, self.true)

    def test_error(self):
        """Check error condition"""
        with assert_raises(AttributeError):
            reachable_set(self.graph, [], self.safe_set)


class GridWorldGraphTest(unittest.TestCase):
    """Test the grid_world_graph function."""

    def test(self):
        """Simple test"""
        # 1 2 3
        # 4 5 6
        graph = grid_world_graph((2, 3))
        graph_true = nx.DiGraph()
        graph_true.add_edges_from(((1, 2),
                                   (2, 3),
                                   (4, 5),
                                   (5, 6)),
                                  action=1)
        graph_true.add_edges_from(((1, 4),
                                   (2, 5),
                                   (3, 6)),
                                  action=2)
        graph_true.add_edges_from(((2, 1),
                                   (3, 2),
                                   (5, 4),
                                   (6, 5)),
                                  action=3)
        graph_true.add_edges_from(((4, 1),
                                   (5, 2),
                                   (6, 3)),
                                  action=4)

        assert_(nx.is_isomorphic(graph, graph_true))


if __name__ == '__main__':
    unittest.main()
