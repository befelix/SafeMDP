from __future__ import division, print_function, absolute_import

import unittest
import GPy
import numpy as np
import networkx as nx
from numpy.testing import *

from .utilities import *

from safemdp.SafeMDP_class import reachable_set, returnable_set
from safemdp.grid_world import compute_true_safe_set, grid_world_graph
from .SafeMDP_class import link_graph_and_safe_set


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
                                   (4, 1)], action=1)
        self.graph.add_edge(2, 3, action=2)

        self.safe_set = np.ones((self.graph.number_of_nodes(),
                                 max_out_degree(self.graph) + 1),
                                dtype=np.bool)
        link_graph_and_safe_set(self.graph, self.safe_set)
        self.true = np.zeros(self.safe_set.shape[0], dtype=np.bool)

    def setUp(self):
        self.safe_set[:] = True

    def _check(self):
        reach = reachable_set(self.graph, [0])
        assert_equal(reach[:, 0], self.true)

    def test_all_safe(self):
        """Test reachable set if everything is safe"""
        self.true[:] = [1, 1, 1, 1, 0]
        self._check()

    def test_unsafe1(self):
        """Test safety aspect"""
        self.safe_set[1, 1] = False
        self.true[:] = [1, 1, 0, 0, 0]
        self._check()

    def test_unsafe2(self):
        """Test safety aspect"""
        self.safe_set[2, 2] = False
        self.true[:] = [1, 1, 1, 0, 0]
        self._check()

    def test_unsafe3(self):
        """Test safety aspect"""
        self.safe_set[2, 1] = False
        self.true[:] = [1, 1, 1, 1, 0]
        self._check()

    def test_unsafe4(self):
        """Test safety aspect"""
        self.safe_set[4, 1] = False
        self.true[:] = [1, 1, 1, 1, 0]
        self._check()

    def test_out(self):
        """Test writing the output"""
        self.safe_set[2, 2] = False
        self.true[:] = [1, 1, 1, 0, 0]
        out = np.zeros_like(self.safe_set)
        reachable_set(self.graph, [0], out=out)
        assert_equal(out[:, 0], self.true)

    def test_error(self):
        """Check error condition"""
        with assert_raises(AttributeError):
            reachable_set(self.graph, [])


class ReturnableSetTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ReturnableSetTest, self).__init__(*args, **kwargs)
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
                                   (4, 1)], action=1)
        self.graph.add_edge(2, 3, action=2)
        self.graph_rev = self.graph.reverse()

        self.safe_set = np.ones((self.graph.number_of_nodes(),
                                 max_out_degree(self.graph) + 1),
                                dtype=np.bool)
        link_graph_and_safe_set(self.graph, self.safe_set)
        self.true = np.zeros(self.safe_set.shape[0], dtype=np.bool)

    def setUp(self):
        self.safe_set[:] = True

    def _check(self):
        ret = returnable_set(self.graph, self.graph_rev, [0])
        assert_equal(ret[:, 0], self.true)

    def test_all_safe(self):
        """Test reachable set if everything is safe"""
        self.true[:] = [1, 1, 1, 0, 1]
        self._check()

    def test_unsafe1(self):
        """Test safety aspect"""
        self.safe_set[1, 1] = False
        self.true[:] = [1, 0, 1, 0, 0]
        self._check()

    def test_unsafe2(self):
        """Test safety aspect"""
        self.safe_set[2, 1] = False
        self.true[:] = [1, 0, 0, 0, 0]
        self._check()

    def test_unsafe3(self):
        """Test safety aspect"""
        self.safe_set[2, 2] = False
        self.true[:] = [1, 1, 1, 0, 1]
        self._check()

    def test_unsafe4(self):
        """Test safety aspect"""
        self.safe_set[4, 1] = False
        self.true[:] = [1, 1, 1, 0, 0]
        self._check()

    def test_out(self):
        """Test writing the output"""
        self.safe_set[1, 1] = False
        self.true[:] = [1, 0, 1, 0, 0]
        out = np.zeros_like(self.safe_set)
        returnable_set(self.graph, self.graph_rev, [0], out=out)
        assert_equal(out[:, 0], self.true)

    def test_error(self):
        """Check error condition"""
        with assert_raises(AttributeError):
            reachable_set(self.graph, [])


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


class TestTrueSafeSet(unittest.TestCase):

    def test_differences_safe(self):
        altitudes = np.array([[1, 2, 3],
                              [2, 3, 4]])
        safe = compute_true_safe_set((2, 3), altitudes.reshape(-1), -1)
        true_safe = np.array([[1, 1, 1, 1, 1, 1],
                              [1, 1, 0, 1, 1, 0],
                              [1, 1, 1, 0, 0, 0],
                              [0, 1, 1, 0, 1, 1],
                              [0, 0, 0, 1, 1, 1]],
                             dtype=np.bool).T

        assert_equal(safe, true_safe)

    def test_differences_unsafe(self):
        altitudes = np.array([[1, 0, 3],
                              [2, 3, 0]])
        safe = compute_true_safe_set((2, 3), altitudes.reshape(-1), -1)
        true_safe = np.array([[1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 1, 1, 0],
                              [1, 0, 1, 0, 0, 0],
                              [0, 1, 1, 0, 1, 0],
                              [0, 0, 0, 1, 1, 0]],
                             dtype=np.bool).T
        assert_equal(safe, true_safe)


if __name__ == '__main__':
    unittest.main()
