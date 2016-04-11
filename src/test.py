from __future__ import division, print_function, absolute_import

import unittest
import GPy
import numpy as np
import networkx as nx
from numpy.testing import *

from .utilities import DifferenceKernel, max_out_degree


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


if __name__ == '__main__':
    unittest.main()
