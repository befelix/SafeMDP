from __future__ import division

import numpy as np


__all__ = ['DifferenceKernel']


class DifferenceKernel(object):
    """
    A fake kernel that can be used to predict differences two function values.

    Given a gp based on measurements, we aim to predict the difference between
    the function values at two different test points, X1 and X2; that is, we
    want to obtain mean and variance of f(X1) - f(X1). Using this fake
    kernel, this can be achieved with
    `mean, var = gp.predict(np.vstack((X1, X2)), kern=DiffKernel(gp.kern))`

    Parameters
    ----------
    kernel: GPy.kern.*
        The kernel used by the GP
    """

    def __init__(self, kernel):
        self.kern = kernel

    def K(self, x1, x2=None):
        """Equivalent of kern.K

        If only x1 is passed then it is assumed to contain the data for both
        whose differences we are computing. Otherwise, x2 will contain these
        extended states (see PosteriorExact._raw_predict in
        GPy/inference/latent_function_inference0/posterior.py)

        Parameters
        ----------
        x1: np.array
        x2: np.array
        """
        dim = self.kern.input_dim
        if x2 is None:
            x10 = x1[:, :dim]
            x11 = x1[:, dim:]
            return (self.kern.K(x10) + self.kern.K(x11) -
                    2 * self.kern.K(x10, x11))
        else:
            x20 = x2[:, :dim]
            x21 = x2[:, dim:]
            return self.kern.K(x1, x20) - self.kern.K(x1, x21)

    def Kdiag(self, x):
        """Equivalent of kern.Kdiag for the difference prediction.

        Parameters
        ----------
        x: np.array
        """
        dim = self.kern.input_dim
        x0 = x[:, :dim]
        x1 = x[:, dim:]
        return (self.kern.Kdiag(x0) + self.kern.Kdiag(x1) -
                2 * np.diag(self.kern.K(x0, x1)))
