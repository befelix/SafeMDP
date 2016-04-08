from __future__ import division

import numpy as np


__all__ = ['DifferenceKernel', 'compute_S_hat0', 'reverse_action',
           'dynamics_vec_ind']


class DifferenceKernel(object):
    """
    A fake kernel that can be used to predict differences two function values.

    Given a gp based on measurements, we aim to predict the difference between
    the function values at two different test points, X1 and X2; that is, we
    want to obtain mean and variance of f(X1) - f(X2). Using this fake
    kernel, this can be achieved with
    `mean, var = gp.predict(np.hstack((X1, X2)), kern=DiffKernel(gp.kern))`

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
                    self.kern.K(x10, x11) - self.kern.K(x11, x10))
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


def compute_S_hat0(s, world_shape, n_actions, altitudes, step_size, h):
    """
    Compute a valid initial safe seed.

    Parameters
    ---------
    s: int or nan
        Vector index of the state where we start computing the safe seed
       from. If it is equal to nan, a state is chosen at random
    world_shape: tuple
        Size of the grid world (rows, columns)
    n_actions: int
        Number of actions available to the agent
    altitudes: np.array
        It contains the flattened n x m matrix where the altitudes of all
        the points in the map are stored
    step_size: tuple
        step sizes along each direction to create a linearly spaced grid
    h: float
        Safety threshold

    Returns
    ------
    S_hat: np.array
        Boolean array n_states x (n_actions + 1).
    """
    # Initialize
    n, m = world_shape
    n_states = n * m
    S_hat = np.zeros((n_states, n_actions + 1), dtype=bool)

    # In case an initial state is given
    if not np.isnan(s):
        S_hat[s, 0] = True
        valid_initial_seed = False
        altitude_prev = altitudes[s]
        if not isinstance(s, np.ndarray):
            s = np.array([s])

        # Loop through actions
        for action in range(1, n_actions + 1):

            # Compute next state to check steepness
            next_vec_ind = dynamics_vec_ind(s, action, world_shape)
            altitude_next = altitudes[next_vec_ind]

            if s != next_vec_ind and -np.abs(altitude_prev - altitude_next) / \
                    step_size[0] >= h:
                S_hat[s, action] = True
                S_hat[next_vec_ind, 0] = True
                S_hat[next_vec_ind, reverse_action(action)] = True
                valid_initial_seed = True

        if valid_initial_seed:
            return S_hat
        else:
            print ("No valid initial seed starting from this state")
            S_hat[s, 0] = False
            return S_hat

    # If an explicit initial state is not given
    else:
        while np.all(np.logical_not(S_hat)):
            initial_state = np.random.choice(n_states)
            S_hat = compute_S_hat0(initial_state, world_shape, n_actions,
                                   altitudes, step_size, h)
        return S_hat


def reverse_action(action):
    # Computes the action that is the opposite of the one given as input

    rev_a = np.mod(action + 2, 4)
    if rev_a == 0:
        rev_a = 4
    return rev_a


def dynamics_vec_ind(states_vec_ind, action, world_shape):
    """
    Dynamic evolution of the system defined in vector representation of
    the states

    Parameters
    ----------
    states_vec_ind: np.array
        Contains all the vector indexes of the states we want to compute
        the dynamic evolution for
    action: int
        action performed by the agent

    Returns
    -------
    next_states_vec_ind: np.array
        vector index of states resulting from applying the action given
        as input to the array of starting points given as input
    """
    n, m = world_shape
    next_states_vec_ind = np.copy(states_vec_ind)
    if action == 1:
        next_states_vec_ind[:] = states_vec_ind + 1
        condition = np.mod(next_states_vec_ind, m) == 0
        next_states_vec_ind[condition] = states_vec_ind[condition]
    elif action == 2:
        next_states_vec_ind[:] = states_vec_ind + m
        condition = next_states_vec_ind >= m * n
        next_states_vec_ind[condition] = states_vec_ind[condition]
    elif action == 3:
        next_states_vec_ind[:] = states_vec_ind - 1
        condition = np.mod(states_vec_ind, m) == 0
        next_states_vec_ind[condition] = states_vec_ind[condition]
    elif action == 4:
        next_states_vec_ind[:] = states_vec_ind - m
        condition = next_states_vec_ind <= -1
        next_states_vec_ind[condition] = states_vec_ind[condition]
    else:
        raise ValueError("Unknown action")
    return next_states_vec_ind
