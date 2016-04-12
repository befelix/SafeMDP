from __future__ import division, print_function

from utilities import *

import numpy as np
import GPy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, cityblock
import networkx as nx


__all__ = ['GridWorld', 'draw_gp_sample', 'manhattan_dist',
           'grid', 'states_to_nodes', 'nodes_to_states']


class SafeMDP(object):
    def __init__(self, graph, gp, S_hat0, h, L, beta=2):
        super(SafeMDP, self).__init__()
        # Scalar for gp confidence intervals
        self.beta = beta

        # Threshold
        self.h = h

        # Lipschitz constant
        self.L = L

        # GP model
        self.gp = gp

        self.graph = graph
        self.graph_reverse = self.graph.reverse()

        num_nodes = self.graph.number_of_nodes()
        num_edges = max_out_degree(graph)
        safe_set_size = (num_nodes, num_edges + 1)

        self.reach = np.empty(safe_set_size, dtype=np.bool)
        self.ret = np.empty(safe_set_size, dtype=np.bool)
        self.G = np.empty(safe_set_size, dtype=np.bool)

        self.S_hat = S_hat0
        self.S_hat0 = self.S_hat.copy()
        self.initial_nodes = self.S_hat0[:, 0].nonzero()[0].tolist()

    def compute_S_hat(self):
        """Compute the safely reachable set given the current safe_set."""
        self.reach[:] = False
        reachable_set(self.graph, self.initial_nodes, self.S, out=self.reach)

        self.S_hat[:] = False
        returnable_set(self.graph, self.graph_reverse, self.initial_nodes,
                       self.reach, out=self.S_hat)

    def add_gp_observations(self, x_new, y_new):
        """Add observations to the gp mode."""
        # Update GP with observations
        self.gp.set_XY(np.vstack((self.gp.X,
                                  x_new)),
                       np.vstack((self.gp.Y,
                                  y_new)))


class GridWorld(SafeMDP):
    """
    Grid world with Safe exploration

    Parameters
    ----------
    gp: GPy.core.GP
        Gaussian process that expresses our current belief over the safety
        feature
    world_shape: shape
                 Tuple that contains the shape of the grid world n x m
    step_size: tuple of floats
               Tuple that contains the step sizes along each direction to
               create a linearly spaced grid
    beta: float
          Scaling factor to determine the amplitude of the confidence
          intervals
    altitudes: np.array
               It contains the flattened n x m matrix where the altitudes
               of all the points in the map are stored
    h: float
       Safety threshold
    S0: np.array
        n_states x (n_actions + 1) array of booleans that indicates which
        states (first column) and which state-action pairs belong to the
        initial safe seed. Notice that, by convention we initialize all
        the states to be safe
    S_hat0: np.array or nan
        n_states x (n_actions + 1) array of booleans that indicates which
        states (first column) and which state-action pairs belong to the
        initial safe seed and satisfy recovery and reachability properties.
        If it is nan, such a boolean matrix is computed during
        initialization
    noise: float
           Standard deviation of the measurement noise
    L: float
       Lipschitz constant to compute expanders
    """
    def __init__(self, gp, world_shape, step_size, beta, altitudes, h, S0,
                 S_hat0, L):

        # Safe set
        self.S = S0
        graph = grid_world_graph(world_shape)
        link_graph_and_safe_set(graph, self.S)
        super(GridWorld, self).__init__(graph, gp, S_hat0, h, L, beta=2)

        self.altitudes = altitudes
        self.world_shape = world_shape
        self.step_size = step_size

        # Grids for the map
        self.coord = grid(self.world_shape, self.step_size)

        # Distances
        self.distance_matrix = cdist(self.coord, self.coord)

        # Confidence intervals
        self.l = np.empty(self.S.shape, dtype=float)
        self.u = np.empty(self.S.shape, dtype=float)
        self.l[:] = -np.inf
        self.u[:] = np.inf
        self.l[self.S] = h

        # Prediction with difference of altitudes
        states_ind = np.arange(np.prod(self.world_shape))
        states_grid = states_ind.reshape(world_shape)

        self._prev_up = states_grid[:, :-1].flatten()
        self._next_up = states_grid[:, 1:].flatten()
        self._prev_right = states_grid[:-1, :].flatten()
        self._next_right = states_grid[1:, :].flatten()

        self._mat_up = np.hstack((self.coord[self._prev_up, :],
                                  self.coord[self._next_up, :]))
        self._mat_right = np.hstack((self.coord[self._prev_right, :],
                                     self.coord[self._next_right, :]))

    def update_confidence_interval(self, jacobian=False):
        """
        Updates the lower and the upper bound of the confidence intervals
        using then posterior distribution over the gradients of the altitudes

        Returns
        -------
        l: np.array
            lower bound of the safety feature (mean - beta*std)
        u: np.array
            upper bound of the safety feature (mean - beta*std)
        """
        if jacobian:
            # Predict safety feature
            mu, s = self.gp.predict_jacobian(self.coord, full_cov=False)
            mu = np.squeeze(mu)

            # Confidence interval
            s = self.beta * np.sqrt(s)

            # State are always safe
            self.l[:, 0] = self.u[:, 0] = self.h

            # Update safety feature
            self.l[:, [1, 2]] = -mu[:, ::-1] - s[:, ::-1]
            self.l[:, [3, 4]] = mu[:, ::-1] - s[:, ::-1]

            self.u[:, [1, 2]] = -mu[:, ::-1] + s[:, ::-1]
            self.u[:, [3, 4]] = mu[:, ::-1] + s[:, ::-1]
        else:
            # Initialize to unsafe
            self.l[:] = self.u[:] = self.h - 1

            # States are always safe
            self.l[:, 0] = self.u[:, 0] = self.h

            # Actions up and down
            mu_up, s_up = self.gp.predict(self._mat_up,
                                          kern=DifferenceKernel(self.gp.kern),
                                          full_cov=False)
            s_up = self.beta * np.sqrt(s_up)

            self.l[self._prev_up, 1, None] = -mu_up - s_up
            self.u[self._prev_up, 1, None] = -mu_up + s_up

            self.l[self._next_up, 3, None] = mu_up - s_up
            self.u[self._next_up, 3, None] = mu_up + s_up

            # Actions left and right
            mu_right, s_right = self.gp.predict(self._mat_right,
                                                kern=DifferenceKernel(
                                                    self.gp.kern), full_cov=False)
            s_right = self.beta * np.sqrt(s_right)
            self.l[self._prev_right, 2, None] = -mu_right - s_right
            self.u[self._prev_right, 2, None] = -mu_right + s_right

            self.l[self._next_right, 4, None] = mu_right - s_right
            self.u[self._next_right, 4, None] = mu_right + s_right

    def compute_expanders(self):
        """Compute the expanders based on the current estimate of S_hat."""
        self.G[:] = False

        for action in range(1, self.S_hat.shape[1]):

            # action-specific safe set
            s_hat = self.S_hat[:, action]

            # Extract distance from safe points to non safe ones
            distance = self.distance_matrix[np.ix_(s_hat, ~self.S[:, action])]

            # Update expanders for this particular action
            self.G[s_hat, action] = np.any(
                self.u[s_hat, action, None] - self.L * distance >= self.h,
                axis=1)

    def update_sets(self):
        """
        Update the sets S, S_hat and G taking with the available observation
        """
        self.update_confidence_interval()
        self.S[:] = self.l >= self.h

        self.compute_S_hat()

        self.compute_expanders()

    def plot_S(self, safe_set, action=0):
        """
        Plot the set of safe states

        Parameters
        ----------
        safe_set: np.array(dtype=bool)
            n_states x (n_actions + 1) array of boolean values that indicates
            the safe set
        action: int
            The action for which we want to plot the safe set.
        """
        plt.figure(action)
        plt.imshow(np.reshape(safe_set[:, action], self.world_shape).T,
                   origin='lower', interpolation='nearest', vmin=0, vmax=1)
        plt.title('action {0}'.format(action))
        plt.show()

    def add_observation(self, node, action):
        """
        Add an observation of the given state-action pair.

        Observing the pair (s, a) means adding an observation of the altitude
        at s and an observation of the altitude at f(s, a)

        Parameters
        ----------
        node: int
            Node index
        action: int
            Action index
        """
        # Observation of next state
        for _, next_node, data in self.graph.edges_iter(node, data=True):
            if data['action'] == action:
                break

        self.add_gp_observations(self.coord[[node, next_node], :],
                                 self.altitudes[[node, next_node], None])

    def target_sample(self):
        """
        Compute the next target (s, a) to sample (highest uncertainty within
        G or S_hat)

        Returns
        -------
        node: int
            The next node to sample
        action: int
            The next action to sample
        """
        if np.any(self.G):
            # Extract elements in G
            expander_id = np.nonzero(self.G)

            # Compute uncertainty
            w = self.u[self.G] - self.l[self.G]

            # Find   max uncertainty
            max_id = np.argmax(w)

        else:
            print('No expanders, using most uncertain element in S_hat'
                  'instead.')

            # Extract elements in S_hat
            expander_id = np.nonzero(self.S_hat)

            # Compute uncertainty
            w = self.u[self.S_hat] - self.l[self.S_hat]

            # Find   max uncertainty
            max_id = np.argmax(w)

        return expander_id[0][max_id], expander_id[1][max_id]


def states_to_nodes(states, step_size):
    """Convert physical states to node numbers.

    Parameters
    ----------
    states: np.array
        States with physical coordinates
    step_size: np.array
        The step size of the grid world

    Returns
    -------
    nodes: np.array
        The node indices corresponding to the states
    """
    states = np.asanyarray(states)
    step_size = np.asanyarray(step_size)
    return np.rint(states / step_size).astype(np.int)


def nodes_to_states(nodes, step_size):
    """Convert node numbers to physical states.

    Parameters
    ----------
    nodes: np.array
        Node indices of the grid world
    step_size: np.array
        Teh step size of the grid world

    Returns
    -------
    states: np.array
        The states in physical coordinates
    """
    nodes = np.asanyarray(nodes)
    step_size = np.asanyarray(step_size)
    return nodes * step_size


def grid(world_shape, step_size):
    """
    Creates grids of coordinates and indices of state space

    Parameters
    ----------
    world_shape: tuple
        Size of the grid world (rows, columns)
    step_size: tuple
        Phyiscal step size in the grid world

    Returns
    -------
    states_ind: np.array
        (n*m) x 2 array containing the indices of the states
    states_coord: np.array
        (n*m) x 2 array containing the coordinates of the states
    """
    # Create grid of indices
    n, m = world_shape
    xx, yy = np.meshgrid(np.arange(n),
                         np.arange(m),
                         indexing='ij')
    states_ind = np.vstack((xx.flatten(), yy.flatten())).T
    return nodes_to_states(states_ind, step_size)


def draw_gp_sample(kernel, world_shape, step_size):
    """
    Draws a sample from a Gaussian process distribution over a user
    specified grid

    Parameters
    ----------
    kernel: GPy kernel
        Defines the GP we draw a sample from
    world_shape: tuple
        Shape of the grid we use for sampling
    step_size: tuple
        Step size along any axis to find linearly spaced points
    """
    # Compute linearly spaced grid
    coord = grid(world_shape, step_size)

    # Draw a sample from GP
    cov = kernel.K(coord) + np.eye(coord.shape[0]) * 1e-10
    sample = np.random.multivariate_normal(np.zeros(coord.shape[0]), cov)
    return sample, coord


def manhattan_dist(a, b):
    return cityblock(a, b)


def link_graph_and_safe_set(graph, safe_set):
    """Link the safe set to the graph model.

    Parameters
    ----------
    graph: nx.DiGraph()
    safe_set: np.array
        Safe set. For each node the edge (i, j) under action (a) is linked to
        safe_set[i, a]
    """
    for node, next_node in graph.edges_iter():
        edge = graph[node][next_node]
        edge['safe'] = safe_set[node:node+1, edge['action']]
