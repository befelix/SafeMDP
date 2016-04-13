import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from src import DifferenceKernel
from src.SafeMDP_class import (reachable_set, returnable_set, SafeMDP,
                               link_graph_and_safe_set)


__all__ = ['compute_true_safe_set', 'compute_true_S_hat', 'compute_S_hat0', 'grid_world_graph',
           'grid', 'GridWorld', 'draw_gp_sample']


def compute_true_safe_set(world_shape, altitude, h):
    """
    Computes the safe set given a perfect knowledge of the map

    Parameters
    ----------
    world_shape: tuple
    altitude: np.array
        1-d vector with altitudes for each node
    h: float
        Safety threshold for height differences

    Returns
    -------
    true_safe: np.array
        Boolean array n_states x (n_actions + 1).
    """

    true_safe = np.zeros((world_shape[0] * world_shape[1], 5), dtype=np.bool)

    altitude_grid = altitude.reshape(world_shape)

    # Reshape so that first dimensions are actions, the rest is the grid world.
    safe_grid = true_safe.T.reshape((5,) + world_shape)

    # Height difference (next height - current height) --> positive if downhill
    up_diff = altitude_grid[:, :-1] - altitude_grid[:, 1:]
    right_diff = altitude_grid[:-1, :] - altitude_grid[1:, :]

    # State are always safe
    true_safe[:, 0] = True

    # Going in the opposite direction
    safe_grid[1, :, :-1] = up_diff >= h
    safe_grid[2, :-1, :] = right_diff >= h
    safe_grid[3, :, 1:] = -up_diff >= h
    safe_grid[4, 1:, :] = -right_diff >= h

    return true_safe


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


def grid_world_graph(world_size):
    """Create a graph that represents a grid world.

    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (up, right, down, left) in the x-y plane. The states are
    ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
    corresponds to a matrix where increasing the row index corresponds to the
    x direction in the graph, and increasing y index corresponds to the y
    direction.

    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)

    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()

    # action 1: go right
    graph.add_edges_from(zip(grid_nodes[:, :-1].reshape(-1),
                             grid_nodes[:, 1:].reshape(-1)),
                         action=1)

    # action 2: go down
    graph.add_edges_from(zip(grid_nodes[:-1, :].reshape(-1),
                             grid_nodes[1:, :].reshape(-1)),
                         action=2)

    # action 3: go left
    graph.add_edges_from(zip(grid_nodes[:, 1:].reshape(-1),
                             grid_nodes[:, :-1].reshape(-1)),
                         action=3)

    # action 4: go up
    graph.add_edges_from(zip(grid_nodes[1:, :].reshape(-1),
                             grid_nodes[:-1, :].reshape(-1)),
                         action=4)

    return graph


def compute_true_S_hat(graph, safe_set, initial_nodes, reverse_graph=None):
    """
    Compute the true safe set with reachability and returnability.

    Parameters
    ----------
    graph: nx.DiGraph
    safe_set: np.array
    initial_nodes: list of int
    reverse_graph: nx.DiGraph
        graph.reverse()

    Returns
    -------
    true_safe: np.array
        Boolean array n_states x (n_actions + 1).
    """
    graph = graph.copy()
    link_graph_and_safe_set(graph, safe_set)
    if reverse_graph is None:
        reverse_graph = graph.reverse()
    reach = reachable_set(graph, initial_nodes)
    ret = returnable_set(graph, reverse_graph, initial_nodes)
    ret &= reach
    return ret


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