from __future__ import division

import numpy as np
import networkx as nx


__all__ = ['DifferenceKernel', 'compute_S_hat0', 'reverse_action',
           'dynamics_vec_ind', 'reachable_set', 'returnable_set',
           'max_out_degree', 'grid_world_graph']


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


def max_out_degree(graph):
    """Compute the maximum out_degree of a graph

    Parameters
    ----------
    graph: nx.DiGraph

    Returns
    -------
    max_out_degree: int
        The maximum out_degree of the graph
    """
    def degree_generator(graph):
        for _, degree in graph.out_degree_iter():
            yield degree
    return max(degree_generator(graph))


def reachable_set(graph, initial_nodes, safe, out=None):
    """
    Compute the safe, reachable set of a graph

    Parameters
    ----------
    graph: nx.DiGraph
        Directed graph. Each edge must have associated action metadata,
        which specifies the action that this edge corresponds to.
    initial_nodes: list
        List of the initial, safe nodes that are used as a starting point to
        compute the reachable set.
    safe: np.array
        Boolean array which on element (i,j) indicates whether taking
        action j at node i is safe.
        i=0 is interpreted as the node without taking an action.
    out: np.array
        The array to write the results to. Is assumed to be False everywhere
        except at the initial nodes

    Returns
    -------
    reachable_set: np.array
        Boolean array that indicates whether a node belongs to the reachable
        set.
    """

    if not initial_nodes:
        raise AttributeError('Set of initial nodes needs to be non-empty.')

    if out is None:
        visited = np.zeros((graph.number_of_nodes(),
                            max_out_degree(graph) + 1),
                           dtype=np.bool)
    else:
        visited = out

    # All nodes in the initial set are visited
    visited[initial_nodes, 0] = True

    stack = list(initial_nodes)

    # TODO: rather than checking if things are safe, specify a safe subgraph?
    while stack:
        node = stack.pop(0)
        # iterate over edges going away from node
        for _, next_node, data in graph.edges_iter(node, data=True):
            action = data['action']
            if (not visited[node, action] and
                    safe[node, action] and
                    safe[next_node, 0]):
                visited[node, action] = True
                if not visited[next_node, 0]:
                    stack.append(next_node)
                    visited[next_node, 0] = True
    if out is None:
        return visited


def returnable_set(graph, reverse_graph, initial_nodes, safe, out=None):
    """
    Compute the safe, returnable set of a graph

    Parameters
    ----------
    graph: nx.DiGraph
        Directed graph. Each edge must have associated action metadata,
        which specifies the action that this edge corresponds to.
    reverse_graph: nx.DiGraph
        The reversed directed graph, `graph.reverse()`
    initial_nodes: list
        List of the initial, safe nodes that are used as a starting point to
        compute the returnable set.
    safe: np.array
        Boolean array which on element (i,j) indicates whether taking
        action j at node i is safe.
        i=0 is interpreted as the node without taking an action.
    out: np.array
        The array to write the results to. Is assumed to be False everywhere
        except at the initial nodes

    Returns
    -------
    returnable_set: np.array
        Boolean array that indicates whether a node belongs to the returnable
        set.
    """

    if not initial_nodes:
        raise AttributeError('Set of initial nodes needs to be non-empty.')

    if out is None:
        visited = np.zeros((graph.number_of_nodes(),
                            max_out_degree(graph) + 1),
                           dtype=np.bool)
    else:
        visited = out

    # All nodes in the initial set are visited
    visited[initial_nodes, 0] = True

    stack = list(initial_nodes)

    # TODO: rather than checking if things are safe, specify a safe subgraph?
    while stack:
        node = stack.pop(0)
        # iterate over edges going into node
        for _, prev_node in reverse_graph.edges_iter(node):
            action = graph.get_edge_data(prev_node, node)['action']
            if (not visited[prev_node, action] and
                    safe[prev_node, action] and
                    safe[prev_node, 0]):
                visited[prev_node, action] = True
                if not visited[prev_node, 0]:
                    stack.append(prev_node)
                    visited[prev_node, 0] = True
    if out is None:
        return visited


def grid_world_graph(world_size):
    """Create a graph that represents a grid world.

    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (right, down, left, up) on the grid. The states are ordered so
    that `np.arange(np.prod(world_size)).reshape(world_size)` corresponds to
    a matrix where each

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


def compute_true_safe_set(self, world_size, altitudes):
    """
    Compute the safe set given a perfect knowledge of the map

    Parameters
    ----------
    world_size: tuple


    Returns
    -------
    true_safe: np.array
        Boolean array n_states x (n_actions + 1).
    """

    # Initialize
    true_safe = np.empty_like(self.S, dtype=bool)

    # All true states are safe
    true_safe[:, 0] = True

    # TODO: This should be a function that takes mean and variance (in
    # TODO: this case 0) and returns the safety matrix. Then we can use the
    # TODO: same function in `update_confidence_intervals(.)`
    # Compute safe (s, a) pairs
    for action in range(1, self.S.shape[1]):
        next_mat_ind = self.dynamics(self.grid_index, action)
        next_vec_ind = mat2vec(next_mat_ind, self.world_shape)
        true_safe[:, action] = ((self.altitudes -
                                 self.altitudes[next_vec_ind]) /
                                self.step_size[0]) >= self.h

    # (s, a) pairs that lead out of boundaries are not safe
    n, m = self.world_shape
    true_safe[m - 1:m * (n + 1) - 1:m, 1] = False
    true_safe[(n - 1) * m:n * m, 2] = False
    true_safe[0:n * m:m, 3] = False
    true_safe[0:m, 4] = False
    return true_safe


def dynamics(self, states, action):
    """
    Dynamics of the system
    The function computes the one time step dynamic evolution of the system
    for any number of initial state and for one given action

    Parameters
    ----------
    states: np.array
        Two dimensional array. Each row contains the (x,y) coordinates of
        the starting points we want to compute the evolution for
    action: int
        Control action (1 = up, 2 = right, 3 = down, 4 = left)

    Returns
    -------
    next_states: np.array
        Two dimensional array. Each row contains the (x,y) coordinates
        of the state that results from applying action to the corresponding
        row of the input states
    """
    n, m = self.world_shape
    if states.ndim == 1:
        states = states.reshape(1, 2)
    next_states = np.copy(states)
    if action == 1:
        next_states[:, 1] += 1
        next_states[next_states[:, 1] > m - 1, 1] = m - 1
    elif action == 2:
        next_states[:, 0] += 1
        next_states[next_states[:, 0] > n - 1, 0] = n - 1
    elif action == 3:
        next_states[:, 1] -= 1
        next_states[next_states[:, 1] < 0, 1] = 0
    elif action == 4:
        next_states[:, 0] -= 1
        next_states[next_states[:, 0] < 0, 0] = 0
    else:
        raise ValueError("Unknown action")
    return next_states


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
    safe_grid = true_safe.T.reshape((5,) + world_shape)

    right_diff = altitude_grid[:, :-1] - altitude_grid[:, 1:]
    down_diff = altitude_grid[:-1, :] - altitude_grid[1:, :]

    true_safe[:, 0] = True
    safe_grid[1, :, :-1] = right_diff >= h
    safe_grid[2, :-1, :] = down_diff >= h
    safe_grid[3, :, 1:] = -right_diff >= h
    safe_grid[4, 1:, :] = -down_diff >= h

    return true_safe


def compute_true_S_hat(self):
    """
    Computes the safe set with reachability and recovery properties
    given a perfect knowledge of the map

    Returns
    -------
    true_safe: np.array
        Boolean array n_states x (n_actions + 1).
    """
    # Initialize
    true_S_hat = np.zeros_like(self.S, dtype=bool)
    self.reach[:] = self.S_hat
    self.ret[:] = self.S_hat

    # Substitute S with true S for update_reachable_set and update_return_set methods
    tmp = np.copy(self.S)
    self.S[:] = self.true_S

    # Reachable and recovery set
    while self.update_reachable_set():
        pass
    while self.update_return_set():
        pass

    # Points are either in S_hat or in the intersection of reachable and
    #  recovery sets
    true_S_hat[:] = np.logical_or(self.S_hat,
                                  np.logical_and(self.ret, self.reach))

    # Reset value of S
    self.S[:] = tmp
    return true_S_hat
