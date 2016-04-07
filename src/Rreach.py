from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt


def dynamics(states, action, world_shape):
    """
    Dynamics of the system

    The function computes the one time step dynamic evolution of the system
    for any number of initial state and for one given action

    Parameters
    ----------
    states: np.array
        wo dimensional array. Each row contains the (x,y) coordinates of the
        starting points we want to compute the evolution for
    action: int
        Control action (1 = up, 2 = right, 3 = down, 4 = left)
    world_shape: tuple
        Tuple that contains the shape of the grid world n x m

    Returns
    -------
    next_states: np.array
        Two dimensional array. Each row contains the (x,y) coordinates of the
        state that results from applying action to the corresponding row of the
        input states
    """
    n, m = world_shape
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


def mat2vec(states_mat_ind, world_shape):
    """
    Converts from matrix indexing to vector indexing

    Parameters
    ----------
    states_mat_ind: np.array
        Each row contains the (x,y) coordinates of each state we want to do the
        conversion for
    world_shape: shape
        Tuple that contains the shape of the grid world n x m

    Returns
    -------
    vec_ind: np.array
        Each element contains the vector indexing of the point in the
        corresponding row of the input states_mat_ind
    """
    m = world_shape[1]
    vec_ind = states_mat_ind[:, 1] + states_mat_ind[:, 0]*m
    return vec_ind.astype(int)


def boolean_dynamics(bool_mat, action, world_shape):
    start = bool_mat.reshape(world_shape).copy()
    end = bool_mat.reshape(world_shape).copy()
    if action == 1:  # moves right by one column
        end[:, 1:] = start[:, 0:-1]
        end[:, -1] = np.logical_or(end[:, -1], start[:, -1])
        end[:, 0] = False
    elif action == 2:  # moves down by one row
        end[1:, :] = start[0:-1, :]
        end[-1, :] = np.logical_or(end[-1, :], start[-1, :])
        end[0, :] = False
    elif action == 3:  # moves left by one column
        end[:, 0:-1] = start[:, 1:]
        end[:, 0] = np.logical_or(end[:, 0], start[:, 0])
        end[:, -1] = False
    elif action == 4:  # moves up by one row
        end[0:-1, :] = start[1:, :]
        end[0, :] = np.logical_or(end[0, :], start[0, :])
        end[-1, :] = False
    else:
        raise ValueError("Unknown action")
    return np.reshape(end, (np.prod(world_shape)))


def r_reach(S_hat, S, world_shape):
    """
    Implements the intersection of R^(reach)(S_hat) and S

    Parameters
    ----------
    S_hat: np.array(dtype=bool)
        n_states x n_action array. Safe set with ergodicity properties from
        previous iteration.
    S: np.array(dtype=bool)
        n_states x n_action array. Set of points above the safety threshold at
        current iteration
    world_shape: shape
        Tuple that contains the shape of the grid world n x m

    Returns
    -------
    return: np.array(dtype=bool)
        n_states x n_action array. Union of S_hat and the set of points
        reachable from S_hat that is above safety threshold
    """

    # Initialize
    reachable = np.zeros(S.shape, dtype=bool)

    # From s to (s,a) pair
    reachable[S_hat[:, 0], 1:] = S[S_hat[:, 0], 1:]

    # From (s,a) to s
    for action in range(1, S.shape[1]):
        tmp = boolean_dynamics(S_hat[:, action], action, world_shape)
        reachable[:, 0] = np.logical_or(reachable[:, 0], tmp)
    reachable[:, 0] = np.logical_and(reachable[:, 0], S[:, 0])

    return np.logical_or(reachable, S_hat)


def plot_S(S, world_shape):
    """
    Plot the set of safe states

    Parameters
    ----------
    S: np.array(dtype=bool)
        n_states x (n_actions + 1) array of boolean values that indicates the
        safe set
    world_shape: shape
        Tuple that contains the shape of the grid world n x m

    Returns
    -------
    none
    """
    for action in range(1): # np.arange(S.shape[1]):
        plt.figure(action)
        plt.imshow(np.reshape(S[:, action], world_shape),
                   origin="lower", interpolation="nearest")
        plt.title("action " + str(action))
    plt.show()


if __name__ == "__main__":
    # Test the code
    n = 4
    world_shape = (6, 4)
    x, y = np.meshgrid(np.arange(world_shape[0]), np.arange(world_shape[1]),
                       indexing="ij")
    states_ind = np.hstack((x.reshape(x.size, 1), y.reshape(y.size, 1)))
    S = np.ones((world_shape[0] * world_shape[1], 5), dtype=bool)

    # Make unsafe all the actions that lead to the point [1, 1]
    S[1, 2] = False
    S[4, 1] = False
    S[9, 4] = False
    S[6, 3] = False
    print(S)
    S_hat = np.zeros((world_shape[0] * world_shape[1], 5), dtype=bool)
    S_hat[0, 0] = True
    print(S_hat)
    for i in range(10):
        S_hat = r_reach(S_hat, S, world_shape)
        plot_S(S_hat, world_shape)
        print(S_hat)
        print(i)
        #input("Press something")
    plt.show()
