import numpy as np
import matplotlib.pyplot as plt


def dynamics(states, action, n):

    """
    Computes the discrete dynamic evolution of the system for any number of states and a given action
    :param states: m x 2 array containing the states we wish to compute the evolution for. Each row contains the x and y
    coordinates of each state
    :param action: control action performed by the agent (1 = up, 2 = right, 3 = down, 4 = left)
    :param n: dimension of the n x n grid that represents the world
    :return: m x 2 containing the dynamic evolution of all the states given as input when the action is applied

    """
    next_states = np.copy(states)
    if action == 1:
        next_states[:, 1] += 1
        next_states[next_states[:, 1] > n - 1, 1] = n - 1
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
        print("Unknown action")
    return next_states


def mat2vec(states_mat_ind, n):
    """
    Converts from matrix indexing of the world to vector indexing
    :param states_mat_ind: m x 2 matrix. FEach row contains the x and y coordinates of each state we want to do the conversion for
    :param n: size of the n x n grid world
    :return: m x 1 array containing the vector index of the points given in input
    """
    vec_ind = states_mat_ind[:, 1] + states_mat_ind[:, 0]*n
    return vec_ind.astype(int)


def r_reach(S_hat, S, states_ind):
    """
    Implements the intersection of R^(reach)(S_hat) and S
    :param S_hat: Safe set with ergodicity properties from previous iteration
    :param S: Set of points above the safety threshold at current iteration
    :param states_ind: State space given with matrix indexing
    :return: Union of S_hat and the set of points reachable from S_hat that is above safety threshold
    """

    n = np.sqrt(S.shape[0])
    # Initialize
    reachable = np.zeros(S.shape, dtype=bool)

    # From s to (s,a) pair
    reachable[S_hat[:, 0], 1:] = S[S_hat[:, 0], 1:]

    # From (s,a) to s
    for action in range(1, S.shape[1]):
        states = states_ind[S_hat[:, action]]
        if states.size > 0:
            next_states_mat_ind = dynamics(states, action, n)
            next_states_vec_ind = mat2vec(next_states_mat_ind, n)
            reachable[next_states_vec_ind, 0] = S[next_states_vec_ind, 0]
    return np.logical_or(reachable, S_hat)


def plot_S(S):
    """
    Plot the set of safe states
    :param S: n_states x (n_actions + 1) matrix of boolean values that indicates the safe set
    :return: none
    """
    n = np.sqrt(S.shape[0])
    for action in np.arange(1): # np.arange(S.shape[1]):
        plt.figure(action)
        plt.imshow(np.reshape(S[:, action], (n, n)), origin="lower")
        plt.title("action " + str(action))
    plt.draw()

# Test the code
n = 4
x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
states_ind = np.hstack((x.reshape(y.size, 1), y.reshape(x.size, 1)))
S = np.ones((n ** 2, 5), dtype=bool)

# Make unsafe all the actions that lead to the point [1, 1]
S[1, 2] = False
S[4, 1] = False
S[9, 4] = False
S[6, 3] = False

S_hat = np.zeros((n ** 2, 5), dtype=bool)
S_hat[0, 0] = True
for i in np.arange(10):
    S_hat = r_reach(S_hat, S, states_ind)
    plot_S(S_hat)
    print(S_hat)
    #input("Press something")
plt.show()
