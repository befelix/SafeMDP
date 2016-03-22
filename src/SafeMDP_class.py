from __future__ import division, print_function
import numpy as np
import GPy
import time
import matplotlib.pyplot as plt


class SafeMDP(object):
    def __init__(self, gp, world_shape, step_size, beta, altitudes, h, S0, S_hat0, noise):

        self.gp = gp
    #    self.kernel = gp.kern
    #    self.likelihood = gp.likelihood
        self.altitudes = altitudes
        self.world_shape = world_shape
        self.step_size = step_size
        self.beta = beta
        self.noise = noise

        # Grids for the map
        self.ind, self.coord = self.grid()

        # Threshold
        self.h = h

        # Safe and expanders sets
        self.S = S0
        self.reach = np.empty_like(self.S, dtype=bool)
        self.ret = np.empty_like(self.S, dtype=bool)
        self.G = np.empty_like(self.S, dtype=bool)
        if np.isnan(S_hat0):
            self.S_hat = self.compute_S_hat0()
        else:
            self.S_hat = S_hat0

        # Target
        self.target_state = np.empty(2, dtype=int)
        self.target_action = np.empty(1, dtype=int)

        # Confidence intervals
        self.l = np.empty(self.S.shape, dtype=float)
        self.u = np.empty(self.S.shape, dtype=float)
        self.l[:] = -np.inf
        self.u[:] = np.inf
        self.l[self.S] = h

        # True sets
        self.true_S = self.compute_true_safe_set()
        self.true_S_hat = self.compute_true_S_hat()

    def grid(self):
        """
        Creates grids of coordinates and indices of state space

        Returns
        -------

        states_ind: np.array
                    (n*m) x 2 array containing the indices of the states
        states_coord: np.array
                      (n*m) x 2 array containing the coordinates of the states
        """
        # Create grid of indices
        n, m = self.world_shape
        xx, yy = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
        states_ind = np.vstack((xx.flatten(), yy.flatten())).T
        # Grid of coordinates (used to compute Gram matrix)
        step1, step2 = self.step_size
        xx, yy = np.meshgrid(np.linspace(0, (n-1) * step1, n), np.linspace(0, (m-1)*step2, m), indexing="ij")
        states_coord = np.vstack((xx.flatten(), yy.flatten())).T
        return states_ind, states_coord

    def update_confidence_interval(self):
        """
        Updates the lower and the upper bound of the confidence intervals using the
        posterior distribution over the gradients of the altitudes

        Returns
        -------
        l: np.array
           lower bound of the safety feature (mean - beta*std)
        u: np.array
           upper bound of the safety feature (mean - beta*std)
        """
        # Predict safety feature
        mu, s = self.gp.predict_jacobian(self.coord, full_cov=False)
        print(np.all(s >= 0))
        mu = np.squeeze(mu)

        # Initialize mean and variance over abstract MDP
        mu_abstract = np.zeros(self.S.shape)
        s_abstract = np.copy(mu_abstract)

        # Safety features for real states s
        mu_abstract[:, 0] = self.h
        s_abstract[:, 0] = 0

        # Safety feature for (s,a) pairs
        mu_abstract[:, 1] = -mu[:, 1]
        mu_abstract[:, 3] = mu[:, 1]
        s_abstract[:, 1] = s[:, 1]
        s_abstract[:, 3] = s[:, 1]
        mu_abstract[:, 2] = -mu[:, 0]
        mu_abstract[:, 4] = mu[:, 0]
        s_abstract[:, 2] = s[:, 0]
        s_abstract[:, 4] = s[:, 0]
        # Lower and upper bound of confidence interval
        self.l = mu_abstract - self.beta*np.sqrt(s_abstract)
        self.u = mu_abstract + self.beta*np.sqrt(s_abstract)

    def boolean_dynamics(self, bool_mat, action):
        """
        Given a boolean array over the state space, it shifts all the boolean
        values according to the dynamics of the system using the action
        provided as input. For example, if true entries of bool_mat indicate
        the safe states, boolean dynamics returns an array whose true entries
        indicate states that can be reached from the safe set with action = action

        Parameters
        ----------
        bool_mat: np.array
                  n_states x 1 array of booleans indicating which initial states satisfy a given property
        action: int
                action we want to compute the dynamics with

        Returns
        -------
        return: np.array
                n_states x 1 array of booleans. If the entry in boolean_mat in input is equal to
                true for a state s, the output will have then entry corresponding to f(s, action)
                set to true (f represents the dynamics of the system)
        """
        start = bool_mat.reshape(self.world_shape).copy()
        end = bool_mat.reshape(self.world_shape).copy()

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
        return np.reshape(end, (np.prod(self.world_shape)))

    def r_reach(self):
        """
        computes the union of the points in self.reach and the points that are
        reachable in one time step from self.reach and that are above safety
        threshold

        Returns
        -------
        changed: bool
                 Indicates whether self.reach and the newly computed set are different or not
        """

        # Initialize
        reachable_from_reach = np.zeros(self.S.shape, dtype=bool)

        # From s to (s,a) pair
        reachable_from_reach[self.reach[:, 0], 1:] = self.S[self.reach[:, 0], 1:]

        # From (s,a) to s
        for action in range(1, self.S.shape[1]):
            tmp = self.boolean_dynamics(self.reach[:, action], action)
            reachable_from_reach[:, 0] = np.logical_or(reachable_from_reach[:, 0], tmp)
        reachable_from_reach[:, 0] = np.logical_and(reachable_from_reach[:, 0], self.S[:, 0])
        reachable_from_reach = np.logical_or(reachable_from_reach, self.reach)
        changed = not np.all(self.reach == reachable_from_reach)
        self.reach[:] = reachable_from_reach
        return changed

    def boolean_inverse_dynamics(self, bool_mat, action):
        """
        Similar to boolean dynamics. The difference is that here the
        boolean_mat input indicates the arrival states that satisfy a given property
        and the function returns the initial states from which the arrival state
        can be reached applying the action input.

        Parameters
        ----------
        bool_mat: np.array
                  n_states x 1 array of booleans indicating which arrival states satisfy a given property
        action: int
                action we want to compute the inverse dynamics with

        Returns
        -------
        return: np.array
                n_states x 1 array of booleans. If the entry in the output is set to true
                for a state s, the input boolean_mat has the entry corresponding to f(s, action)
                equal to true (f represents the dynamics of the system)
        """
        start = bool_mat.reshape(self.world_shape).copy()
        end = bool_mat.reshape(self.world_shape).copy()

        if action == 3:  # moves right by one column
            start[:, 1:] = end[:, 0:-1]
            start[:, 0] = end[:, 0]

        elif action == 4:  # moves down by one row
            start[1:, :] = end[0:-1, :]
            start[0, :] = end[0, :]

        elif action == 1:  # moves left by one column
            start[:, 0:-1] = end[:, 1:]
            start[:, -1] = end[:, -1]

        elif action == 2:  # moves up by one row
            start[0:-1, :] = end[1:, :]
            start[-1, :] = end[-1, :]

        else:
            raise ValueError("Unknown action")
        return np.reshape(start, (np.prod(self.world_shape)))

    def r_ret(self):
        """
        computes the union of the points in self.ret and the points from which
        it is possible to recover to self.ret and that are above safety
        threshold

        Returns
        -------
        changed: bool
                 Indicates whether self.ret and the newly computed set are different or not
        """

        # Initialize
        recover_to_ret = np.zeros(self.S.shape, dtype=bool)

        # From s in S to (s,a) in ret
        recover_to_ret[self.S[:, 0], 0] = np.any(np.logical_and(self.S[self.S[:, 0], 1:], self.ret[self.S[:, 0], 1:]), axis=1)

        # From (s,a) in S to s in ret
        for action in range(1, self.S.shape[1]):
            tmp = self.boolean_inverse_dynamics(self.ret[:, 0], action)
            recover_to_ret[:, action] = np.logical_and(tmp, self.S[:, action])
        recover_to_ret = np.logical_or(recover_to_ret, self.ret)
        changed = not np.all(self.ret == recover_to_ret)
        self.ret[:] = recover_to_ret
        return changed

    def update_sets(self):
        """
        Updates the sets S, S_hat and G taking with the available observation
        """
        self.update_confidence_interval()
        self.S = self.l >= self.h

        # Actions that takes agent out of boundaries are assumed to be unsafe !!!!!!NEED TO CHANGE THIS IN REAL S TOO
        n, m = self.world_shape
        self.S[m-1:m*(n+1)-1:m, 1] = False
        self.S[(n-1)*m:n*m, 2] = False
        self.S[0:n*m:m, 3] = False
        self.S[0:m, 4] = False

        self.reach[:] = self.S_hat
        self.ret[:] = self.S_hat

        while self.r_reach():
            pass
        while self.r_ret():
            pass
        self.S_hat[:] = np.logical_or(self.S_hat, np.logical_and(self.reach, self.ret))

    def plot_S(self, S):
        """
        Plot the set of safe states

        Parameters
        ----------

        S: np.array(dtype=bool)
           n_states x (n_actions + 1) array of boolean values that indicates the safe set

        """
        for action in range(1):
            plt.figure(action)
            plt.imshow(np.reshape(S[:, action], self.world_shape).T, origin="lower", interpolation="nearest")
            plt.title("action " + str(action))
        plt.show()

    def add_obs(self, state_mat_ind, action):
        """
        Adds an observation of the given state-action pair. Observing the pair
        (s, a) means to add an observation of the altitude at s and an
        observation of the altitude at f(s, a)

        Parameters
        ----------
        state: np.array
               i,j indexing of the state of the target state action pair
        action: int
                action of the target state action pair
        """

        # Observation of previous state
        state_vec_ind = mat2vec(state_mat_ind, self.world_shape)
        obs_state = self.altitudes[state_vec_ind] + self.noise*np.random.randn(1)
        tmpX = np.vstack((self.gp.X, self.coord[state_vec_ind, :].reshape(1, 2)))
        tmpY = np.vstack((self.gp.Y, obs_state))

        # Observation of next state
        next_state_mat_ind = self.dynamics(state_mat_ind, action)
        next_state_vec_ind = mat2vec(next_state_mat_ind, self.world_shape)
        obs_next_state = self.altitudes[next_state_vec_ind] + self.noise*np.random.randn(1)

        # Update observations
        tmpX = np.vstack((tmpX, self.coord[next_state_vec_ind, :].reshape(1, 2)))
        tmpY = np.vstack((tmpY, obs_next_state))
        self.gp.set_XY(tmpX, tmpY)

    def target_sample(self):
        """
        Computes the next target (s, a) to sample (highest uncertainty within S_hat)
        """
        # Extract elements in S_hat
        non_z = np.nonzero(self.S_hat)

        # Compute uncertainty
        w = self.u[self.S_hat] - self.l[self.S_hat]

        # Find state with max uncertainty
        ind = np.argmax(w)
        state = non_z[0][ind]

        # Staore (s, a) pair
        self.target_state[:] = vec2mat(state, self.world_shape)
        self.target_action = non_z[1][ind]

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
                      of the state that results from applying action to the corresponding row of the input states
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

    def compute_true_safe_set(self):
        """
        Computes the safe set given a perfect knowledge of the map

        Returns
        -------
        true_safe: np.array
                   Boolean array n_states x (n_actions + 1).
        """

        # Initialize
        true_safe = np.empty_like(self.S, dtype=bool)

        # All true states are safe
        true_safe[:, 0] = True

        # Compute safe (s, a) pairs
        for action in range(1, self.S.shape[1]):
            next_mat_ind = self.dynamics(self.ind, action)
            next_vec_ind = mat2vec(next_mat_ind, self.world_shape)
            true_safe[:, action] = ((self.altitudes - self.altitudes[next_vec_ind])/self.step_size[0]) >= self.h
        return true_safe

    def compute_true_S_hat(self):
        """
        Computes the safe set with reachability and recovery properties given a perfect knowledge of the map

        Returns
        ------
        true_safe: np.array
                   Boolean array n_states x (n_actions + 1).
        """
        # Initialize
        true_S_hat = np.zeros_like(self.S, dtype=bool)
        self.reach[:] = self.S_hat
        self.ret[:] = self.S_hat

        # Substitute S with true S for r_reach and r_ret methods
        tmp = np.copy(self.S)
        self.S[:] = self.true_S

        # Reachable and recovery set
        while self.r_reach():
            pass
        while self.r_ret():
            pass

        # Points are either in S_hat or in the intersection of reachable and recovery sets
        true_S_hat[:] = np.logical_or(self.S_hat, np.logical_and(self.ret, self.reach))

        # Reset value of S
        self.S[:] = tmp
        return true_S_hat

    def compute_S_hat0(self):
        """
        Compute a random initial safe seed. WARNING:  at the moment actions for returning are not included

        Returns
        ------
        S_hat: np.array
               Boolean array n_states x (n_actions + 1).
        """
        # Initialize
        safe = np.zeros(self.S.shape[1] - 1, dtype=bool)
        S_hat = np.zeros_like(self.S, dtype=bool)

        # Loop until you find a valid initial seed
        while not np.any(safe):
            # Pick random state
            s = np.random.choice(self.ind.shape[0])

            # Compute next state for every action and check safety of (s, a) pair
            s_next = np.empty(self.S.shape[1] - 1, dtype=int)
            for action in range(1, self.S.shape[1]):

                s_next[action - 1] = mat2vec(self.dynamics(self.ind[s, :], action), self.world_shape).astype(int)
                alt = self.altitudes[s]
                alt_next = self.altitudes[s_next[action - 1]]

                if s != s_next[action - 1] and (alt-alt_next)/self.step_size[0] >= self.h:
                    safe[action - 1] = True
        # Set initial state, (s, a) pairs and arrival state as safe
        s_next = s_next[safe]
        S_hat[s, 0] = True
        S_hat[s_next, 0] = True
        S_hat[s, 1:] = safe
        return S_hat


def vec2mat(vec_ind, world_shape):
    """
    Converts from vector indexing to matrix indexing

    Parameters
    ----------
    vec_ind: np.array
             Each element contains the vector indexing of a state we want to do the convesrion for
    world_shape: shape
                 Tuple that contains the shape of the grid world n x m

    Returns
    -------
    return: np.array
            ith row contains the (x,y) coordinates of the ith element of the input vector vec_ind
    """
    n, m = world_shape
    row = np.floor(vec_ind/m)
    col = np.mod(vec_ind, m)
    return np.array([row, col]).astype(int)


def mat2vec(states_mat_ind, world_shape):
    """
    Converts from matrix indexing to vector indexing

    Parameters
    ----------
    states_mat_ind: np.array
                    Each row contains the (x,y) coordinates of each state we want to do the conversion for
    world_shape: shape
                 Tuple that contains the shape of the grid world n x m

    Returns
    -------
    vec_ind: np.array
             Each element contains the vector indexing of the point in the
             corresponding row of the input states_mat_ind
    """
    if states_mat_ind.ndim == 1:
        states_mat_ind = states_mat_ind.reshape(1, 2)
    m = world_shape[1]
    vec_ind = states_mat_ind[:, 1] + states_mat_ind[:, 0]*m
    return vec_ind.astype(int)


def draw_GP(kernel, world_shape, step_size):
    """
    Draws a sample from a Gaussian process distribution over a user specified grid

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
    n, m = world_shape
    step1, step2 = step_size
    xx, yy = np.meshgrid(np.linspace(0, (n-1) * step1, n), np.linspace(0, (m-1)*step2, m), indexing="ij")
    coord = np.vstack((xx.flatten(), yy.flatten())).T

    # Draw a sample from GP
    cov = kernel.K(coord)
    sample = np.random.multivariate_normal(np.zeros(coord.shape[0]), cov)
    return sample, coord

# test
#if __name__ == "main":
# kernel = GPy.kern.RBF(input_dim=2, lengthscale=(1., 1.), variance=1., ARD=True)
# lik = GPy.likelihoods.Gaussian(variance=1)
# lik.constrain_bounded(1e-5, 10000.)
# X = np.random.rand(200, 2)
# Y = (np.sin(X[:, 0]) + np.cos(X[:, 1]*2.)).reshape(200, 1) + 0.01*np.random.randn(200, 1)
# gp = GPy.core.GP(X, Y, kernel, lik)
# gp.optimize()
# print(gp)
#
# world_shape = (11, 11)
# step_size = (0.1, 0.1)
# beta = 3
# altitudes = np.random.rand(world_shape[0], world_shape[1]).reshape(np.prod(world_shape))
# noise = 0.1
# h = -10
# S0 = np.ones((np.prod(world_shape), 5), dtype=bool)
# S0[1, 2] = False
# S0[world_shape[1], 1] = False
# S0[world_shape[1]+2, 3] = False
# S0[world_shape[1]*2 + 1, 4] = False
#
# S0[5, 1] = False
# S0[5, 2] = False
# S0[5, 3] = False
# S0[5, 4] = False
#
# S_hat0 = np.zeros((np.prod(world_shape), 5), dtype=bool)
# S_hat0[0, 0] = True
# S_hat0 = np.nan
# x = SafeMDP(gp, world_shape, step_size, beta, altitudes, h, S0, S_hat0, noise)
#
#
# t = time.time()
# x.update_sets()
# print (str(time.time() - t) + "seconds elapsed")
# x.plot_S(x.reach)
# x.plot_S(x.ret)
# x.update_confidence_interval()
# x.target_sample()
# x.add_obs(x.target_state, x.target_action)

noise = 0.001
kernel = GPy.kern.RBF(input_dim=2, lengthscale=(2., 2.), variance=1., ARD=True)
lik = GPy.likelihoods.Gaussian(variance=noise ** 2)
lik.constrain_bounded(1e-6, 10000.)

world_shape = (40, 40)
step_size = (0.5, 0.5)
altitudes, coord = draw_GP(kernel, world_shape, step_size)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_trisurf(coord[:, 0], coord[:, 1], altitudes)
#plt.show()

beta = 3
ind = np.random.choice(range(coord.shape[0]), 200)
X = coord[ind, :]
Y = altitudes[ind].reshape(200, 1) + np.random.randn(200, 1)
gp = GPy.core.GP(X, Y, kernel, lik)
#gp.optimize()
S0 = np.zeros((np.prod(world_shape), 5), dtype=bool)
S0[:, 0] = True
S_hat0 = np.nan
h = -0.8

# Define SafeMDP object
x = SafeMDP(gp, world_shape, step_size, beta, altitudes, h, S0, S_hat0, noise)

# Insert samples from (s, a) in S_hat0 and remove samples used for optimizing hyperparameters
tmp = np.arange(x.ind.shape[0])
s_vec_ind = tmp[np.any(x.S_hat[:, 1:], axis=1)]
state = vec2mat(s_vec_ind, x.world_shape).T
tmp = np.arange(1, x.S.shape[1])
actions = tmp[x.S_hat[s_vec_ind, 1:].squeeze()]
for i in range(1):
    x.add_obs(state, np.random.choice(actions))
x.gp.set_XY(x.gp.X[200:, :], x.gp.Y[200:])

l_old = np.copy(x.l)

t = time.time()
for i in range(60):
#    x.plot_S(x.S_hat)
#    x.plot_S(x.S)
    x.update_sets()
    x.target_sample()
    x.add_obs(x.target_state, x.target_action)
    target_state_vec_ind = mat2vec(x.target_state, x.world_shape)
    next_state = x.dynamics(x.target_state, x.target_action)
    next_state_vec_ind = mat2vec(next_state, x.world_shape)
    target_state_coord = x.coord[target_state_vec_ind, :]
    next_state_coord = x.coord[next_state_vec_ind, :]
    w = x.u - x.l
    print("w of target (s, a) " + str(x.u[target_state_vec_ind, x.target_action] - x.l[target_state_vec_ind, x.target_action]))
    print("Max uncertainty of S_hat "+ str(np.max(w[x.S_hat])))
    print (x.target_state, x.target_action)
    print(i)

print (str(time.time() - t) + "seconds elapsed")
x.plot_S(x.S_hat)
x.plot_S(x.true_S_hat)
print(np.sum(np.logical_and(x.true_S_hat, np.logical_not(x.S_hat))))  # in true S_hat and not S_hat
print(np.sum(np.logical_and(x.S_hat, np.logical_not(x.true_S_hat))))
