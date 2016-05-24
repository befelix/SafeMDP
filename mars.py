from __future__ import division, print_function

import sys
import os
import time

import GPy
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import networkx as nx
from scipy import interpolate

from src.grid_world import *
from plot_utilities import *
from mars_utilities import *

print(sys.version)


def check_shortest_path(source, next_sample, G, h, altitudes):
    # Extract safe graph
    safe_edges = [edge for edge in G.edges_iter(data=True) if edge[2]['safe']]
    graph_safe = nx.DiGraph(safe_edges)

    # for node, next_node in graph_safe.out_edges(nbunch=source, data=False):
    #     graph_safe.add_edge(node, next_node)

    # Compute shortest path
    target = next_sample[0]
    action = next_sample[1]
    path = nx.astar_path(graph_safe, source, target)

    for _, next_node, data in graph_safe.out_edges(nbunch=target, data=True):
        if data["action"] == action:
            path = path + [next_node]

    # Check shortest path safety
    path_altitudes = altitudes[path]
    path_safety = np.all(-np.diff(path_altitudes) >= h)
    # print(-np.diff(path_altitudes) >= h)
    # print(path)
    return path_safety, path[-1], path


def path_to_boolean_matrix(path, G, S):
    bool_mat = np.zeros_like(S, dtype=bool)
    for i in range(len(path) - 1):
        prev = path[i]
        succ = path[i + 1]
        for _, next_node , data in G.out_edges(nbunch=prev, data=True):
            if next_node == succ:
                bool_mat[prev, 0] = True
                a = data["action"]
                bool_mat[prev, a] = True
                break
    bool_mat[succ, 0] = True
    return bool_mat


# Control plotting and saving
plot_performance = True
plot_completeness = True
plot_exploration_gp = True
plot = plot_performance or plot_completeness or plot_exploration_gp
save_performance = True
plot_for_paper = True
random_experiment = False
non_safe_experiment = False
non_ergodic_experiment = False
no_expanders_exploration = False

# Get mars data
altitudes, coord, world_shape, step_size, num_of_points = mars_map()
# Safety threshold
h = -np.tan(np.pi / 9. + np.pi / 36.) * step_size[0]

# Lipschitz
L = 0.2

# Scaling factor for confidence interval
beta = 2.

# Initialize safe sets
starting_x = 60
starting_y = 61
start = starting_x * world_shape[1] + starting_y
S_hat0 = compute_S_hat0(start, world_shape, 4, altitudes,
                        step_size, h)
S0 = np.copy(S_hat0)
S0[:, 0] = True

# Initialize for performance
time_steps = 20
length = 14.5
sigma_n = 0.075

coverage_over_t = np.empty(time_steps, dtype=float)
dist_from_confidence_interval = np.zeros( altitudes.size, dtype=float)

# Initialize data for GP
X = coord[start, :].reshape(1, 2)
Y = altitudes[start].reshape(1, 1)

# Define and initialize GP
kernel = GPy.kern.Matern52(input_dim=2, lengthscale=length,
                           variance=100.)
lik = GPy.likelihoods.Gaussian(variance=sigma_n ** 2)
gp = GPy.core.GP(X, Y, kernel, lik)

# Define SafeMDP object
x = GridWorld(gp, world_shape, step_size, beta, altitudes, h, S0,
              S_hat0, L, update_dist=25)

# Insert samples from (s, a) in S_hat0 (needs to be more general in
# case for the central state not all actions are safe)
tmp = np.arange(x.coord.shape[0])
s_vec_ind = np.random.choice(tmp[np.all(x.S_hat[:, 1:], axis=1)])

for i in range(5):
    x.add_observation(s_vec_ind, 1)
    x.add_observation(s_vec_ind, 2)
    x.add_observation(s_vec_ind, 3)
    x.add_observation(s_vec_ind, 4)

x.gp.set_XY(X=x.gp.X[1:, :], Y=x.gp.Y[1:, :])
# True S_hat for misclassification
h_hard = -np.tan(np.pi / 6.) * step_size[0]
true_S = compute_true_safe_set(x.world_shape, x.altitudes, h_hard)
true_S_hat = compute_true_S_hat(x.graph, true_S, x.initial_nodes)

# true S_hat with statistical error for completeness
epsilon = sigma_n
true_S_epsilon = compute_true_safe_set(x.world_shape, x.altitudes,
                                       x.h + beta * epsilon)
true_S_hat_epsilon = compute_true_S_hat(x.graph, true_S_epsilon,
                                        x.initial_nodes)
max_size = float(np.count_nonzero(true_S_hat_epsilon))
# Simulation loop
t = time.time()
unsafe_count = 0
source = start
for i in range(time_steps):
    x.update_sets()
    next_sample = x.target_sample()
    x.add_observation(*next_sample)
    path_safety, source, path = check_shortest_path(source,
                                                    next_sample,
                                              x.graph, h_hard, altitudes)
    unsafe_count += not path_safety

    # Performance
    coverage = 100 * np.count_nonzero(np.logical_and(x.S_hat,
                                                true_S_hat_epsilon))/max_size
    false_safe = np.count_nonzero(np.logical_and(x.S_hat, ~true_S_hat))

    # Store and print
    coverage_over_t[i] = coverage
    print(coverage, false_safe, unsafe_count, i)

print(str(time.time() - t) + "seconds elapsed")
print(sigma_n, length)

mu, var = x.gp.predict(x.coord, include_likelihood=False)
sigma = x.beta * np.sqrt(var)
l = np.squeeze(mu - sigma)
u = np.squeeze(mu + sigma)

if plot_exploration_gp:
    fig = plt.figure()
    title = "{0} noise, {1} lengthscale".format(sigma_n, length)
    plt.title(title)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_trisurf(x.coord[:, 0], x.coord[:, 1], altitudes)

    ax1 = fig.add_subplot(121, projection='3d',sharez=ax2)
    ax1.plot_trisurf(x.coord[:, 0], x.coord[:, 1], np.squeeze(mu),
                     alpha=0.5)
    ax1.scatter(x.gp.X[:, 0], x.gp.X[:, 1], x.gp.Y)

# Above u
diff_u = altitudes - u
dist_from_confidence_interval[ diff_u > 0] = diff_u[diff_u > 0]

# Below l
diff_l = altitudes - l
dist_from_confidence_interval[diff_l < 0] = diff_l[diff_l < 0]
# Plot safe sets
# x.plot_S(true_S_hat_epsilon)
# x.plot_S(x.S_hat)

print("Noise: " + str(sigma_n))
print("Lengthscales: " + str(length))
print("Number of points for interpolation: " + str(num_of_points))
print("False safe: ")
print(false_safe)
print("Unsafe evaluations: ")
print(unsafe_count)

if plot_performance:
    plt.figure()
    max_value = np.max(dist_from_confidence_interval)
    min_value = np.min(dist_from_confidence_interval)
    limit = np.max([max_value, np.abs(min_value)])
    plt.imshow(
        np.reshape(dist_from_confidence_interval,x.world_shape).T, origin='lower',
        interpolation='nearest', vmin=-limit, vmax=limit)
    title = "Distance from confidence interval noise {0} - lengthscale {1} - errors {2}".format(
        sigma_n, length, false_safe)
    plt.title(title)
    plt.colorbar()

if plot_completeness:
            plt.figure()
            plt.plot(coverage_over_t)

            title = "Coverage over time -noise {0} - lengthscale {1} - " \
                    "errors {2}".format(sigma_n, length, false_safe)
            plt.title(title)

if plot:
    plt.show()

if save_performance:
    file_name = "mars_errors {0} time steps, {1} n, {2} l, {3} points".format(
        time_steps, sigma_n, length, num_of_points)

    np.savez(file_name, false_safe=false_safe, coverage=coverage,
             coverage_over_t=coverage_over_t, length=length,
             sigma_n=sigma_n,
             dist_from_confidence_interval=dist_from_confidence_interval,
             time_steps=time_steps, world_shape=world_shape)

if plot_for_paper:
    # Plot 2D for paper
    plot_paper(altitudes, x.S_hat, world_shape, './safe_exploration.pdf')


########################## NON SAFE ###########################################
if non_safe_experiment:
    kernel = GPy.kern.Matern52(input_dim=2, lengthscale=length,
                               variance=100.)
    lik = GPy.likelihoods.Gaussian(variance=sigma_n ** 2)
    gp = GPy.core.GP(X, Y, kernel, lik)
    h = -np.tan(np.pi / 9. + np.pi / 36.) * step_size[0]
    # Define SafeMDP object
    x = GridWorld(gp, world_shape, step_size, beta, altitudes, h, np.ones_like(
        S0, dtype=bool), S_hat0, L, update_dist=25)

    # Insert samples from (s, a) in S_hat0 (needs to be more general in
    # case for the central state not all actions are safe)
    tmp = np.arange(x.coord.shape[0])
    s_vec_ind = np.random.choice(tmp[np.all(x.S_hat[:, 1:], axis=1)])

    for i in range(5):
        x.add_observation(s_vec_ind, 1)
        x.add_observation(s_vec_ind, 2)
        x.add_observation(s_vec_ind, 3)
        x.add_observation(s_vec_ind, 4)

    # Remove samples used for GP initialization
    x.gp.set_XY(x.gp.X[n_samples:, :], x.gp.Y[n_samples:])

    t = time.time()
    unsafe_count = 0
    source = start
    trajectory = []
    unsafe = False
    for i in range(time_steps):
        x.update_sets()
        next_sample = x.target_sample()
        x.add_observation(*next_sample)
        path_safety, source, path = check_shortest_path(source, next_sample,
                                                      x.graph, h_hard, altitudes)
        if path_safety:
            trajectory = trajectory + path
        else:
            for j in range(len(path) - 1):
                prev = path[j]
                succ = path[j + 1]
                if altitudes[prev] - altitudes[succ] >= h_hard:
                    trajectory = trajectory + [prev]
                else:
                    unsafe = True
                    break
        # Performance
        # trajectory = np.unique(trajectory)
        # visited = np.zeros_like(S0, dtype=bool)
        # visited[trajectory, 0] = True
        visited = path_to_boolean_matrix(trajectory, x.graph, S0)
        coverage = 100 * np.count_nonzero(np.logical_and(visited,
                                                    true_S_hat_epsilon))/max_size
        false_safe = np.count_nonzero(np.logical_and(x.S_hat, ~true_S_hat))

        # Store and print
        print(coverage, false_safe, unsafe_count, i)
        if unsafe:
            break
    plot_paper(altitudes, visited, world_shape, './no_safe_exploration.pdf')
    x.plot_S(visited)

############################### RANDOM ########################################
if random_experiment:
    kernel = GPy.kern.Matern52(input_dim=2, lengthscale=length,
                               variance=100.)
    lik = GPy.likelihoods.Gaussian(variance=sigma_n ** 2)
    gp = GPy.core.GP(X, Y, kernel, lik)
    # Define SafeMDP object
    x = GridWorld(gp, world_shape, step_size, beta, altitudes, h, np.ones_like(
        S0, dtype=bool), S_hat0, L, update_dist=25)

    source = start
    trajectory = []
    unsafe = False
    for i in range(time_steps):
        a = np.random.choice([1, 2, 3, 4])
        for _, next_node, data in x.graph.out_edges(nbunch=source, data=True):
            if data["action"] == a:
                h_prev = altitudes[source]
                h_succ = altitudes[next_node]
                source = next_node
                if h_prev - h_succ >= h_hard:
                    trajectory = trajectory + [next_node]
                else:
                    unsafe = True
                    break
        if unsafe:
            break
    # trajectory = np.unique(trajectory)
    # visited = np.zeros_like(S0, dtype=bool)
    # visited[trajectory, 0] = True
    visited = path_to_boolean_matrix(trajectory, x.graph, S0)
    coverage = 100 * np.count_nonzero(np.logical_and(visited,
                                                true_S_hat_epsilon))/max_size
    false_safe = np.count_nonzero(np.logical_and(x.S_hat, ~true_S_hat))
    print(coverage, i)
    plot_paper(altitudes, visited, world_shape, './rand_exploration.pdf')
    x.plot_S(visited)

############################# NON ERGODIC #####################################
if non_ergodic_experiment:
    kernel = GPy.kern.Matern52(input_dim=2, lengthscale=length,
                               variance=100.)
    lik = GPy.likelihoods.Gaussian(variance=sigma_n ** 2)
    gp = GPy.core.GP(X, Y, kernel, lik)
    h = -np.tan(np.pi / 9. + np.pi / 36.) * step_size[0]

    # Need to remove expanders otherwise next sample will be in G and
    # therefore in S_hat before I can set S_hat = S
    L_non_ergodic = 100.

    # Define SafeMDP object
    x = GridWorld(gp, world_shape, step_size, beta, altitudes, h, S0, S_hat0,
                  L_non_ergodic, update_dist=25)

    # Insert samples from (s, a) in S_hat0 (needs to be more general in
    # case for the central state not all actions are safe)
    tmp = np.arange(x.coord.shape[0])
    s_vec_ind = np.random.choice(tmp[np.all(x.S_hat[:, 1:], axis=1)])

    for i in range(5):
        x.add_observation(s_vec_ind, 1)
        x.add_observation(s_vec_ind, 2)
        x.add_observation(s_vec_ind, 3)
        x.add_observation(s_vec_ind, 4)

    # Remove samples used for GP initialization
    x.gp.set_XY(x.gp.X[n_samples:, :], x.gp.Y[n_samples:])

    # Simulation loop
    t = time.time()
    unsafe_count = 0
    source = start
    for i in range(time_steps):
        x.update_sets()
        x.S_hat = x.S.copy()
        next_sample = x.target_sample()
        x.add_observation(*next_sample)
        try:
            path_safety, source, path = check_shortest_path(source, next_sample,
                                                  x.graph, h_hard, altitudes)
        except Exception:
            print ("No safe path available")
            break
        unsafe_count += not path_safety

    x.S_hat[:, 0] = np.any(x.S_hat[:, 1:], axis=1)
    # Performance
    coverage = 100 * np.count_nonzero(np.logical_and(x.S_hat,
                                                true_S_hat_epsilon))/max_size

    # Store and print
    print(coverage, unsafe_count, i)
    plot_paper(altitudes, x.S_hat, world_shape,
               './no_ergodic_exploration.pdf')
    x.plot_S(x.S_hat)

################################## NO EXPANDERS ###############################
if no_expanders_exploration:
    kernel = GPy.kern.Matern52(input_dim=2, lengthscale=length,
                               variance=100.)
    lik = GPy.likelihoods.Gaussian(variance=sigma_n ** 2)
    gp = GPy.core.GP(X, Y, kernel, lik)
    h = -np.tan(np.pi / 9. + np.pi / 36.) * step_size[0]

    L_no_expanders = 100.

    x = GridWorld(gp, world_shape, step_size, beta, altitudes, h, S0, S_hat0,
                  L_no_expanders, update_dist=25)

    # Insert samples from (s, a) in S_hat0 (needs to be more general in
    # case for the central state not all actions are safe)
    tmp = np.arange(x.coord.shape[0])
    s_vec_ind = np.random.choice(tmp[np.all(x.S_hat[:, 1:], axis=1)])

    for i in range(5):
        x.add_observation(s_vec_ind, 1)
        x.add_observation(s_vec_ind, 2)
        x.add_observation(s_vec_ind, 3)
        x.add_observation(s_vec_ind, 4)

    # Remove samples used for GP initialization
    x.gp.set_XY(x.gp.X[n_samples:, :], x.gp.Y[n_samples:])

    source = start
    unsafe_count = 0
    for i in range(int(time_steps)):
        x.update_sets()
        next_sample = x.target_sample()
        x.add_observation(*next_sample)
        path_safety, source, path = check_shortest_path(source,
                                                            next_sample,
                                                      x.graph, h_hard, altitudes)
        unsafe_count += not path_safety
        # Performance
        coverage = 100 * np.count_nonzero(np.logical_and(x.S_hat,
                                                    true_S_hat_epsilon))/max_size
        false_safe = np.count_nonzero(np.logical_and(x.S_hat, ~true_S))
        print(coverage, false_safe, unsafe_count, i)
    plot_paper(altitudes, x.S_hat, world_shape, './no_G_exploration.pdf')
    x.plot_S(x.S_hat)

