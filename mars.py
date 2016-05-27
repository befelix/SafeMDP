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

from plot_utilities import *
from mars_utilities import *
from src.grid_world import *

print(sys.version)


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


def safe_subpath(path, altitudes, h):
    subpath = [path[0]]
    for j in range(len(path) - 1):
        prev = path[j]
        succ = path[j + 1]
        if altitudes[prev] - altitudes[succ] >= h:
            subpath = subpath + [succ]
        else:
            break
    return subpath


# Control plotting and saving
save_performance = True
random_experiment = False
non_safe_experiment = True
non_ergodic_experiment = False
no_expanders_exploration = False

# Get mars data
altitudes, coord, world_shape, step_size, num_of_points = mars_map()

# Initialize object for simulation
start, x, true_S_hat, true_S_hat_epsilon, h_hard = initialize_SafeMDP_object(
    altitudes, coord, world_shape, step_size)

# Initialize for performance storage
time_steps = 20
coverage_over_t = np.empty(time_steps, dtype=float)

# Simulation loop
t = time.time()
unsafe_count = 0
source = start

for i in range(time_steps):

    # Simulation
    x.update_sets()
    next_sample = x.target_sample()
    x.add_observation(*next_sample)
    path = shortest_path(source, next_sample, x.graph)
    source = path[-1]

    # Performances
    unsafe_transitions, coverage, false_safe = performance_metrics(path, x,
                                                                   true_S_hat_epsilon,
                                                                   true_S_hat, h_hard)
    unsafe_count += unsafe_transitions
    coverage_over_t[i] = coverage
    print(coverage, false_safe, unsafe_count, i)

print(str(time.time() - t) + "seconds elapsed")

# Posterior over heights for plotting
mu_alt, var_alt = x.gp.predict(x.coord, include_likelihood=False)


print("Number of points for interpolation: " + str(num_of_points))
print("False safe: " +str(false_safe))
print("Unsafe evaluations: " + str(unsafe_count))
print("Coverage: " + str(coverage))

if save_performance:
    file_name = "mars safe experiment"

    np.savez(file_name, false_safe=false_safe, coverage=coverage,
             coverage_over_t=coverage_over_t, mu_alt=mu_alt,
             var_alt=var_alt, altitudes=x.altitudes, S_hat=x.S_hat,
             time_steps=time_steps, world_shape=world_shape, X=x.gp.X,
             Y=x.gp.Y, coord=x.coord, beta=x.beta)


########################## NON SAFE ###########################################
if non_safe_experiment:

    # Get mars data
    altitudes, coord, world_shape, step_size, num_of_points = mars_map()

    # Initialize object for simulation
    start, x, true_S_hat, true_S_hat_epsilon, h_hard = initialize_SafeMDP_object(
        altitudes, coord, world_shape, step_size)

    # Assume all transitions are safe
    x.S[:] = True

    # Simulation loop
    t = time.time()
    unsafe_count = 0
    source = start
    trajectory = []

    for i in range(time_steps):
        x.update_sets()
        next_sample = x.target_sample()
        x.add_observation(*next_sample)
        path = shortest_path(source, next_sample, x.graph)
        source = path[-1]

        # Check safety
        path_altitudes = x.altitudes[path]
        unsafe_count = np.sum(-np.diff(path_altitudes) < h_hard)

        if unsafe_count == 0:
            trajectory = trajectory + path[:-1]
        else:
            trajectory = trajectory + safe_subpath(path, altitudes, h_hard)

        # Convert trajectory to S_hat-like matrix
        visited = path_to_boolean_matrix(trajectory, x.graph, x.S)

        # Normalization factor
        max_size = float(np.count_nonzero(true_S_hat_epsilon))

        # Performance
        coverage = 100 * np.count_nonzero(
            np.logical_and(visited, true_S_hat_epsilon)) / max_size

        # Print
        print("Unsafe evaluations: " + str(unsafe_count))
        print("Coverage: " + str(coverage))

        if unsafe_count > 0:
            break

    if save_performance:
        file_name = "mars unsafe experiment"

        np.savez(file_name, coverage=coverage, altitudes=x.altitudes,
                visited=visited, world_shape=world_shape)

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

