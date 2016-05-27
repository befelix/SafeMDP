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
non_safe_experiment = False
non_ergodic_experiment = False
no_expanders_exploration = True

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
        print("----UNSAFE EXPERIMENT---")
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

    # Get mars data
    altitudes, coord, world_shape, step_size, num_of_points = mars_map()

    # Initialize object for simulation
    start, x, true_S_hat, true_S_hat_epsilon, h_hard = initialize_SafeMDP_object(
        altitudes, coord, world_shape, step_size)

    source = start
    trajectory = [source]

    for i in range(time_steps):

        # Choose action at random
        a = np.random.choice([1, 2, 3, 4])

        # Add resulting state to trajectory
        for _, next_node, data in x.graph.out_edges(nbunch=source, data=True):
            if data["action"] == a:
                trajectory = trajectory + [next_node]
                source = next_node

    # Check safety
    path_altitudes = x.altitudes[trajectory]
    unsafe_count = np.sum(-np.diff(path_altitudes) < h_hard)

    # Get trajectory up to first unsafe transition
    trajectory = safe_subpath(trajectory, x.altitudes, h_hard)

    # Convert trajectory to S_hat-like matrix
    visited = path_to_boolean_matrix(trajectory, x.graph, x.S)

    # Normalization factor
    max_size = float(np.count_nonzero(true_S_hat_epsilon))

    # Performance
    coverage = 100 * np.count_nonzero(np.logical_and(visited,
                                                true_S_hat_epsilon))/max_size
    # Print
    print("----RANDOM EXPERIMENT---")
    print("Unsafe evaluations: " + str(unsafe_count))
    print("Coverage: " + str(coverage))

    if save_performance:
        file_name = "mars random experiment"

        np.savez(file_name, coverage=coverage, altitudes=x.altitudes,
                visited=visited, world_shape=world_shape)

############################# NON ERGODIC #####################################
if non_ergodic_experiment:

    # Get mars data
    altitudes, coord, world_shape, step_size, num_of_points = mars_map()

    # Need to remove expanders otherwise next sample will be in G and
    # therefore in S_hat before I can set S_hat = S
    L_non_ergodic = 1000.

    # Initialize object for simulation
    start, x, true_S_hat, true_S_hat_epsilon, h_hard = initialize_SafeMDP_object(
        altitudes, coord, world_shape, step_size, L=L_non_ergodic)

    # Simulation loop
    unsafe_count = 0
    source = start

    for i in range(time_steps):
        x.update_sets()

        # Remove ergodicity properties
        x.S_hat = x.S.copy()

        next_sample = x.target_sample()
        x.add_observation(*next_sample)
        try:
            path = shortest_path(source, next_sample, x.graph)
            source = path[-1]

            # Check safety
            path_altitudes = x.altitudes[path]
            unsafe_transitions = np.sum(-np.diff(path_altitudes) < h_hard)
            unsafe_count += unsafe_transitions
        except Exception:
            print ("No safe path available")
            break

    # For coverage we consider every state that has at least one action
    # classified as safe
    x.S_hat[:, 0] = np.any(x.S_hat[:, 1:], axis=1)

    # Normalization factor
    max_size = float(np.count_nonzero(true_S_hat_epsilon))

    # Performance
    coverage = 100 * np.count_nonzero(np.logical_and(x.S_hat,
                                                true_S_hat_epsilon))/max_size

    # Print
    print("----NON ERGODIC EXPERIMENT---")
    print("Unsafe evaluations: " + str(unsafe_count))
    print("Coverage: " + str(coverage))

    if save_performance:
        file_name = "mars non ergodic experiment"

        np.savez(file_name, coverage=coverage, altitudes=x.altitudes,
                S_hat=x.S_hat, world_shape=world_shape)

################################## NO EXPANDERS ###############################
if no_expanders_exploration:

    # Get mars data
    altitudes, coord, world_shape, step_size, num_of_points = mars_map()

    L_no_expanders = 1000.

    # Initialize object for simulation
    start, x, true_S_hat, true_S_hat_epsilon, h_hard = initialize_SafeMDP_object(
        altitudes, coord, world_shape, step_size, L=L_no_expanders)

    # Initialize for performance storage
    coverage_over_t = np.empty(time_steps, dtype=float)

    # Simulation loop
    source = start
    unsafe_count = 0

    for i in range(int(time_steps)):

        #Simulation
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

    # Print
    print("----NO EXPANDER EXPERIMENT---")
    print("False safe: " + str(false_safe))
    print("Unsafe evaluations: " + str(unsafe_count))
    print("Coverage: " + str(coverage))

    if save_performance:
        file_name = "mars no G experiment"

        np.savez(file_name, false_safe=false_safe, coverage=coverage,
            coverage_over_t=coverage_over_t, altitudes=x.altitudes, S_hat=x.S_hat,
            world_shape=world_shape)
