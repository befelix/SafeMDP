from __future__ import division, print_function

import sys
import os
import time

import GPy
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

from src.grid_world import *
from plot_utilities import *

print(sys.version)

# Control plotting and saving
plot_map = False
plot_performance = False
plot_completeness = False
plot_initial_gp = False
plot_exploration_gp = False
plot = plot_map or plot_performance or plot_completeness or plot_initial_gp \
       or plot_exploration_gp
save_performance = True
plot_for_paper = False

# Extract and plot Mars data
world_shape = (120, 70)#(60, 42)
step_size = (1., 1.)
gdal.UseExceptions()

# Download data files
if not os.path.exists('./mars.tif'):
    if not os.path.exists("./mars.IMG"):
        import urllib

        # Download the IMG file
        urllib.urlretrieve(
            "http://www.uahirise.org/PDS/DTM/PSP/ORB_010200_010299"
            "/PSP_010228_1490_ESP_016320_1490"
            "/DTEEC_010228_1490_016320_1490_A01.IMG", "mars.IMG")
    # Convert to tif
    os.system("gdal_translate -of GTiff ./mars.IMG ./mars.tif")

ds = gdal.Open("./mars.tif")
band = ds.GetRasterBand(1)
elevation = band.ReadAsArray()

# Extract and shift interesting area
startX = 2890  # 2960 2ith world_shape = (60, 60)
startY = 1955  # Before it was 1965 with world_shape = [60, 60]
altitudes = np.copy(elevation[startX:startX + world_shape[0],
                    startY:startY + world_shape[1]])
mean_val = (np.max(altitudes) + np.min(altitudes)) / 2.
altitudes[:] = altitudes - mean_val

# Plot area
if plot_map:
    plt.imshow(altitudes.T, origin="lower", interpolation="nearest")
    plt.colorbar()
altitudes = altitudes.flatten()

# Define coordinates
n, m = world_shape
step1, step2 = step_size
xx, yy = np.meshgrid(np.linspace(0, (n - 1) * step1, n),
                     np.linspace(0, (m - 1) * step2, m), indexing="ij")
coord = np.vstack((xx.flatten(), yy.flatten())).T

# Safety threshold
h = -np.tan(np.pi / 9. + np.pi / 36.) * step_size[0]

# Lipschitz
L = 0.2

# Scaling factor for confidence interval
beta = 2.

# Initialize safe sets
S0 = np.zeros((np.prod(world_shape), 5), dtype=bool)
S0[:, 0] = True
starting_x = 60
starting_y = 61
start = starting_x * world_shape[1] + starting_y
S_hat0 = compute_S_hat0(start, world_shape, 4, altitudes,
                        step_size, h) # 113 when you go back to 60 by 60 map
#  or 2093 with (150, 42)

# Initialize for performance
time_steps = 600
lengthScale = np.linspace(22., 7., num=1)
noise = np.linspace(0.05, 0.11, num=1)
parameters_shape = (noise.size, lengthScale.size)

size_S_hat = np.empty(parameters_shape, dtype=int)
true_S_hat_minus_S_hat = np.empty(parameters_shape, dtype=float)
S_hat_minus_true_S_hat = np.empty(parameters_shape, dtype=int)
completeness = np.empty(parameters_shape + (time_steps,), dtype=float)
dist_from_confidence_interval = np.zeros(parameters_shape + (altitudes.size,),
                                         dtype=float)

# Initialize data for GP
n_samples = 100
ind = np.random.choice(range(altitudes.size), n_samples)
X = coord[ind, :]
Y = altitudes[ind].reshape(n_samples, 1)

# Loop over lengthscales and noise values
for index_l, length in enumerate(lengthScale):
    for index_n, sigma_n in enumerate(noise):

        # Define and initialize GP
        # kernel = GPy.kern.RBF(input_dim=2, lengthscale=length,
        #                       variance=100.)
        kernel = GPy.kern.Matern52(input_dim=2, lengthscale=length,
                                   variance=100.)
        lik = GPy.likelihoods.Gaussian(variance=sigma_n ** 2)
        gp = GPy.core.GP(X, Y, kernel, lik)

        if plot_initial_gp:
            mu, var = gp.predict(coord, include_likelihood=False)
            sigma = beta * np.sqrt(var)
            l = np.squeeze(mu - sigma)
            u = np.squeeze(mu + sigma)
            fig = plt.figure()
            title = "{0} noise, {1} lengthscale".format(sigma_n, length)
            plt.title(title)

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot_trisurf(coord[:, 0], coord[:, 1], altitudes)

            ax1 = fig.add_subplot(121, projection='3d', sharez=ax2)
            ax1.plot_trisurf(coord[:, 0], coord[:, 1], np.squeeze(mu), alpha=0.5)
            ax1.scatter(X[:, 0], X[:, 1], Y, depthshade=False, s=40)

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

        # Remove samples used for GP initialization
        x.gp.set_XY(x.gp.X[n_samples:, :], x.gp.Y[n_samples:])

        # True S_hat for misclassification
        h_hard = -np.tan(np.pi / 6.) * step_size[0]
        true_S = compute_true_safe_set(x.world_shape, x.altitudes, h_hard)
        true_S_hat = compute_true_S_hat(x.graph, true_S, x.initial_nodes)

        # true S_hat with statistical error for completeness
        epsilon = sigma_n
        true_S_epsilon = compute_true_safe_set(x.world_shape, x.altitudes,
                                               x.h + epsilon)
        true_S_hat_epsilon = compute_true_S_hat(x.graph, true_S_epsilon,
                                                x.initial_nodes)
        max_size = float(np.count_nonzero(true_S_hat_epsilon))

        # Simulation loop
        t = time.time()

        for i in range(time_steps):
            x.update_sets()
            next_sample = x.target_sample()
            x.add_observation(*next_sample)

            # Performance
            coverage = 100 * np.count_nonzero(np.logical_and(x.S_hat,
                                                        true_S_hat_epsilon))/max_size
            false_safe = np.count_nonzero(np.logical_and(x.S_hat, ~true_S_hat))

            # Store and print
            completeness[index_n, index_l, i] = coverage
            print(coverage, false_safe)

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
        dist_from_confidence_interval[index_n, index_l, diff_u > 0] = diff_u[
            diff_u > 0]

        # Below l
        diff_l = altitudes - l
        dist_from_confidence_interval[index_n, index_l, diff_l < 0] = diff_l[
            diff_l < 0]
        # Plot safe sets
        # x.plot_S(true_S_hat_epsilon)
        # x.plot_S(x.S_hat)

        size_S_hat[index_n, index_l] = np.sum(x.S_hat)
        true_S_hat_minus_S_hat[index_n, index_l] = coverage
        S_hat_minus_true_S_hat[index_n, index_l] = false_safe

print("Noise: " + str(noise))
print("Lengthscales: " + str(lengthScale))
print("Size S_hat:")
print(size_S_hat)
print("Coverage:")
print(true_S_hat_minus_S_hat)
print("False safe: ")
print(S_hat_minus_true_S_hat)

if plot_performance:

    # As a function of noise
    plt.figure()
    plt.plot(noise, S_hat_minus_true_S_hat)
    plt.xlabel("Noise")
    title = "{0} time steps, {1}-{2} noise, {3}-{4} lengthscale".format\
        (time_steps, noise[0], noise[-1], lengthScale[0], lengthScale[-1])
    plt.title(title)

    # As a function of lengthscale
    plt.figure()
    plt.plot(lengthScale, S_hat_minus_true_S_hat.T)
    plt.xlabel("Lengthscale")
    plt.title(title)

    for index_l, length in enumerate(lengthScale):
        for index_n, sigma_n in enumerate(noise):
            plt.figure()
            max_value = np.max(dist_from_confidence_interval[index_n,
                               index_l, :])
            min_value = np.min(dist_from_confidence_interval[index_n,
                               index_l, :])
            limit = np.max([max_value, np.abs(min_value)])
            plt.imshow(
                np.reshape(dist_from_confidence_interval[index_n, index_l, :],
                           x.world_shape).T, origin='lower',
                interpolation='nearest', vmin=-limit, vmax=limit)
            title = "noise {0} - lengthscale {1} - errors {2}".format(
                sigma_n, length, S_hat_minus_true_S_hat[index_n, index_l])
            plt.title(title)
            plt.colorbar()

if plot_completeness:
    for index_l, length in enumerate(lengthScale):
        for index_n, sigma_n in enumerate(noise):
            plt.figure()
            plt.plot(completeness[index_n, index_l, :])

            title = "noise {0} - lengthscale {1} - errors {2}".format(
                sigma_n, length, S_hat_minus_true_S_hat[index_n, index_l])
            plt.title(title)

if plot:
    plt.show()

if save_performance:
    file_name = "mars_errors {0} time steps, {1}-{2} n, {3}-{4} l".format(
        time_steps, noise[0], noise[-1], lengthScale[0], lengthScale[-1])

    np.savez(file_name, S_hat_minus_true_S_hat=S_hat_minus_true_S_hat,
             true_S_hat_minus_S_hat=true_S_hat_minus_S_hat,
             completeness=completeness, lengthScale=lengthScale, noise=noise,
             dist_from_confidence_interval=dist_from_confidence_interval,
             time_steps=time_steps, world_shape=world_shape)

if plot_for_paper:
    # Plot 2D for paper
    plot_paper(altitudes, x.S_hat, world_shape)

    # Plot 3D for paper
    plot_paper(altitudes, x.S_hat, world_shape, surf=True, coord=coord)
