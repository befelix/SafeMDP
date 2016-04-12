from __future__ import division, print_function

import time
import numpy as np
import os
from osgeo import gdal
import GPy
import matplotlib.pyplot as plt

from SafeMDP_class import *
from utilities import *

# Extract and plot Mars data
world_shape = (60, 60)
step_size = (1., 1.)
gdal.UseExceptions()

# Download data files
if not os.path.exists('./mars.tif'):
    if not os.path.exists("./mars.IMG"):
        import urllib

        # Download the IMG file
        urllib.urlretrieve(
            "http://www.uahirise.org//PDS/DTM/ESP/ORB_033600_033699"
            "/ESP_033617_1990_ESP_034316_1990"
            "/DTEED_033617_1990_034316_1990_A01.IMG", "mars.IMG")

    # Convert to tif
    os.system("gdal_translate -of GTiff ./mars.IMG ./mars.tif")

ds = gdal.Open("./mars.tif")
band = ds.GetRasterBand(1)
elevation = band.ReadAsArray()
startX = 11370
startY = 3110
altitudes = np.copy(elevation[startX:startX + world_shape[0],
                    startY:startY + world_shape[1]])
mean_val = (np.max(altitudes) + np.min(altitudes)) / 2.
altitudes[:] = altitudes - mean_val

plt.imshow(altitudes.T, origin="lower", interpolation="nearest")
plt.colorbar()
plt.show()
altitudes = altitudes.flatten()

# Define coordinates
n, m = world_shape
step1, step2 = step_size
xx, yy = np.meshgrid(np.linspace(0, (n - 1) * step1, n),
                     np.linspace(0, (m - 1) * step2, m), indexing="ij")
coord = np.vstack((xx.flatten(), yy.flatten())).T

# Define coordinates
n, m = world_shape
step1, step2 = step_size
xx, yy = np.meshgrid(np.linspace(0, (n - 1) * step1, n),
                     np.linspace(0, (m - 1) * step2, m), indexing="ij")
coord = np.vstack((xx.flatten(), yy.flatten())).T

# Safety threshold
h = -np.tan(np.pi / 6.)

# Lipschitz
L = 1.

# Scaling factor for confidence interval
beta = 2

# Initialize safe sets
S0 = np.zeros((np.prod(world_shape), 5), dtype=bool)
S0[:, 0] = True
S_hat0 = compute_S_hat0(2550, world_shape, 4, altitudes,
                        step_size, h)

# Initialize for performance
lengthScale = np.linspace(6.5, 7., num=2)
size_true_S_hat = np.empty_like(lengthScale, dtype=int)
size_S_hat = np.empty_like(lengthScale, dtype=int)
true_S_hat_minus_S_hat = np.empty_like(lengthScale, dtype=int)
S_hat_minus_true_S_hat = np.empty_like(lengthScale, dtype=int)

# Initialize data for GP
n_samples = 1
ind = np.random.choice(range(altitudes.size), n_samples)
X = coord[ind, :]
Y = altitudes[ind].reshape(n_samples, 1) + np.random.randn(n_samples,
                                                           1)

for index, length in enumerate(lengthScale):

    # Define and initialize GP
    noise = 0.04
    kernel = GPy.kern.RBF(input_dim=2, lengthscale=length,
                          variance=121.)
    lik = GPy.likelihoods.Gaussian(variance=noise ** 2)
    gp = GPy.core.GP(X, Y, kernel, lik)

    # Define SafeMDP object
    x = GridWorld(gp, world_shape, step_size, beta, altitudes, h, S0,
                  S_hat0, L)

    # Insert samples from (s, a) in S_hat0
    tmp = np.arange(x.coord.shape[0])
    s_vec_ind = np.random.choice(tmp[np.any(x.S_hat[:, 1:], axis=1)])
    tmp = np.arange(1, x.S.shape[1])
    actions = tmp[x.S_hat[s_vec_ind, 1:].squeeze()]
    for i in range(1):
        x.add_observation(s_vec_ind, np.random.choice(actions))

    # Remove samples used for GP initialization and possibly
    # hyperparameters optimization
    x.gp.set_XY(x.gp.X[n_samples:, :], x.gp.Y[n_samples:])

    t = time.time()
    for i in range(50):
        x.update_sets()
        next_sample = x.target_sample()
        x.add_observation(*next_sample)
        # print (x.target_state, x.target_action)
        # print(i)
        print(np.any(x.G))
    print(str(time.time() - t) + "seconds elapsed")

    # Plot safe sets
    x.plot_S(x.S_hat)
    x.plot_S(x.true_S_hat)

    # Print and store performance
    print(np.sum(np.logical_and(x.true_S_hat,
                                np.logical_not(x.S_hat))))
    # in true S_hat and not S_hat
    print(np.sum(np.logical_and(x.S_hat,
                                np.logical_not(x.true_S_hat))))
    # in S_hat and not true S_hat
    size_S_hat[index] = np.sum(x.S_hat)
    size_true_S_hat[index] = np.sum(x.true_S_hat)
    true_S_hat_minus_S_hat[index] = np.sum(
        np.logical_and(x.true_S_hat, np.logical_not(x.S_hat)))
    S_hat_minus_true_S_hat[index] = np.sum(
        np.logical_and(x.S_hat, np.logical_not(x.true_S_hat)))
