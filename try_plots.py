from plot_utilities import *
import numpy as np
import cPickle
import time

# Safe plots
data = np.load("mars safe experiment.npz")
mu = data["mu_alt"]
var = data["var_alt"]
beta = data["beta"]
world_shape = data["world_shape"]
altitudes = data["altitudes"]
X = data["X"]
Y = data["Y"]
coord = data["coord"]
coverage_over_t = data["coverage_over_t"]
S_hat = data["S_hat"]

plot_dist_from_C(mu, var, beta, altitudes, world_shape)
plot_coverage(coverage_over_t)
plot_paper(altitudes, S_hat, world_shape, './safe_exploration.pdf')

# Unsafe plot
data = np.load("mars unsafe experiment.npz")
coverage = data["coverage"]
altitudes = data["altitudes"]
visited = data["visited"]
world_shape = data["world_shape"]

plot_paper(altitudes, visited, world_shape, './no_safe_exploration1.pdf')

# Random plot
data = np.load("mars random experiment.npz")
coverage = data["coverage"]
altitudes = data["altitudes"]
visited = data["visited"]
world_shape = data["world_shape"]

plot_paper(altitudes, visited, world_shape, './random_exploration1.pdf')

# Non ergodic plot
data = np.load("mars non ergodic experiment.npz")
coverage = data["coverage"]
altitudes = data["altitudes"]
S_hat = data["S_hat"]
world_shape = data["world_shape"]

plot_paper(altitudes, S_hat, world_shape, './no_ergodic_exploration1.pdf')

# No expanders plot
data = np.load("mars no G experiment.npz")
coverage = data["coverage"]
altitudes = data["altitudes"]
S_hat = data["S_hat"]
world_shape = data["world_shape"]
coverage_over_t = data["coverage_over_t"]
plot_coverage(coverage_over_t)

plot_paper(altitudes, S_hat, world_shape, './no_G_exploration1.pdf')