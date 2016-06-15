from safemdp.grid_world import *
from osgeo import gdal
from scipy import interpolate
import numpy as np
import os
import matplotlib.pyplot as plt
import GPy


def mars_map(plot_map=False, interpolation=False):
    """
    Extract the map for the simulation from the HiRISE data. If the HiRISE
    data is not in the current folder it will be downloaded and converted to
    GeoTiff extension with gdal.

    Parameters
    ----------
    plot_map: bool
        If true plots the map that will be used for exploration
    interpolation: bool
        If true the data of the map will be interpolated with splines to
        obtain a finer grid

    Returns
    -------
    altitudes: np.array
        1-d vector with altitudes for each node
    coord: np.array
        Coordinate of the map we use for exploration
    world_shape: tuple
        Size of the grid world (rows, columns)
    step_size: tuple
        Step size for the grid (row, column)
    num_of_points: int
        Interpolation parameter. Indicates the scaling factor for the
        original step size
    """

    # Define the dimension of the map we want to investigate and its resolution
    world_shape = (120, 70)
    step_size = (1., 1.)

    # Download and convert to GEOtiff Mars data
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

    # Read the data with gdal module
    gdal.UseExceptions()
    ds = gdal.Open("./mars.tif")
    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray()

    # Extract the area of interest
    startX = 2890
    startY = 1955
    altitudes = np.copy(elevation[startX:startX + world_shape[0],
                        startY:startY + world_shape[1]])

    # Center the data
    mean_val = (np.max(altitudes) + np.min(altitudes)) / 2.
    altitudes[:] = altitudes - mean_val

    # Define coordinates
    n, m = world_shape
    step1, step2 = step_size
    xx, yy = np.meshgrid(np.linspace(0, (n - 1) * step1, n),
                         np.linspace(0, (m - 1) * step2, m), indexing="ij")
    coord = np.vstack((xx.flatten(), yy.flatten())).T

    # Interpolate data
    if interpolation:

        # Interpolating function
        spline_interpolator = interpolate.RectBivariateSpline(
            np.linspace(0, (n - 1) * step1, n),
            np.linspace(0, (m - 1) * step1, m), altitudes)

        # New size and resolution
        num_of_points = 1
        world_shape = tuple([(x - 1) * num_of_points + 1 for x in world_shape])
        step_size = tuple([x / num_of_points for x in step_size])

        # New coordinates and altitudes
        n, m = world_shape
        step1, step2 = step_size
        xx, yy = np.meshgrid(np.linspace(0, (n - 1) * step1, n),
                             np.linspace(0, (m - 1) * step2, m), indexing="ij")
        coord = np.vstack((xx.flatten(), yy.flatten())).T

        altitudes = spline_interpolator(np.linspace(0, (n - 1) * step1, n),
                                        np.linspace(0, (m - 1) * step2, m))
    else:
        num_of_points = 1

    # Plot area
    if plot_map:
        plt.imshow(altitudes.T, origin="lower", interpolation="nearest")
        plt.colorbar()
        plt.show()
    altitudes = altitudes.flatten()

    return altitudes, coord, world_shape, step_size, num_of_points


def initialize_SafeMDP_object(altitudes, coord, world_shape, step_size, L=0.2,
                              beta=2, length=14.5, sigma_n=0.075, start_x=60,
                              start_y=61):
    """

    Parameters
    ----------
    altitudes: np.array
        1-d vector with altitudes for each node
    coord: np.array
        Coordinate of the map we use for exploration
    world_shape: tuple
        Size of the grid world (rows, columns)
    step_size: tuple
        Step size for the grid (row, column)
    L: float
        Lipschitz constant to compute expanders
    beta: float
        Scaling factor for confidence intervals
    length: float
        Lengthscale for Matern kernel
    sigma_n:
        Standard deviation for gaussian noise
    start_x: int
        x coordinate of the starting point
    start_y:
        y coordinate of the starting point

    Returns
    -------
    start: int
        Node number of initial state
    x: SafeMDP
        Instance of the SafeMDP class for the mars exploration problem
    true_S_hat: np.array
        True S_hat if safety feature is known with no error and h_hard is used
    true_S_hat_epsilon: np.array
        True S_hat if safety feature is known up to epsilon and h is used
    h_hard: float
        True safety thrshold. It can be different from the safety threshold
        used for classification in case the agent needs to use extra caution
        (in our experiments h=25 deg, h_har=30 deg)
    """

    # Safety threshold
    h = -np.tan(np.pi / 9. + np.pi / 36.) * step_size[0]

    #Initial node
    start = start_x * world_shape[1] + start_y

    # Initial safe sets
    S_hat0 = compute_S_hat0(start, world_shape, 4, altitudes,
                        step_size, h)
    S0 = np.copy(S_hat0)
    S0[:, 0] = True

    # Initialize GP
    X = coord[start, :].reshape(1, 2)
    Y = altitudes[start].reshape(1, 1)
    kernel = GPy.kern.Matern52(input_dim=2, lengthscale=length, variance=100.)
    lik = GPy.likelihoods.Gaussian(variance=sigma_n ** 2)
    gp = GPy.core.GP(X, Y, kernel, lik)

    # Define SafeMDP object
    x = GridWorld(gp, world_shape, step_size, beta, altitudes, h, S0,
              S_hat0, L, update_dist=25)

    # Add samples about actions from starting node
    for i in range(5):
        x.add_observation(start, 1)
        x.add_observation(start, 2)
        x.add_observation(start, 3)
        x.add_observation(start, 4)

    x.gp.set_XY(X=x.gp.X[1:, :], Y=x.gp.Y[1:, :]) # Necessary for results as in
    # paper

    # True safe set for false safe
    h_hard = -np.tan(np.pi / 6.) * step_size[0]
    true_S = compute_true_safe_set(x.world_shape, x.altitudes, h_hard)
    true_S_hat = compute_true_S_hat(x.graph, true_S, x.initial_nodes)

    # True safe set for completeness
    epsilon = sigma_n * beta
    true_S_epsilon = compute_true_safe_set(x.world_shape, x.altitudes,
                                           x.h + epsilon)
    true_S_hat_epsilon = compute_true_S_hat(x.graph, true_S_epsilon,
                                            x.initial_nodes)

    return start, x, true_S_hat, true_S_hat_epsilon, h_hard


def performance_metrics(path, x, true_S_hat_epsilon, true_S_hat, h_hard):
    """

    Parameters
    ----------
    path: np.array
        Nodes of the shortest safe path
    x: SafeMDP
         Instance of the SafeMDP class for the mars exploration problem
     true_S_hat_epsilon: np.array
        True S_hat if safety feature is known up to epsilon and h is used
    true_S_hat: np.array
        True S_hat if safety feature is known with no error and h_hard is used
    h_hard: float
        True safety thrshold. It can be different from the safety threshold
        used for classification in case the agent needs to use extra caution
        (in our experiments h=25 deg, h_har=30 deg)

    Returns
    -------
    unsafe_transitions: int
        Number of unsafe transitions along the path
    coverage: float
        Percentage of coverage of true_S_hat_epsilon
    false_safe: int
        Number of misclassifications (classifing something as safe when it
        acutally is unsafe according to h_hard )
    """

    # Count unsafe transitions along the path
    path_altitudes = x.altitudes[path]
    unsafe_transitions = np.sum(-np.diff(path_altitudes) < h_hard)

    # Coverage
    max_size = float(np.count_nonzero(true_S_hat_epsilon))
    coverage = 100 * np.count_nonzero(np.logical_and(x.S_hat,
                                                true_S_hat_epsilon))/max_size
    # False safe
    false_safe = np.count_nonzero(np.logical_and(x.S_hat, ~true_S_hat))

    return unsafe_transitions, coverage, false_safe
