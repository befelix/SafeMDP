from osgeo import gdal
from scipy import interpolate
import numpy as np
import os
import matplotlib.pyplot as plt


def mars_map(plot_map=False, interpolation=False):

    # Define the dimension of the map we want to investigate and its resolution
    world_shape = (120, 70)#(60, 42)
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
