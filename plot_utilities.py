import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import ColorConverter


def paper_figure(figsize, subplots=None, **kwargs):
    """Define default values for font, fontsize and use latex"""

    def cm2inch(cm_tupl):
        """Convert cm to inches"""
        inch = 2.54
        return (cm / inch for cm in cm_tupl)

    if subplots is None:
        fig = plt.figure(figsize=cm2inch(figsize))
    else:
        fig, ax = plt.subplots(subplots[0], subplots[1],
                               figsize=cm2inch(figsize), **kwargs)

    # Parameters for IJRR
    params = {
              'font.family': 'serif',
              'font.serif': ['Times',
                             'Palatino',
                             'New Century Schoolbook',
                             'Bookman',
                             'Computer Modern Roman'],
              'font.sans-serif': ['Times',
                                  'Helvetica',
                                  'Avant Garde',
                                  'Computer Modern Sans serif'],
              'text.usetex': True,
              # Make sure mathcal doesn't use the Times style
              'text.latex.preamble':
                  r'\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}',

              'axes.labelsize': 9,
              'axes.linewidth': .75,

              'font.size': 9,
              'legend.fontsize': 9,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,

              # 'figure.dpi': 150,
              # 'savefig.dpi': 600,
              'legend.numpoints': 1,
              }

    rcParams.update(params)

    if subplots is None:
        return fig
    else:
        return fig, ax


def format_figure(axis, cbar=None):
    axis.spines['top'].set_linewidth(0.1)
    axis.spines['top'].set_alpha(0.5)
    axis.spines['right'].set_linewidth(0.1)
    axis.spines['right'].set_alpha(0.5)
    axis.xaxis.set_ticks_position('bottom')
    axis.yaxis.set_ticks_position('left')

    axis.set_xticks(np.arange(0, 121, 30))
    yticks = np.arange(0, 71, 35)
    axis.set_yticks(yticks)
    axis.set_yticklabels(['{0}'.format(tick) for tick in yticks[::-1]])

    axis.set_xlabel(r'distance [m]')
    axis.set_ylabel(r'distance [m]', labelpad=2)
    if cbar is not None:
        cbar.set_label(r'altitude [m]')

        cbar.set_ticks(np.arange(0, 36, 10))

        for spine in cbar.ax.spines.itervalues():
            spine.set_linewidth(0.1)
        cbar.ax.yaxis.set_tick_params(color=emulate_color('k', 0.7))

    plt.tight_layout(pad=0.1)


def emulate_color(color, alpha=1, background_color=(1, 1, 1)):
    """Take an RGBA color and an RGB background, return the emulated RGB color.

    The RGBA color with transparency alpha is converted to an RGB color via
    emulation in front of the background_color.
    """
    to_rgb = ColorConverter().to_rgb
    color = to_rgb(color)
    background_color = to_rgb(background_color)
    return [(1 - alpha) * bg_col + alpha * col
            for col, bg_col in zip(color, background_color)]


def plot_paper(altitudes, S_hat, world_shape, fileName, surf=False,
    coord=np.array([])):
    tw = cw = 13.968
    # tw = 17.14256
    cmap = 'jet'
    alpha = 1.
    alpha_world = 0.25
    size_wb = np.array([cw / 2.2, tw / 4.])
    #size_wb = np.array([cw / 4.2, cw / 4.2])

    altitudes -= np.nanmin(altitudes)
    vmin, vmax = (np.nanmin(altitudes), np.nanmax(altitudes))
    origin = 'lower'

    fig = paper_figure(size_wb)

    altitudes2 = altitudes.copy()
    altitudes2[~S_hat[:, 0]] = np.nan

    if not surf:
        axis = fig.gca()
        c = axis.imshow(np.reshape(altitudes, world_shape).T, origin=origin, vmin=vmin,
                        vmax=vmax, cmap=cmap, alpha=alpha_world)

        cbar = plt.colorbar(c)
        #cbar = None
        plt.imshow(np.reshape(altitudes2, world_shape).T, origin=origin, vmin=vmin,
                   vmax=vmax, interpolation='nearest', cmap=cmap, alpha=alpha)
        format_figure(axis, cbar)
        plt.savefig(fileName, transparent=False, format="pdf")
        plt.show()
    elif coord.size > 0:
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(coord[:, 0], coord[:, 1], altitudes, alpha=alpha_world)
        ax.plot_trisurf(coord[:, 0], coord[:, 1], altitudes2, alpha=alpha)
        plt.show()
    else:
        raise ValueError("Coordinates are needed for 3D plots")
