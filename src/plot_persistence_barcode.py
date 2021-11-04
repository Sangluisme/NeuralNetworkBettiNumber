import matplotlib.pyplot as plt
import numpy as np

def _array_handler(a):
    '''
    :param a: if array, assumes it is a (n x 2) np.array and return a
                persistence-compatible list (padding with 0), so that the
                plot can be performed seamlessly.
    '''
    if isinstance(a[0][1], np.float64) or isinstance(a[0][1], float):
        return [[0, x] for x in a]
    else:
        return a


def __min_birth_max_death(persistence, band=0.0):
    """This function returns (min_birth, max_death) from the persistence.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param band: band
    :type band: float.
    :returns: (float, float) -- (min_birth, max_death).
    """
    # Look for minimum birth date and maximum death date for plot optimisation
    max_death = 0
    min_birth = persistence[0][1][0]
    for interval in reversed(persistence):
        if float(interval[1][1]) != float("inf"):
            if float(interval[1][1]) > max_death:
                max_death = float(interval[1][1])
        if float(interval[1][0]) > max_death:
            max_death = float(interval[1][0])
        if float(interval[1][0]) < min_birth:
            min_birth = float(interval[1][0])
    if band > 0.0:
        max_death += band
    return (min_birth, max_death)

def plot_persistence_barcode(
    persistence=[],
    persistence_file="",
    alpha=0.6,
    max_intervals=1000,
    max_barcodes=1000,
    inf_delta=0.1,
    legend=False,
    colormap=None,
    axes=None,
    fontsize=16,
):
    """This function plots the persistence bar code from persistence values list
    , a np.array of shape (N x 2) (representing a diagram 
    in a single homology dimension), 
    or from a `persistence diagram <fileformats.html#persistence-diagram>`_ file.

    :param persistence: Persistence intervals values list. Can be grouped by dimension or not.
    :type persistence: an array of (dimension, array of (birth, death)) or an array of (birth, death).
    :param persistence_file: A `persistence diagram <fileformats.html#persistence-diagram>`_ file style name
        (reset persistence if both are set).
    :type persistence_file: string
    :param alpha: barcode transparency value (0.0 transparent through 1.0
        opaque - default is 0.6).
    :type alpha: float.
    :param max_intervals: maximal number of intervals to display.
        Selected intervals are those with the longest life time. Set it
        to 0 to see all. Default value is 1000.
    :type max_intervals: int.
    :param inf_delta: Infinity is placed at :code:`((max_death - min_birth) x
        inf_delta)` above :code:`max_death` value. A reasonable value is
        between 0.05 and 0.5 - default is 0.1.
    :type inf_delta: float.
    :param legend: Display the dimension color legend (default is False).
    :type legend: boolean.
    :param colormap: A matplotlib-like qualitative colormaps. Default is None
        which means :code:`matplotlib.cm.Set1.colors`.
    :type colormap: tuple of colors (3-tuple of float between 0. and 1.).
    :param axes: A matplotlib-like subplot axes. If None, the plot is drawn on
        a new set of axes.
    :type axes: `matplotlib.axes.Axes`
    :param fontsize: Fontsize to use in axis.
    :type fontsize: int
    :returns: (`matplotlib.axes.Axes`): The axes on which the plot was drawn.
    """

    persistence = _array_handler(persistence)

    if max_barcodes != 1000:
        print("Deprecated parameter. It has been replaced by max_intervals")
        max_intervals = max_barcodes

    if max_intervals > 0 and max_intervals < len(persistence):
        # Sort by life time, then takes only the max_intervals elements
        persistence = sorted(
            persistence,
            key=lambda life_time: life_time[1][1] - life_time[1][0],
            reverse=True,
        )[:max_intervals]
    
    if colormap == None:
        colormap = plt.cm.Set1.colors
    if axes == None:
        fig, axes = plt.subplots(1, 1)

    persistence = sorted(persistence, key=lambda birth: birth[1][0])

    (min_birth, max_death) = __min_birth_max_death(persistence)
    ind = 0
    delta = (max_death - min_birth) * inf_delta
    # Replace infinity values with max_death + delta for bar code to be more
    # readable
    infinity = max_death + delta
    axis_start = min_birth - delta
    # Draw horizontal bars in loop
    for interval in reversed(persistence):
        if float(interval[1][1]) != float("inf"):
            # Finite death case
            axes.barh(
                ind,
                (interval[1][1] - interval[1][0]),
                height=0.8,
                left=interval[1][0],
                alpha=alpha,
                color=colormap[interval[0]],
                linewidth=0,
            )
        else:
            # Infinite death case for diagram to be nicer
            axes.barh(
                ind,
                (infinity - interval[1][0]),
                height=0.8,
                left=interval[1][0],
                alpha=alpha,
                color=colormap[interval[0]],
                linewidth=0,
            )
        ind = ind + 1

    if legend:
        dimensions = list(set(item[0] for item in persistence))
        axes.legend(
            handles=[
                mpatches.Patch(color=colormap[dim], label=str(dim))
                for dim in dimensions
            ],
            loc="lower right",
        )

    axes.set_title("Persistence barcode", fontsize=fontsize)

    # Ends plot on infinity value and starts a little bit before min_birth
    axes.axis([axis_start, infinity, 0, ind])
    return axes

