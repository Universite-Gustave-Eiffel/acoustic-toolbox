"""Plotting functions using [`matplotlib`](https://matplotlib.org/) library.

Warning:
   You need to have [`matplotlib`](https://matplotlib.org/) installed in order to use this module.
"""

from typing import Literal
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import NullLocator, FixedLocator
from matplotlib.ticker import ScalarFormatter, NullFormatter
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

from acoustic_toolbox.bands import octave, third


class OctaveBandScale(mscale.ScaleBase):
    """Octave band scale."""

    name = "octave"

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis)

    def get_transform(self):
        return self.BandTransform()

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(FixedLocator(octave(16, 16000)))
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax

    class BandTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = False

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            return np.log10(a + 1)


mscale.register_scale(OctaveBandScale)


class ThirdBandScale(mscale.ScaleBase):
    """Third-octave band scale."""

    name = "third"

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis)

    def get_transform(self):
        return self.BandTransform()

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(NullLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(FixedLocator(third(12.5, 20000)))
        axis.set_minor_formatter(ScalarFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax

    class BandTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = False

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            return np.log10(a + 1)


mscale.register_scale(ThirdBandScale)


def plot_octave(
    data: np.ndarray,
    octaves: np.ndarray,
    axes: Axes | None = None,
    kHz: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    separator: str | None = None,
    *args,
    **kwargs,
) -> list[Line2D]:
    """Plot octave bands from `data` levels and `octaves` bands.

    Args:
      data: levels in a 1-D NumPy array.
      octaves: octaves in a 1-D NumPy array.
        Note that you can use [`acoustic_toolbox.bands.octave`][acoustic_toolbox.bands.octave]
        for this or manually enter all bands.
      axes: a `matplotlib.axes` object.
      kHz: if `True` it shows "1k" or "2k" instead of "1000" or "2000" as
      tick labels.
      xlabel: a `str` containing label for x axis.
      ylabel: a `str` containing label for y axis.
      title: a `str` containing title.
      separator: a `str` defining the decimal separator.  By default takes '.'
        or ',' values according to system settings (when separator is None).
      *args:
      **kwargs:

    Returns:
      A list of [`matplotlib.lines.Line2D`][matplotlib.lines.Line2D] objects.

    See Also:
      [`matplotlib.axes.Axes.plot`][matplotlib.axes.Axes.plot]
    """
    band_type = "octave"
    k_ticks = kHz
    return plot_bands(
        data,
        octaves,
        axes,
        band_type,
        k_ticks,
        xlabel,
        ylabel,
        title,
        separator,
        *args,
        **kwargs,
    )


def plot_third(
    data: np.ndarray,
    thirds: np.ndarray,
    axes: Axes | None = None,
    kHz: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    separator: str | None = None,
    *args,
    **kwargs,
) -> list[Line2D]:
    """Plot third octave bands from `data` levels and `thirds` bands.

    Args:
      data: levels in an 1-D NumPy array.
      thirds: thirds in an 1-D NumPy array. Note that you can use
        [`acoustic_toolbox.bands.third`][acoustic_toolbox.bands.third] for this
        or manually enter all bands.
      axes: a `matplotlib.axes` object.
      kHz: if `True` it shows "1k" or "2.5k" instead of "1000" or "2500" as
        tick labels.
      xlabel: a `str` containing label for x axis (optional).
      ylabel: a `str` containing label for y axis (optional).
      title: a `str` containing title (optional).
      separator: a `str` defining the decimal separator. By default takes '.'
        or ',' values according to system settings (when separator is None).
      *args:
      **kwargs:

    Returns:
      A list of [`matplotlib.lines.Line2D`][matplotlib.lines.Line2D] objects.

    See Also:
      [`matplotlib.axes.Axes.plot`][matplotlib.axes.Axes.plot]
    """
    band_type = "third"
    k_ticks = kHz
    return plot_bands(
        data,
        thirds,
        axes,
        band_type,
        k_ticks,
        xlabel,
        ylabel,
        title,
        separator,
        *args,
        **kwargs,
    )


def plot_bands(
    data: np.ndarray,
    bands: np.ndarray,
    axes: Axes,
    band_type: Literal["octave", "third"],
    k_ticks: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    separator: str | None = None,
    *args,
    **kwargs,
) -> list[Line2D]:
    """Plot bands from `data` levels and `bands`.

    Only use if you want to plot from arbitrary octave or third octave data.

    Args:
      data: levels in an 1-D NumPy array.
      bands: bands in an 1-D NumPy array.
      axes: `matplotlib.axes` object.
      band_type: `'octave'` or `'third'` are accepted values.
      k_ticks: if `True` it shows "1k" or "2.5k" instead of "1000" or "2500" as
        tick labels.
      xlabel: label for x axis.
      ylabel: label for y axis.
      title: title.
      separator: decimal separator. By default takes '.' or ',' values
        according to system settings (when separator is None).
      *args: additional arguments for `matplotlib.axes.plot`.
      **kwargs: additional arguments for `matplotlib.axes.plot`.

    Returns:
      A list of [`matplotlib.lines.Line2D`][matplotlib.lines.Line2D] objects.

    See Also:
      [`matplotlib.axes.Axes.plot`][matplotlib.axes.Axes.plot]
    """
    if axes is None:
        axes = plt.gca()
    factor = 0.1
    min_auto = bands[0] * (1 - factor)
    max_auto = bands[-1] * (1 + factor)
    axes.set_xlim([min_auto, max_auto])
    axes.set_xscale(band_type)

    # Set tick labels.
    ticklabels = _get_ticklabels(band_type, k_ticks, separator)
    if band_type == "third":
        third_ticks = True
    elif band_type == "octave":
        third_ticks = False
    axes.set_xticklabels(ticklabels, minor=third_ticks)

    # Set x and y labels and title.
    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if title is not None:
        axes.set_title(title)
    return axes.plot(bands, data, *args, **kwargs)


TICKS_OCTAVE = [
    "16",
    "31.5",
    "63",
    "125",
    "250",
    "500",
    "1000",
    "2000",
    "4000",
    "8000",
    "16000",
]
"""Octave center frequencies as strings."""

TICKS_OCTAVE_KHZ = [
    "16",
    "31.5",
    "63",
    "125",
    "250",
    "500",
    "1k",
    "2k",
    "4k",
    "8k",
    "16k",
]
"""Octave center frequencies as strings. Uses kHz notation."""

TICKS_THIRD_OCTAVE = [
    "12.5",
    "16",
    "20",
    "25",
    "31.5",
    "40",
    "50",
    "63",
    "80",
    "100",
    "125",
    "160",
    "200",
    "250",
    "315",
    "400",
    "500",
    "630",
    "800",
    "1000",
    "1250",
    "1600",
    "2000",
    "2500",
    "3150",
    "4000",
    "5000",
    "6300",
    "8000",
    "10000",
    "12500",
    "16000",
    "20000",
]
"""Third-octave center frequencies as strings."""

TICKS_THIRD_OCTAVE_KHZ = [
    "12.5",
    "16",
    "20",
    "25",
    "31.5",
    "40",
    "50",
    "63",
    "80",
    "100",
    "125",
    "160",
    "200",
    "250",
    "315",
    "400",
    "500",
    "630",
    "800",
    "1000",
    "1250",
    "1600",
    "2000",
    "2500",
    "3150",
    "4000",
    "5000",
    "6300",
    "8000",
    "10000",
    "12500",
    "16000",
    "20000",
]
"""Third-octave center frequencies as strings. Uses kHz notation."""


def _get_ticklabels(band_type, kHz, separator):
    """Return a list with all labels for octave or third-octave band cases."""
    if separator is None:
        import locale

        separator = locale.localeconv()["decimal_point"]

    if band_type == "octave":
        if kHz is True:
            ticklabels = TICKS_OCTAVE_KHZ
        else:
            ticklabels = TICKS_OCTAVE
    else:
        if kHz is True:
            ticklabels = TICKS_THIRD_OCTAVE_KHZ
        else:
            ticklabels = TICKS_THIRD_OCTAVE

    ticklabels = _set_separator(ticklabels, separator)
    return ticklabels


def _set_separator(ticklabels, separator):
    """Set the decimal separator. Note that this is a 'private' function, so you can set the decimal separator directly in plotting functions."""
    if separator == ".":
        bands_sep = ticklabels
    else:
        bands_sep = []
        for item in ticklabels:
            decimal_number_format = item.replace(".", separator)
            bands_sep.append(decimal_number_format)
    return bands_sep
