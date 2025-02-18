"""The bands module contains functions for working with octave and third-octave frequency bands, based on [IEC 61672-1:2013](standards/iec_61672_1_2013.md).

Functions:
    octave: Generate center frequencies for octave bands.
    octave_low: Calculate lower corner frequencies for octave bands.
    octave_high: Calculate upper corner frequencies for octave bands.
    third: Generate center frequencies for third-octave bands.
    third_low: Calculate lower corner frequencies for third-octave bands.
    third_high: Calculate upper corner frequencies for third-octave bands.
    third2oct: Convert third-octave band levels to octave band levels.
"""

import numpy as np
from typing import Literal
from numpy.typing import NDArray

# from acoustic_toolbox.decibel import dbsum
import acoustic_toolbox
from acoustic_toolbox.standards.iec_61672_1_2013 import (
    NOMINAL_OCTAVE_CENTER_FREQUENCIES,
    NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
)

OCTAVE_CENTER_FREQUENCIES = NOMINAL_OCTAVE_CENTER_FREQUENCIES
"""Preferred nominal octave band center frequencies."""

THIRD_OCTAVE_CENTER_FREQUENCIES = NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES
"""Preferred nominal third-octave band center frequencies."""


def octave(first: float, last: float) -> NDArray[np.float64]:
    """Generate a Numpy array for central frequencies of octave bands.

    There are more information on how to calculate 'real' bands in
    http://blog.prosig.com/2006/02/17/standard-octave-bands/

    Args:
        first: First octave centerfrequency.
        last: Last octave centerfrequency.

    Returns:
        Array of octave band center frequencies in Hz.
    """
    # octave_bands = OCTAVE_CENTER_FREQUENCIES
    # low = np.where(octave_bands == first)[0]
    # high = np.where(octave_bands == last)[0]
    # return octave_bands[low: high+1]
    return acoustic_toolbox.signal.OctaveBand(
        fstart=first, fstop=last, fraction=1
    ).nominal


def octave_low(first: float, last: float) -> NDArray[np.float64]:
    """Lower cornerfrequencies of octaves.

    Args:
        first: First octave centerfrequency.
        last: Last octave centerfrequency.

    Returns:
        Array of lower corner frequencies in Hz.
    """
    return octave(first, last) / np.sqrt(2.0)
    # return acoustic_toolbox.signal.OctaveBand(fstart=first, fstop=last, fraction=1).lower


def octave_high(first: float, last: float) -> NDArray[np.float64]:
    """Upper cornerfrequencies of octaves.

    Args:
        first: First octave centerfrequency.
        last: Last octave centerfrequency.

    Returns:
        Array of upper corner frequencies in Hz.
    """
    return octave(first, last) * np.sqrt(2.0)
    # return acoustic_toolbox.signal.OctaveBand(fstart=first, fstop=last, fraction=1).upper


def third(first: float, last: float) -> NDArray[np.float64]:
    """Generate a Numpy array for central frequencies of third octave bands.

    Args:
        first: First third octave centerfrequency.
        last: Last third octave centerfrequency.

    Returns:
        Array of third octave band center frequencies in Hz.
    """
    # third_oct_bands = THIRD_OCTAVE_CENTER_FREQUENCIES
    # low = np.where(third_oct_bands == first)[0]
    # high = np.where(third_oct_bands == last)[0]
    # return third_oct_bands[low: high+1]
    return acoustic_toolbox.signal.OctaveBand(
        fstart=first, fstop=last, fraction=3
    ).nominal


def third_low(first: float, last: float) -> NDArray[np.float64]:
    """Lower cornerfrequencies of third-octaves.

    Args:
        first: First third octave centerfrequency.
        last: Last third octave centerfrequency.

    Returns:
        Array of lower corner frequencies in Hz.
    """
    return third(first, last) / 2.0 ** (1.0 / 6.0)
    # return acoustic_toolbox.signal.OctaveBand(fstart=first, fstop=last, fraction=3).lower


def third_high(first: float, last: float) -> NDArray[np.float64]:
    """Higher cornerfrequencies of third-octaves.

    Args:
        first: First third octave centerfrequency.
        last: Last third octave centerfrequency.

    Returns:
        Array of upper corner frequencies in Hz.
    """
    return third(first, last) * 2.0 ** (1.0 / 6.0)
    # return Octaveband(fstart=first, fstop=last, fraction=3).upper


def third2oct(
    levels: NDArray[np.float64], axis: int | None = None
) -> NDArray[np.float64]:
    """Calculate Octave levels from third octave levels.

    Args:
        levels: Array containing third octave levels.
        axis: Axis over which to perform the summation.

    Returns:
        Array containing octave band levels.

    Note:
        The number of elements along the summation axis should be a factor of 3. : Third octave levels

    Raises:
        ValueError: If the shape of levels array is not compatible with the operation.
    """
    levels = np.array(levels)
    axis = axis if axis is not None else levels.ndim - 1

    try:
        assert levels.shape[axis] % 3 == 0
    except AssertionError:
        raise ValueError("Wrong shape.")
    shape = list(levels.shape)
    shape[axis] = shape[axis] // 3
    shape.insert(axis + 1, 3)
    levels = np.reshape(levels, shape)
    return np.squeeze(acoustic_toolbox.decibel.dbsum(levels, axis=axis + 1))


def _check_band_type(
    freqs: NDArray[np.float64],
) -> Literal["octave", "octave-unsorted", "third", "third-unsorted"] | None:
    """Check if an array contains octave or third octave bands values sorted or unsorted.

    Args:
        freqs: Array of frequencies to check.

    Returns:
        The type of band ("octave", "octave-unsorted", "third", "third-unsorted") or None if not recognized.
    """
    octave_bands = octave(16, 16000)
    third_oct_bands = third(12.5, 20000)

    def _check_sort(freqs: NDArray[np.float64], bands: NDArray[np.float64]) -> bool:
        """Check if frequencies are in sorted order within the bands.

        Args:
            freqs: Array of frequencies to check.
            bands: Reference band frequencies.

        Returns:
            bool: True if frequencies are in sorted order.
        """
        index = np.where(np.isin(bands, freqs))[0]
        band_pos = index - index[0]
        return (band_pos == np.arange(band_pos.size)).all()

    if np.isin(freqs, octave_bands).all():
        is_sorted = _check_sort(freqs, octave_bands)
        return "octave" if is_sorted else "octave-unsorted"
    elif np.isin(freqs, third_oct_bands).all():
        is_sorted = _check_sort(freqs, third_oct_bands)
        return "third" if is_sorted else "third-unsorted"
    else:
        return None
