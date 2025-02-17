"""Weightings according to [IEC 61672-1:2003](https://webstore.iec.ch/en/publication/19903).

Attributes:
    THIRD_OCTAVE_A_WEIGHTING: A-weighting filter for preferred 1/3-octave band center frequencies.
    THIRD_OCTAVE_C_WEIGHTING: C-weighting filter for preferred 1/3-octave band center frequencies.

Functions:
    a_weighting: A-weighting.
    c_weighting: C-weighting.
    z2a: Apply A-weighting to Z-weighted signal.
    a2z: Remove A-weighting from A-weighted signal.
    z2c: Apply C-weighting to Z-weighted signal.
    c2z: Remove C-weighting from C-weighted signal.
"""

import numpy as np

from acoustic_toolbox.bands import third

THIRD_OCTAVE_A_WEIGHTING = np.array(
    [
        -63.4,
        -56.7,
        -50.5,
        -44.7,
        -39.4,
        -34.6,
        -30.2,
        -26.2,
        -22.5,
        -19.1,
        -16.1,
        -13.4,
        -10.9,
        -8.6,
        -6.6,
        -4.8,
        -3.2,
        -1.9,
        -0.8,
        +0.0,
        +0.6,
        +1.0,
        +1.2,
        +1.3,
        +1.2,
        +1.0,
        +0.5,
        -0.1,
        -1.1,
        -2.5,
        -4.3,
        -6.6,
        -9.3,
    ]
)
"""A-weighting filter for preferred 1/3-octave band center frequencies, as specified in [`acoustic_toolbox.bands.THIRD_OCTAVE_CENTER_FREQUENCIES`][acoustic_toolbox.bands.THIRD_OCTAVE_CENTER_FREQUENCIES]."""

THIRD_OCTAVE_C_WEIGHTING = np.array(
    [
        -11.2,
        -8.5,
        -6.2,
        -4.4,
        -3.0,
        -2.0,
        -1.3,
        -0.8,
        -0.5,
        -0.3,
        -0.2,
        -0.1,
        +0.0,
        +0.0,
        +0.0,
        +0.0,
        +0.0,
        +0.0,
        +0.0,
        +0.0,
        +0.0,
        -0.1,
        -0.2,
        -0.3,
        -0.5,
        -0.8,
        -1.3,
        -2.0,
        -3.0,
        -4.4,
        -6.2,
        -8.5,
        -11.2,
    ]
)
"""C-weighting filter for preferred 1/3-octave band center frequencies, as specified in [`acoustic_toolbox.bands.THIRD_OCTAVE_CENTER_FREQUENCIES`][acoustic_toolbox.bands.THIRD_OCTAVE_CENTER_FREQUENCIES]."""


def a_weighting(first: float, last: float) -> np.ndarray:
    """Select frequency weightings between ``first`` and ``last`` centerfrequencies from A-weighting.

    Possible values for these frequencies are third-octave frequencies
    between 12.5 Hz and 20,000 Hz (including them).

    Args:
      first: First third-octave centerfrequency.
      last: Last third-octave centerfrequency.

    Returns:
        A-weighting between `first` and `last` center frequencies.
    """
    return _weighting("a", first, last)


def c_weighting(first: float, last: float) -> np.ndarray:
    """Select frequency weightings between ``first`` and ``last`` centerfrequencies from C-weighting.

    Possible values for these frequencies are third-octave frequencies
    between 12.5 Hz and 20,000 Hz (including them).

    Args:
      first: First third-octave centerfrequency.
      last: Last third-octave centerfrequency.

    Returns:
        C-weighting between `first` and `last` center frequencies.
    """
    return _weighting("c", first, last)


def _weighting(filter_type: str, first: float, last: float) -> np.ndarray:
    third_oct_bands = third(12.5, 20000.0).tolist()
    low = third_oct_bands.index(first)
    high = third_oct_bands.index(last)

    if filter_type == "a":
        freq_weightings = THIRD_OCTAVE_A_WEIGHTING

    elif filter_type == "c":
        freq_weightings = THIRD_OCTAVE_C_WEIGHTING

    return freq_weightings[low : high + 1]


def z2a(levels: np.ndarray, first: float, last: float) -> np.ndarray:
    """Apply A-weighting to Z-weighted signal.

    Args:
      levels: Z-weighted signal.
      first: First third-octave centerfrequency.
      last: Last third-octave centerfrequency.

    Returns:
        A-weighted signal.
    """
    return levels + a_weighting(first, last)


def a2z(levels: np.ndarray, first: float, last: float) -> np.ndarray:
    """Remove A-weighting from A-weighted signal.

    Args:
      levels: A-weighted signal.
      first: First third-octave centerfrequency.
      last: Last third-octave centerfrequency.

    Returns:
        Z-weighted signal.
    """
    return levels - a_weighting(first, last)


def z2c(levels: np.ndarray, first: float, last: float) -> np.ndarray:
    """Apply C-weighting to Z-weighted signal.

    Args:
      levels: Z-weighted signal.
      first: First third-octave centerfrequency.
      last: Last third-octave centerfrequency.

    Returns:
        C-weighted signal.
    """
    return levels + c_weighting(first, last)


def c2z(levels: np.ndarray, first: float, last: float) -> np.ndarray:
    """Remove C-weighting from C-weighted signal.

    Args:
      levels: C-weighted signal.
      first: First third-octave centerfrequency.
      last: Last third-octave centerfrequency.

    Returns:
        Z-weighted signal.
    """
    return levels - c_weighting(first, last)


def a2c(levels: np.ndarray, first: float, last: float) -> np.ndarray:
    """Go from A-weighted to C-weighted.

    Args:
      levels: A-weighted signal.
      first: First third-octave centerfrequency.
      last: Last third-octave centerfrequency.

    Returns:
        C-weighted signal.
    """
    dB = a2z(levels, first, last)
    return z2c(dB, first, last)


def c2a(levels: np.ndarray, first: float, last: float) -> np.ndarray:
    """Go from C-weighted to A-weighted.

    Args:
      levels: C-weighted signal.
      first: First third-octave centerfrequency.
      last: Last third-octave centerfrequency.

    Returns:
        A-weighted signal.
    """
    dB = c2z(levels, first, last)
    return z2a(dB, first, last)


__all__ = [
    "THIRD_OCTAVE_A_WEIGHTING",
    "THIRD_OCTAVE_C_WEIGHTING",
    "a_weighting",
    "c_weighting",
    "z2a",
    "a2z",
    "z2c",
    "c2z",
    "a2c",
    "c2a",
]
