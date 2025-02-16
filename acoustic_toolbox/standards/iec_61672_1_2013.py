"""IEC 61672-1:2013

This module implements IEC 61672-1:2013 which provides electroacoustical performance specifications
for three kinds of sound measuring instruments:

1. Time-weighting sound level meters that measure exponential-time-weighted, frequency-weighted sound levels
2. Integrating-averaging sound level meters that measure time-averaged, frequency-weighted sound levels
3. Integrating sound level meters that measure frequency-weighted sound exposure levels

The module provides functions for:
- Frequency weighting (A, C, Z)
- Time weighting (Fast, Slow)
- Sound level calculations

Reference:
    IEC 61672-1:2013: http://webstore.iec.ch/webstore/webstore.nsf/artnum/048669
"""

import io
import os
import pkgutil
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from numpy.typing import NDArray
from scipy.signal import zpk2tf
from scipy.signal import lfilter, bilinear
from .iso_tr_25417_2007 import REFERENCE_PRESSURE


WEIGHTING_DATA = pd.read_csv(
    io.BytesIO(
        pkgutil.get_data(
            "acoustic_toolbox", os.path.join("data", "iec_61672_1_2013.csv")
        )
    ),
    sep=",",
    index_col=0,
)
"""DataFrame with indices, nominal frequencies and weighting values."""

NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES: NDArray[np.float64] = np.array(
    WEIGHTING_DATA.nominal
)
"""Nominal 1/3-octave frequencies. See table 3."""

NOMINAL_OCTAVE_CENTER_FREQUENCIES: NDArray[np.float64] = np.array(
    WEIGHTING_DATA.nominal
)[2::3]
"""Nominal 1/1-octave frequencies. Based on table 3."""

REFERENCE_FREQUENCY: float = 1000.0
"""Reference frequency. See table 3."""

EXACT_THIRD_OCTAVE_CENTER_FREQUENCIES: NDArray[np.float64] = (
    REFERENCE_FREQUENCY * 10.0 ** (0.01 * (np.arange(10, 44) - 30))
)
"""Exact third-octave center frequencies. See table 3."""

WEIGHTING_A: NDArray[np.float64] = np.array(WEIGHTING_DATA.A)
"""Frequency weighting A. See table 3."""

WEIGHTING_C: NDArray[np.float64] = np.array(WEIGHTING_DATA.C)
"""Frequency weighting C. See table 3."""

WEIGHTING_Z: NDArray[np.float64] = np.array(WEIGHTING_DATA.Z)
"""Frequency weighting Z. See table 3."""

WEIGHTING_VALUES: Dict[str, NDArray[np.float64]] = {
    "A": WEIGHTING_A,
    "C": WEIGHTING_C,
    "Z": WEIGHTING_Z,
}
"""Dictionary with weighting values 'A', 'C' and 'Z' weighting."""

FAST: float = 0.125
"""FAST time-constant."""

SLOW: float = 1.000
"""SLOW time-constant."""


def time_averaged_sound_level(
    pressure: NDArray[np.float64],
    sample_frequency: float,
    averaging_time: float,
    reference_pressure: float = REFERENCE_PRESSURE,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate time-averaged sound pressure level.

    Args:
        pressure: Dynamic pressure.
        sample_frequency: Sample frequency in Hz.
        averaging_time: Averaging time in seconds.
        reference_pressure: Reference pressure. Defaults to REFERENCE_PRESSURE.

    Returns:
        Tuple containing:
            - Time points in seconds
            - Time-averaged sound pressure levels in dB
    """
    levels = 10.0 * np.log10(
        average(pressure**2.0, sample_frequency, averaging_time)
        / reference_pressure**2.0
    )
    times = np.arange(levels.shape[-1]) * averaging_time
    return times, levels


def average(
    data: NDArray[np.float64], sample_frequency: float, averaging_time: float
) -> NDArray[np.float64]:
    """Average the sound pressure squared.

    Args:
        data: Energetic quantity, e.g. p².
        sample_frequency: Sample frequency in Hz.
        averaging_time: Averaging time in seconds.

    Returns:
        Averaged data with time weighting applied using a low-pass filter
        with one real pole at -1/τ.

    Note:
        Because fs·ti is generally not an integer, samples are discarded.
        This results in a drift of samples for longer signals (e.g. 60 minutes at 44.1 kHz).
    """
    averaging_time = np.asarray(averaging_time)
    sample_frequency = np.asarray(sample_frequency)
    samples = data.shape[-1]
    n = np.floor(averaging_time * sample_frequency).astype(int)
    data = data[..., 0 : n * (samples // n)]  # Drop the tail of the signal.
    newshape = list(data.shape[0:-1])
    newshape.extend([-1, n])
    data = data.reshape(newshape)
    # data = data.reshape((-1, n))
    return data.mean(axis=-1)


def time_weighted_sound_level(
    pressure: NDArray[np.float64],
    sample_frequency: float,
    integration_time: float,
    reference_pressure: float = REFERENCE_PRESSURE,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate time-weighted sound pressure level.

    Args:
        pressure: Dynamic pressure.
        sample_frequency: Sample frequency in Hz.
        integration_time: Integration time in seconds.
        reference_pressure: Reference pressure. Defaults to REFERENCE_PRESSURE.

    Returns:
        Tuple containing:
            - Time points in seconds
            - Time-weighted sound pressure levels in dB
    """
    levels = 10.0 * np.log10(
        integrate(pressure**2.0, sample_frequency, integration_time)
        / reference_pressure**2.0
    )
    times = np.arange(levels.shape[-1]) * integration_time
    return times, levels


def integrate(
    data: NDArray[np.float64], sample_frequency: float, integration_time: float
) -> NDArray[np.float64]:
    """Integrate the sound pressure squared using exponential integration.

    Args:
        data: Energetic quantity, e.g. p².
        sample_frequency: Sample frequency in Hz.
        integration_time: Integration time in seconds.

    Returns:
        Integrated data with time weighting applied using a low-pass filter
        with one real pole at -1/τ.

    Note:
        Because fs·ti is generally not an integer, samples are discarded.
        This results in a drift of samples for longer signals (e.g. 60 minutes at 44.1 kHz).
    """
    samples = data.shape[-1]
    b, a = zpk2tf([1.0], [1.0, integration_time], [1.0])
    b, a = bilinear(b, a, fs=sample_frequency)
    # b, a = bilinear([1.0], [1.0, integration_time], fs=sample_frequency) # Bilinear: Analog to Digital filter.
    n = np.floor(integration_time * sample_frequency).astype(int)
    data = data[..., 0 : n * (samples // n)]
    newshape = list(data.shape[0:-1])
    newshape.extend([-1, n])
    data = data.reshape(newshape)
    # data = data.reshape((-1, n)) # Divide in chunks over which to perform the integration.
    return (
        lfilter(b, a, data)[..., n - 1] / integration_time
    )  # Perform the integration. Select the final value of the integration.


def fast(data: NDArray[np.float64], fs: float) -> NDArray[np.float64]:
    """Apply fast (F) time-weighting.

    Args:
        data: Energetic quantity, e.g. p².
        fs: Sample frequency in Hz.

    Returns:
        Data with FAST time-weighting applied.

    See Also:
        integrate: Base integration function used for time-weighting.
    """
    return integrate(data, fs, FAST)
    # return time_weighted_sound_level(data, fs, FAST)


def slow(data: NDArray[np.float64], fs: float) -> NDArray[np.float64]:
    """Apply slow (S) time-weighting.

    Args:
        data: Energetic quantity, e.g. p².
        fs: Sample frequency in Hz.

    Returns:
        Data with SLOW time-weighting applied.

    See Also:
        integrate: Base integration function used for time-weighting.
    """
    return integrate(data, fs, SLOW)
    # return time_weighted_sound_level(data, fs, SLOW)


def fast_level(
    data: NDArray[np.float64], fs: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate time-weighted (FAST) sound pressure level.

    Args:
        data: Dynamic pressure.
        fs: Sample frequency in Hz.

    Returns:
        Tuple containing:
            - Time points in seconds
            - FAST time-weighted sound pressure levels in dB

    See Also:
        time_weighted_sound_level: Base function for calculating time-weighted levels.
    """
    return time_weighted_sound_level(data, fs, FAST)


def slow_level(
    data: NDArray[np.float64], fs: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate time-weighted (SLOW) sound pressure level.

    Args:
        data: Dynamic pressure.
        fs: Sample frequency in Hz.

    Returns:
        Tuple containing:
            - Time points in seconds
            - SLOW time-weighted sound pressure levels in dB

    See Also:
        time_weighted_sound_level: Base function for calculating time-weighted levels.
    """
    return time_weighted_sound_level(data, fs, SLOW)


# ---- Annex E - Analytical expressions for frequency-weightings C, A, and Z.-#

_POLE_FREQUENCIES: Dict[int, float] = {
    1: 20.60,
    2: 107.7,
    3: 737.9,
    4: 12194.0,
}
"""Approximate values for pole frequencies f_1, f_2, f_3 and f_4.

See section E.4.1 of the standard.
"""

_NORMALIZATION_CONSTANTS: Dict[str, float] = {
    "A": -2.000,
    "C": -0.062,
}
"""Normalization constants $C_{1000}$ and $A_{1000}$.

See section E.4.2 of the standard.
"""


def weighting_function_a(
    frequencies: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Calculate A-weighting function in decibels.

    Args:
        frequencies: Vector of frequencies at which to evaluate the weighting.

    Returns:
        Vector with A-weighting scaling factors in dB, calculated using
        equation E.6 from the standard with A₁₀₀₀ = -2 dB.

    Note:
        Implementation of equation E.6 from the standard.
    """
    f = np.asarray(frequencies)
    offset = _NORMALIZATION_CONSTANTS["A"]
    f1, f2, f3, f4 = _POLE_FREQUENCIES.values()
    weighting = (
        20.0
        * np.log10(
            (f4**2.0 * f**4.0)
            / (
                (f**2.0 + f1**2.0)
                * np.sqrt(f**2.0 + f2**2.0)
                * np.sqrt(f**2.0 + f3**2.0)
                * (f**2.0 + f4**2.0)
            )
        )
        - offset
    )
    return weighting


def weighting_function_c(
    frequencies: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Calculate C-weighting function in decibels.

    Args:
        frequencies: Vector of frequencies at which to evaluate the weighting.

    Returns:
        Vector with C-weighting scaling factors in dB, calculated using
        equation E.1 from the standard with C₁₀₀₀ = -0.062 dB.

    Note:
        Implementation of equation E.1 from the standard.
    """
    f = np.asarray(frequencies)
    offset = _NORMALIZATION_CONSTANTS["C"]
    f1, _, _, f4 = _POLE_FREQUENCIES.values()
    weighting = (
        20.0 * np.log10((f4**2.0 * f**2.0) / ((f**2.0 + f1**2.0) * (f**2.0 + f4**2.0)))
        - offset
    )
    return weighting


def weighting_function_z(
    frequencies: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Calculate Z-weighting function in decibels.

    Args:
        frequencies: Vector of frequencies at which to evaluate the weighting.

    Returns:
        Vector of zeros (Z-weighting applies no frequency weighting).
    """
    frequencies = np.asarray(frequencies)
    return np.zeros_like(frequencies)


WEIGHTING_FUNCTIONS: Dict[str, callable] = {
    "A": weighting_function_a,
    "C": weighting_function_c,
    "Z": weighting_function_z,
}
"""Dictionary with available weighting functions 'A', 'C' and 'Z'."""


def weighting_system_a() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get A-weighting filter as polynomial transfer function.

    Returns:
        Tuple containing:
            - Numerator coefficients of the transfer function
            - Denominator coefficients of the transfer function

    Note:
        Implementation of equation E.6 from the standard.
    """
    f1 = _POLE_FREQUENCIES[1]
    f2 = _POLE_FREQUENCIES[2]
    f3 = _POLE_FREQUENCIES[3]
    f4 = _POLE_FREQUENCIES[4]
    offset = _NORMALIZATION_CONSTANTS["A"]
    numerator = np.array(
        [(2.0 * np.pi * f4) ** 2.0 * (10 ** (-offset / 20.0)), 0.0, 0.0, 0.0, 0.0]
    )
    part1 = [1.0, 4.0 * np.pi * f4, (2.0 * np.pi * f4) ** 2.0]
    part2 = [1.0, 4.0 * np.pi * f1, (2.0 * np.pi * f1) ** 2.0]
    part3 = [1.0, 2.0 * np.pi * f3]
    part4 = [1.0, 2.0 * np.pi * f2]
    denomenator = np.convolve(np.convolve(np.convolve(part1, part2), part3), part4)
    return numerator, denomenator


def weighting_system_c() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get C-weighting filter as polynomial transfer function.

    Returns:
        Tuple containing:
            - Numerator coefficients of the transfer function
            - Denominator coefficients of the transfer function

    Note:
        Implementation of equation E.1 from the standard.
    """
    f1 = _POLE_FREQUENCIES[1]
    f4 = _POLE_FREQUENCIES[4]
    offset = _NORMALIZATION_CONSTANTS["C"]
    numerator = np.array(
        [(2.0 * np.pi * f4) ** 2.0 * (10 ** (-offset / 20.0)), 0.0, 0.0]
    )
    part1 = [1.0, 4.0 * np.pi * f4, (2.0 * np.pi * f4) ** 2.0]
    part2 = [1.0, 4.0 * np.pi * f1, (2.0 * np.pi * f1) ** 2.0]
    denomenator = np.convolve(part1, part2)
    return numerator, denomenator


def weighting_system_z() -> Tuple[list[float], list[float]]:
    """Get Z-weighting filter as polynomial transfer function.

    Returns:
        Tuple containing:
            - [1] (numerator for unity gain)
            - [1] (denominator for unity gain)

    Note:
        Z-weighting applies no frequency weighting (0.0 dB at all frequencies),
        therefore corresponds to multiplication by 1.
    """
    numerator = [1]
    denomenator = [1]
    return numerator, denomenator


WEIGHTING_SYSTEMS: Dict[str, callable] = {
    "A": weighting_system_a,
    "C": weighting_system_c,
    "Z": weighting_system_z,
}
"""Dictionary with available weighting systems 'A', 'C' and 'Z'."""
