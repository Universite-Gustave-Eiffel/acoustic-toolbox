"""IEC 61672-1:2013

This module implements IEC 61672-1:2013 which provides electroacoustical performance specifications
for three kinds of sound measuring instruments:

1. Time-weighting sound level meters that measure exponential-time-weighted, frequency-weighted sound levels
2. Integrating-averaging sound level meters that measure time-averaged, frequency-weighted sound levels
3. Integrating sound level meters that measure frequency-weighted sound exposure levels

The module provides functions for:
- Frequency weighting (A, C, Z)
- Time weighting (Fast, Slow)
- Decibel level calculations

It uses the pyoctaveband package for accurate time and frequency weighting filters.

Reference:
    IEC 61672-1:2013: http://webstore.iec.ch/webstore/webstore.nsf/artnum/048669
    pyoctaveband package: https://pypi.org/project/pyoctaveband/
"""

import io
import os
import pkgutil
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from numpy.typing import NDArray
from scipy.signal import sosfilt, sosfreqz
from .iso_tr_25417_2007 import REFERENCE_PRESSURE

from pyoctaveband import time_weighting, WeightingFilter

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


def time_averaged_level(
    signal: NDArray[np.float64],
    sample_frequency: float,
    integration_time: float,
    reference: float = REFERENCE_PRESSURE,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate time-averaged level.

    Args:
        signal: Dynamic pressure.
        sample_frequency: Sample frequency in Hz.
        integration_time: Integration time in seconds.
        reference: Reference for decibels. Defaults to REFERENCE_PRESSURE.

    Returns:
        Tuple containing:
            - Time points in seconds
            - Time-averaged levels in dB
    """
    signal = np.asarray(signal)
    integration_time = np.asarray(integration_time)
    sample_frequency = np.asarray(sample_frequency)
    signal_samples = signal.shape[-1]

    step = integration_time * sample_frequency
    n_steps = int(signal_samples / step)

    # Calculate end indices for each chunk (exclusive)
    target_boundaries = np.arange(1, n_steps + 1) * step
    end_indices = np.floor(target_boundaries - 1e-4).astype(int) + 1

    # Calculate start indices for reduceat
    start_indices = np.concatenate(([0], end_indices[:-1]))

    # Truncate signal to the end of the last full chunk
    limit = end_indices[-1]
    sq_signal_truncated = signal[..., :limit]**2.0

    # Sum squared values in each chunk
    sums = np.add.reduceat(sq_signal_truncated, start_indices, axis=-1)

    # Calculate chunk lengths (number of samples per chunk)
    lengths = np.diff(np.concatenate(([0], end_indices)))

    # Compute mean squared pressure
    means = sums / lengths

    levels = 10.0 * np.log10(means / reference**2.0)
    times = np.arange(levels.shape[-1]) * integration_time
    return times, levels


def time_weighted_level(
    signal: NDArray[np.float64],
    sample_frequency: float,
    time_mode: str,
    integration_time: float = None,
    reference: float = REFERENCE_PRESSURE,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate time-weighted levels at a given integration timestep.

    Args:
        signal: raw signal (can be multi-dimensional).
        sample_frequency: Sample frequency in Hz.
        time_mode: Time weighting mode, either "fast" or "slow"
        integration_time: timestep in seconds. defaults to 125ms for "fast" mode and 1s for "slow" mode.
        reference: Reference for decibels. Defaults to REFERENCE_PRESSURE.

    Returns:
        Tuple containing:
            - Time points in seconds
            - Time-weighted levels in dB
    """
    if time_mode.lower() not in ["fast", "slow"]:
        raise ValueError("time_mode must be either 'fast' or 'slow'.")

    if integration_time is None:
        integration_time = (FAST if time_mode.lower() == "fast" else SLOW)

    signal_samples = signal.shape[-1]

    # get time-weighted squared signal
    tw_sq_signal = time_weighting(signal, sample_frequency, mode=time_mode.lower())

    step = integration_time * sample_frequency
    n_steps = int(signal_samples / step)

    indices = np.arange(1, n_steps + 1) * step
    # select the last sample before each integration time step
    # we use - 1e-9 so that :
    # - when step is an integer (e.g. 4000 * 0.125 = 500), we get the correct index (499 instead of 500)
    # - when step is not an integer (e.g. 44100 * 0.125 = 5512.5), we get the correct index (5512 instead of 5512.5)
    indices = np.floor(indices - 1e-4).astype(int)

    tw_sq_values = tw_sq_signal[..., indices]

    levels = 10.0 * np.log10(tw_sq_values / reference**2.0)
    times = np.arange(levels.shape[-1]) * integration_time

    return times, levels




def frequency_weighting(
    signal: NDArray[np.float64],
    sample_frequency: float,
    weighting: str = 'A',
    zero_phase: bool = False
) -> NDArray[np.float64]:
    """Apply frequency weighting to a signal.

    Args:
        signal: Input signal (raw pressure/voltage).
        sample_frequency: Sample rate.
        weighting: 'A', 'C' or 'Z'.
        zero_phase: Prevent phase shift.
            If True, processing is done in the frequency domain (FFT) to ensure
            zero phase shift while preserving the exact IEC 61672-1 magnitude response.
            (Note: Standard forward-backward filtering would square the magnitude response).

    Returns:
        Frequency-weighted signal.
    """
    if weighting == "Z":
        return signal

    wf = WeightingFilter(fs=sample_frequency, curve=weighting)

    if zero_phase:
        # Frequency domain filtering (zero phase)
        # Get frequency response
        w, h = sosfreqz(wf.sos, worN=signal.shape[-1], fs=sample_frequency)
        # FFT of the signal
        signal_fft = np.fft.rfft(signal, axis=-1)
        # Interpolate filter response to match FFT bins
        h_interp = np.interp(
            np.fft.rfftfreq(signal.shape[-1], d=1/sample_frequency),
            w,
            np.abs(h)
        )
        # Apply filter in frequency domain
        weighted_signal_fft = signal_fft * h_interp
        # Inverse FFT to get time-domain signal
        weighted_signal = np.fft.irfft(weighted_signal_fft, n=signal.shape[-1], axis=-1)
    else:
        # Time domain filtering (causal)
        weighted_signal = sosfilt(wf.sos, signal, axis=-1)

    return weighted_signal
