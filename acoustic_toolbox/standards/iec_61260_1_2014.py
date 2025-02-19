"""IEC 61260-1:2014 specifies performance requirements for analogue, sampled-data, and digital implementations of band-pass filters.

Functions:
    exact_center_frequency: Calculate exact center frequencies for given band indices
    lower_frequency: Calculate lower band-edge frequencies
    upper_frequency: Calculate upper band-edge frequencies
    index_of_frequency: Calculate the band index for a given frequency
    nominal_center_frequency: Calculate nominal center frequencies

Constants:
    - [NOMINAL_OCTAVE_CENTER_FREQUENCIES][acoustic_toolbox.standards.iec_61260_1_2014.NOMINAL_OCTAVE_CENTER_FREQUENCIES]: Nominal octave center frequencies
    - [NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES][acoustic_toolbox.standards.iec_61260_1_2014.NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES]: Nominal third-octave center frequencies
    - [REFERENCE_FREQUENCY][acoustic_toolbox.standards.iec_61260_1_2014.REFERENCE_FREQUENCY]: Reference frequency
    - [OCTAVE_FREQUENCY_RATIO][acoustic_toolbox.standards.iec_61260_1_2014.OCTAVE_FREQUENCY_RATIO]: Octave frequency ratio
"""

import acoustic_toolbox
import numpy as np

NOMINAL_OCTAVE_CENTER_FREQUENCIES = np.array(
    [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
)
"""Nominal octave center frequencies."""

NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES = np.array(
    [
        25.0,
        31.5,
        40.0,
        50.0,
        63.0,
        80.0,
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        315.0,
        400.0,
        500.0,
        630.0,
        800.0,
        1000.0,
        1250.0,
        1600.0,
        2000.0,
        2500.0,
        3150.0,
        4000.0,
        5000.0,
        6300.0,
        8000.0,
        10000.0,
        12500.0,
        16000.0,
        20000.0,
    ]
)
"""Nominal third-octave center frequencies in the audio range."""

REFERENCE_FREQUENCY = 1000.0
"""Reference frequency."""

OCTAVE_FREQUENCY_RATIO = 10.0 ** (3.0 / 10.0)
"""Octave frequency ratio $G$.

See equation 1.
"""


def exact_center_frequency(
    x, fraction=1, ref=REFERENCE_FREQUENCY, G=OCTAVE_FREQUENCY_RATIO
) -> float | np.ndarray:
    r"""Center frequencies $f_m$ for band indices $x$. See equation 2 and 3.

    The center frequencies are given by
    $$
    f_m = f_r G^{x/b}
    $$

    In case the bandwidth designator $b$ is an even number, the center frequencies are given by
    $$
    f_m = f_r G^{(2x+1)/2b}
    $$

    See equation 2 and 3 of the standard.

    Args:
        x: Band index $x$.
        fraction: Bandwidth designator $b$. For example, for 1/3-octave filter $b=3$. Defaults to 1.
        ref: Reference center frequency $f_r$.
        G: Octave frequency ratio $G$.

    Returns:
        Center frequencies $f_m$ calculated according to equations 2 and 3.
    """
    fraction = np.asarray(fraction)
    uneven = (fraction % 2).astype("bool")
    return ref * G ** ((2.0 * x + 1.0) / (2.0 * fraction)) * np.logical_not(
        uneven
    ) + uneven * ref * G ** (x / fraction)


def lower_frequency(
    center: float | np.ndarray, fraction=1, G=OCTAVE_FREQUENCY_RATIO
) -> float | np.ndarray:
    """Lower band-edge frequencies. See equation 4.

    The lower band-edge frequencies are given by
    $$
    f_1 = f_m G^{-1/2b}
    $$

    See equation 4 of the standard.

    Args:
        center: Center frequencies $f_m$.
        fraction: Bandwidth designator $b$.
        G: Octave frequency ratio $G$.

    Returns:
        Lower band-edge frequencies $f_1$
    """
    return center * G ** (-1.0 / (2.0 * fraction))


def upper_frequency(
    center: float | np.ndarray, fraction=1, G=OCTAVE_FREQUENCY_RATIO
) -> float | np.ndarray:
    """Upper band-edge frequencies. See equation 5.

    The upper band-edge frequencies are given by
    $$
    f_2 = f_m G^{+1/2b}
    $$

    See equation 5 of the standard.

    Args:
        center: Center frequencies $f_m$.
        fraction: Bandwidth designator $b$.
        G: Octave frequency ratio $G$.

    Returns:
        Upper band-edge frequencies $f_2$.
    """
    return center * G ** (+1.0 / (2.0 * fraction))


def index_of_frequency(
    frequency, fraction=1, ref=REFERENCE_FREQUENCY, G=OCTAVE_FREQUENCY_RATIO
) -> int | np.ndarray:
    r"""Calculate the band index for a given frequency.

    The index of the center frequency is given by

    $$
    x = \text{round}\left(b \frac{\log(f/f_r)}{\log(G)}\right)
    $$

    See equation 6 of the standard.

    Note:
        This equation is not part of the standard.
        However, it follows from [`exact_center_frequency`](acoustic_toolbox.standards.iec_61260_1_2014.exact_center_frequency).

    Args:
        frequency: Frequencies $f$
        fraction: Bandwidth designator $b$.
        ref: Reference frequency.
        G: Octave frequency ratio $G$.

    Returns:
        Band indices $x$
    """
    fraction = np.asarray(fraction)
    uneven = (fraction % 2).astype("bool")
    return (
        np.round((2.0 * fraction * np.log(frequency / ref) / np.log(G) - 1.0))
        / 2.0
        * np.logical_not(uneven)
        + uneven
        * np.round(fraction * np.log(frequency / ref) / np.log(G)).astype("int16")
    ).astype("int16")


def _roundn(x, n):
    """Round a number to n significant digits.

    Args:
        x: Number to round.
        n: Number of significant digits.

    Returns:
        float: Rounded number.
    """
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)


def _nominal_center_frequency(center, fraction):
    """Calculate nominal frequency according to standard.

    Args:
        center: Exact mid-frequency to be rounded.
        fraction: Bandwidth designator or fraction.

    Returns:
        float: Nominal center frequency according to standard sections E.1-E.3.

    See Also:
        exact_center_frequency: Function to calculate exact center frequencies

    Raises:
        NotImplementedError: If fraction > 24.
        ValueError: If fraction is invalid.
    """
    b = fraction
    x = center

    # Section E.1: 1/1-octaves
    if b == 1:
        n = index_of_frequency(x, b)
        if -6 <= n < 5:  # Correspond to indices when n=0 corresponds to 1000 Hz
            return acoustic_toolbox.standards.iec_61672_1_2013.NOMINAL_OCTAVE_CENTER_FREQUENCIES[
                n + 6
            ]
        elif n >= 5:
            return 2.0 * _nominal_center_frequency(
                exact_center_frequency(n - 1, b), b
            )  # WARNING: Unclear in standard!
        else:
            return (
                1.0
                / 2.0
                * _nominal_center_frequency(exact_center_frequency(n + 1, b), b)
            )  # WARNING: Unclear in standard!

    # Section E.2: 1/2-octaves
    elif b == 2:
        return _roundn(x, 2)

    # Section E.1: 1/3-octaves
    elif b == 3:
        n = index_of_frequency(x, b)

        if -20 <= n < 14:  # Correspond to indices when n=0 corresponds to 1000 Hz
            return acoustic_toolbox.standards.iec_61672_1_2013.NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES[
                n + 20
            ]
        elif n >= 14:
            return 10.0 * _nominal_center_frequency(
                exact_center_frequency(n - 10, b), b
            )  # WARNING: Unclear in standard!
        else:
            return (
                1.0
                / 10.0
                * _nominal_center_frequency(exact_center_frequency(n + 10, b), b)
            )  # WARNING: Unclear in standard!

    # Section E3.3: 1/4 to 1/24-octaves, inclusive
    elif 4 <= b <= 24:
        msd = x // 10.0 ** np.floor(np.log10(x))
        if msd < 5:
            return _roundn(x, 2)  # E3.2
        else:
            return _roundn(x, 1)  # E3.3

    # Section E3.5: > 1/24-octaves
    elif b > 24:
        raise NotImplementedError("b > 24 is not implemented")
    else:
        raise ValueError("Wrong value for b")


nominal_center_frequency = np.vectorize(_nominal_center_frequency)
"""Vectorized function to calculate nominal center frequencies.

Args:
    center: Exact center frequency.
    fraction: Band designator or fraction.

Returns:
    float or ndarray: Nominal center frequencies calculated according to standard.
"""
