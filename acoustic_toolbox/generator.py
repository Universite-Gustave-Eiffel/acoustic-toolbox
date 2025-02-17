"""The generator module provides signal generators.

The following functions calculate ``N`` samples and return an array containing the samples.

For indefinitely long iteration over the samples, consider using the output of these functions
in [`itertools.cycle`][itertools.cycle].

# Noise

Different types of noise are available. The following table lists the color
of noise and how the power and power density change per octave:

| Color  | Power | Power density |
|--------|:-----:|:-------------:|
| White  | +3 dB |    0 dB       |
| Pink   |  0 dB |   -3 dB       |
| Blue   | +6 dB |   +3 dB       |
| Brown  | -3 dB |   -6 dB       |
| Violet | +9 dB |   +6 dB       |

The colored noise is created by generating pseudo-random numbers using
[`np.random.randn`][numpy.random.randn] and then multiplying these with a curve typical for the color.
Afterwards, an inverse DFT is performed using [`np.fft.irfft`][numpy.fft.irfft].
Finally, the noise is normalized using [`acoustic_toolbox.signal.normalize`][acoustic_toolbox.signal.normalize].

## All colors

Functions:
    noise: Generate noise of a specified color.
    noise_generator: Generate `N` amount of unique samples and cycle over these samples.

## Per color

Functions:
    white: Generate white noise with constant power density and flat narrowband spectrum.
    pink: Generate pink noise with equal power in proportionally wide bands.
    blue: Generate blue noise with power increasing 6 dB per octave.
    brown: Generate brown noise with power decreasing -3 dB per octave.
    violet: Generate violet noise with power increasing +9 dB per octave.
    heaviside: Returns the value 0 for `x < 0`, 1 for `x > 0`, and 1/2 for `x = 0`.

See Also:
    For related functions, check [`scipy.signal`][scipy.signal].
"""

import itertools
from typing import Generator
import numpy as np

try:
    from pyfftw.interfaces.numpy_fft import (
        irfft,
    )  # Performs much better than numpy's fftpack
except ImportError:  # Use monkey-patching np.fft perhaps instead?
    from numpy.fft import irfft  # pylint: disable=ungrouped-imports

from .signal import normalize


def noise(
    N: int, color: str = "white", state: np.random.RandomState | None = None
) -> np.ndarray:
    """Noise generator.

    Args:
      N: Amount of samples.
      color: Color of noise.
      state: State of PRNG.

    Returns:
        Array of noise samples.
    """
    try:
        return _noise_generators[color](N, state)
    except KeyError:
        raise ValueError("Incorrect color.")


def white(N: int, state: np.random.RandomState | None = None) -> np.ndarray:
    """White noise.

    White noise has a constant power density. Its narrowband spectrum is therefore flat.
    The power in white noise will increase by a factor of two for each octave band,
    and therefore increases with 3 dB per octave.

    Args:
        N: Amount of samples.
        state: State of PRNG.

    Returns:
        Array of white noise samples.
    """
    state = np.random.RandomState() if state is None else state
    return state.randn(N)


def pink(N: int, state: np.random.RandomState | None = None) -> np.ndarray:
    """Pink noise.

    Pink noise has equal power in bands that are proportionally wide.
    Power density decreases with 3 dB per octave.

    Note:
        This method uses the filter with the following coefficients.
        $$
        B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        $$
        $$
        A = [1, -2.494956002, 2.017265875, -0.522189400]
        $$

        The filter is applied using [`scipy.signal.lfilter`][scipy.signal.lfilter]:
        ```python
        from scipy.signal import lfilter
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        return lfilter(b, a, np.random.randn(N))
        ```

        Another way would be using the FFT:
        ```python
        x = np.random.randn(N)
        X = rfft(x) / N
        S = np.sqrt(np.arange(len(X)) + 1.0)  # +1 to avoid divide by zero
        y = (irfft(X / S)).real
        ```

    Args:
        N: Amount of samples.
        state: State of PRNG.

    Returns:
        Array of pink noise samples.
    """
    # This method uses the filter with the following coefficients.
    # b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    # a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
    # return lfilter(B, A, np.random.randn(N))
    # Another way would be using the FFT
    # x = np.random.randn(N)
    # X = rfft(x) / N
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.0)  # +1 to avoid divide by zero
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def blue(N: int, state: np.random.RandomState | None = None) -> np.ndarray:
    """Blue noise.

    Power increases with 6 dB per octave.
    Power density increases with 3 dB per octave.

    Args:
        N: Amount of samples.
        state: State of PRNG.

    Returns:
        Array of blue noise samples.
    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)))  # Filter
    y = (irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def brown(N: int, state: np.random.RandomState | None = None) -> np.ndarray:
    """Brown noise.

    Power decreases with -3 dB per octave.
    Power density decreases with 6 dB per octave.

    Args:
        N: Amount of samples.
        state: State of PRNG.

    Returns:
        Array of violet noise samples.
    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.arange(len(X)) + 1  # Filter
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def violet(N: int, state: np.random.RandomState | None = None) -> np.ndarray:
    """Violet noise.

    Power increases with +9 dB per octave.
    Power density increases with +6 dB per octave.


    Args:
        N: Amount of samples.
        state: State of PRNG.

    Returns:
        Array of violet noise samples.
    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.arange(len(X))  # Filter
    y = (irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


_noise_generators = {
    "white": white,
    "pink": pink,
    "blue": blue,
    "brown": brown,
    "violet": violet,
}


def noise_generator(
    N: int = 44100, color: str = "white", state: np.random.RandomState | None = None
) -> Generator[float, None, None]:
    """Noise generator.

    Generate `N` amount of unique samples and cycle over these samples.

    Args:
      N: Amount of unique samples to generate.
      color: Color of noise.
      state: State of PRNG.

    Returns:
        Generator of noise samples.
    """
    # yield from itertools.cycle(noise(N, color)) # Python 3.3
    for sample in itertools.cycle(noise(N, color, state)):
        yield sample


def heaviside(N: int) -> np.ndarray:
    """Heaviside.

    Returns the value 0 for `x < 0`, 1 for `x > 0`, and 1/2 for `x = 0`.

    Args:
      N: Amount of samples.

    Returns:
        Array of heaviside samples.
    """
    return 0.5 * (np.sign(N) + 1)


__all__ = [
    "noise",
    "white",
    "pink",
    "blue",
    "brown",
    "violet",
    "noise_generator",
    "heaviside",
]
