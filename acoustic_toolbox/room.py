"""The room acoustics module contains several functions to calculate the reverberation time in spaces."""

import numpy as np
from numpy.typing import NDArray

from scipy.io import wavfile
from scipy import stats

from acoustic_toolbox.utils import _is_1d
from acoustic_toolbox.signal import bandpass
from acoustic_toolbox.bands import (
    _check_band_type,
    octave_low,
    octave_high,
    third_low,
    third_high,
)

SOUNDSPEED = 343.0


def mean_alpha(alphas, surfaces):
    r"""Calculate mean of absorption coefficients.

    Args:
      alphas: Absorption coefficients $\alpha$.
      surfaces: Surfaces $S$.

    Returns:
      Mean absorption coefficient $\alpha_{m}$.
    """
    axis = None if np.ndim(alphas) == 0 else 0
    return np.average(alphas, axis=axis, weights=surfaces)


def nrc(alphas):
    r"""Calculate Noise Reduction Coefficient (NRC) from four absorption coefficient values (250, 500, 1000 and 2000 Hz).

    Args:
      alphas: Absorption coefficients $\alpha$.

    Returns:
      NRC
    """
    alpha_axis = alphas.ndim - 1
    return np.mean(alphas, axis=alpha_axis)


def t60_sabine(
    surfaces: NDArray[np.float64],
    alpha: NDArray[np.float64],
    volume: float,
    c: float = SOUNDSPEED,
):
    r"""Reverberation time according to Sabine.

    Sabine's formula for the reverberation time is:
      $$
      T_{60} = \frac{24 \ln(10)}{c} \frac{V}{S\alpha}
      $$

    Args:
      surfaces: Surface of the room $S$. NumPy array that contains different surfaces.
      alpha: Absorption coefficient of the room $\alpha$.
        Contains absorption coefficients of ``surfaces``.
        It could be one value or some values in different bands (1D and 2D
        array, respectively).
      volume: Volume of the room $V$.
      c: Speed of sound $c$.

    Returns:
      Reverberation time $T_{60}$


    """
    mean_alpha_ = np.average(alpha, axis=0, weights=surfaces)
    S = np.sum(surfaces, axis=0)
    A = S * mean_alpha_
    t60 = 4.0 * np.log(10.0**6.0) * volume / (c * A)
    return t60


def t60_eyring(
    surfaces: NDArray[np.float64],
    alpha: NDArray[np.float64],
    volume: float,
    c: float = SOUNDSPEED,
):
    r"""Reverberation time according to Eyring.

      Eyring's formula for the reverberation time is:
      $$
      T_{60} = \frac{24 \ln{10} V}{c \left( 4 mV - S \ln{\left( 1 - \alpha \right)} \right)}
      $$

    Args:
      surfaces: Surfaces $S$.
      alpha: Mean absorption coefficient $\alpha$ or by frequency bands
      volume: Volume of the room $V$.
      c: Speed of sound $c$.

    Returns:
      Reverberation time $T_{60}$

    """
    mean_alpha_ = np.average(alpha, axis=0, weights=surfaces)
    S = np.sum(surfaces, axis=0)
    A = -S * np.log(1 - mean_alpha_)
    t60 = 4.0 * np.log(10.0**6.0) * volume / (c * A)
    return t60


def t60_millington(
    surfaces: NDArray[np.float64],
    alpha: NDArray[np.float64],
    volume: float,
    c: float = SOUNDSPEED,
):
    r"""Reverberation time according to Millington.

    Args:
      surfaces: Surfaces $S$.
      alpha: Mean absorption coefficient $\alpha$ or by frequency bands
      volume: Volume of the room $V$.
      c: Speed of sound $c$.

    Returns:
      Reverberation time $T_{60}$

    """
    mean_alpha_ = np.average(alpha, axis=0, weights=surfaces)
    A = -np.sum(surfaces[:, np.newaxis] * np.log(1.0 - mean_alpha_), axis=0)
    t60 = 4.0 * np.log(10.0**6.0) * volume / (c * A)
    return t60


def t60_fitzroy(
    surfaces: NDArray[np.float64],
    alpha: NDArray[np.float64],
    volume: float,
    c: float = SOUNDSPEED,
):
    r"""Reverberation time according to Fitzroy.

    Args:
      surfaces: Surfaces $S$.
      alpha: Mean absorption coefficient $\alpha$ or by frequency bands
      volume: Volume of the room $V$.
      c: Speed of sound $c$.

    Returns:
      Reverberation time $T_{60}$

    """
    Sx = np.sum(surfaces[0:2])
    Sy = np.sum(surfaces[2:4])
    Sz = np.sum(surfaces[4:6])
    St = np.sum(surfaces)
    alpha = _is_1d(alpha)
    a_x = np.average(alpha[:, 0:2], weights=surfaces[0:2], axis=1)
    a_y = np.average(alpha[:, 2:4], weights=surfaces[2:4], axis=1)
    a_z = np.average(alpha[:, 4:6], weights=surfaces[4:6], axis=1)
    factor = -(Sx / np.log(1.0 - a_x) + Sy / np.log(1.0 - a_y) + Sz / np.log(1 - a_z))
    t60 = 4.0 * np.log(10.0**6.0) * volume * factor / (c * St**2.0)
    return t60


def t60_arau(
    Sx: float,
    Sy: float,
    Sz: float,
    alpha: tuple[float, float, float],
    volume: float,
    c: float = SOUNDSPEED,
):
    r"""Reverberation time according to Arau.

    For more details, please see
    [http://www.arauacustica.com/files/publicaciones/pdf_esp_7.pdf](http://www.arauacustica.com/files/publicaciones/pdf_esp_7.pdf)

    Args:
      Sx: Total surface perpendicular to x-axis (yz-plane) $S_{x}$.
      Sy: Total surface perpendicular to y-axis (xz-plane) $S_{y}$.
      Sz: Total surface perpendicular to z-axis (xy-plane) $S_{z}$.
      alpha: Absorption coefficients $\mathbf{\alpha} = \left[ \alpha_x, \alpha_y, \alpha_z \right]$
      volume: Volume of the room $V$.
      c: Speed of sound $c$.

    Returns:
      Reverberation time $T_{60}$



    """
    a_x = -np.log(1 - alpha[0])
    a_y = -np.log(1 - alpha[1])
    a_z = -np.log(1 - alpha[2])
    St = np.sum(np.array([Sx, Sy, Sz]))
    A = St * a_x ** (Sx / St) * a_y ** (Sy / St) * a_z ** (Sz / St)
    t60 = 4.0 * np.log(10.0**6.0) * volume / (c * A)
    return t60


def t60_impulse(
    file_name: str, bands: NDArray[np.float64], rt: str = "t30"
) -> NDArray[np.float64]:
    """Reverberation time from a WAV impulse response.

    Args:
      file_name: name of the WAV file containing the impulse response.
      bands: Octave or third bands as NumPy array.
      rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.

    Returns:
      Reverberation time $T_{60}$
    """
    fs, raw_signal = wavfile.read(file_name)
    band_type = _check_band_type(bands)

    if band_type == "octave":
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == "third":
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    rt = rt.lower()
    if rt == "t30":
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == "t20":
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == "t10":
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == "edt":
        init = 0.0
        end = -10.0
        factor = 6.0

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        # Filtering signal
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1] ** 2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))

        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]
        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample : end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)
    return t60


def clarity(
    time: float, signal, fs: int, bands: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Clarity $C_i$ determined from an impulse response.

    Args:
      time: Time in miliseconds (e.g.: 50, 80).
      signal: Impulse response.
      fs: Sample frequency.
      bands: Bands of calculation (optional). Only support standard octave and third-octave bands.

    Returns:
      Clarity $C_i$
    """
    # TODO: Not tested
    band_type = _check_band_type(bands)

    if band_type == "octave":
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == "third":
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    c = np.zeros(bands.size)
    for band in range(bands.size):
        filtered_signal = bandpass(signal, low[band], high[band], fs, order=8)
        h2 = filtered_signal**2.0
        t = int((time / 1000.0) * fs + 1)
        c[band] = 10.0 * np.log10((np.sum(h2[:t]) / np.sum(h2[t:])))
    return c


def c50_from_file(file_name: str, bands: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clarity for 50 miliseconds $C_{50}$ from a file.

    Args:
      file_name: File name (only WAV is supported).
      bands: Bands of calculation (optional). Only support standard octave and third-octave bands.

    Returns:
      Clarity $C_{50}$
    """
    fs, signal = wavfile.read(file_name)
    return clarity(50.0, signal, fs, bands)


def c80_from_file(file_name: str, bands: np.ndarray):
    """Clarity for 80 miliseconds $C_{80}$ from a file.

    Args:
      file_name: File name (only WAV is supported).
      bands: Bands of calculation (optional). Only support standard octave and third-octave bands.

    Returns:
      Clarity $C_{80}$
    """
    fs, signal = wavfile.read(file_name)
    return clarity(80.0, signal, fs, bands)
