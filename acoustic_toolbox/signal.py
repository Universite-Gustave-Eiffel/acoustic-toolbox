"""The signal module contains all kinds of signal processing related functions.

# Filtering

Classes:
    Filterbank: Fractional-Octave filter bank.

Functions:
    bandpass_filter: Band-pass filter.
    bandpass: Filter signal with band-pass filter.
    lowpass: Filter signal with low-pass filter.
    highpass: Filter signal with high-pass filter.
    octave_filter: Fractional-octave band-pass filter.
    convolve: Perform convolution of a signal with a linear time-variant system.

# Windowing

Functions:
    window_scaling_factor: Calculate window scaling factor.
    apply_window: Apply window to signal.

# Spectra

Functions:
    amplitude_spectrum: Amplitude spectrum of instantaneous signal.
    auto_spectrum: Auto-spectrum of instantaneous signal.
    power_spectrum: Power spectrum of instantaneous signal.
    density_spectrum: Density spectrum of instantaneous signal.
    angle_spectrum: Phase angle spectrum of instantaneous signal.
    phase_spectrum: Phase spectrum of instantaneous signal.

# Frequency bands

Classes:
    Frequencies: Object describing frequency bands.
    EqualBand: Equal bandwidth spectrum.
    OctaveBand: Fractional-octave band spectrum.

Functions:
    integrate_bands: Reduce frequency resolution of power spectrum.
    octaves: Calculate level per 1/1-octave in frequency domain.
    third_octaves: Calculate level per 1/3-octave in frequency domain.

# Hilbert transform

Functions:
    amplitude_envelope: Instantaneous amplitude of tone.
    instantaneous_phase: Instantaneous phase of tone.
    instantaneous_frequency: Determine instantaneous frequency of tone.

# Conversion

Functions:
    decibel_to_neper: Convert decibel to neper.
    neper_to_decibel: Convert neper to decibel.

# Other

Functions:
    isolate: Isolate signals.
    zero_crossings: Determine the positions of zero crossings in data.
    rms: Root mean squared of signal.
    ms: Mean value of signal squared.
    normalize: Normalize power in signal.
    ir2fr: Convert impulse response into frequency response.
    wvd: Wigner-Ville Distribution.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Generator
from scipy.sparse import spdiags
from scipy.signal import (
    butter,
    lfilter,
    freqz,
    filtfilt,
    sosfilt,
    lti,
    cheby1,
    firwin,
    hilbert,
)
from scipy.signal._arraytools import even_ext, odd_ext, const_ext

import acoustic_toolbox.octave
import acoustic_toolbox.bands
from acoustic_toolbox.standards.iso_tr_25417_2007 import REFERENCE_PRESSURE
from acoustic_toolbox.standards.iec_61672_1_2013 import (
    NOMINAL_OCTAVE_CENTER_FREQUENCIES,
    NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
)

try:
    from pyfftw.interfaces.numpy_fft import rfft  # type: ignore
except ImportError:
    from numpy.fft import rfft


def bandpass_filter(
    lowcut: float, highcut: float, fs: float, order: int = 8, output: str = "sos"
) -> tuple | None:
    """Band-pass filter.

    Args:
        lowcut: Lower cut-off frequency.
        highcut: Upper cut-off frequency.
        fs: Sample frequency.
        order: Filter order. Defaults to 8.
        output: Output type. {'ba', 'zpk', 'sos'}.

    Returns:
        tuple: Filter coefficients depending on `output`.

    See Also:
        [`scipy.signal.butter`][scipy.signal.butter]: For more details on the Butterworth filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order / 2, [low, high], btype="band", output=output)


def bandpass(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 8,
    zero_phase: bool = False,
) -> np.ndarray:
    """Filter signal with band-pass filter.

    Args:
        signal: Signal to be filtered.
        lowcut: Lower cut-off frequency.
        highcut: Upper cut-off frequency.
        fs: Sample frequency.
        order: Filter order. Defaults to 8.
        zero_phase: If True, uses `filtfilt` to prevent phase error.

    Returns:
        Filtered signal.

    See Also:
        [`bandpass_filter`][acoustic_toolbox.signal.bandpass_filter]: The filter that is used.
    """
    sos = bandpass_filter(lowcut, highcut, fs, order, output="sos")
    if zero_phase:
        return _sosfiltfilt(sos, signal)
    else:
        return sosfilt(sos, signal)


def bandstop(signal, lowcut, highcut, fs, order=8, zero_phase: bool = False):
    """Filter signal with band-stop filter.

    Args:
        signal: Signal to be filtered.
        lowcut: Lower cut-off frequency.
        highcut: Upper cut-off frequency.
        fs: Sample frequency.
        order: Filter order. Defaults to 8.
        zero_phase: If True, uses filtfilt to prevent phase error. Defaults to False.

    Returns:
        Filtered signal.

    See Also:
        [`lowpass`][acoustic_toolbox.signal.lowpass], [`highpass`][acoustic_toolbox.signal.highpass]: Used to create the band-stop filter.
    """
    return lowpass(
        signal, lowcut, fs, order=(order // 2), zero_phase=zero_phase
    ) + highpass(signal, highcut, fs, order=(order // 2), zero_phase=zero_phase)


def lowpass(signal, cutoff, fs, order=4, zero_phase: bool = False):
    """Filter signal with low-pass filter.

    A Butterworth filter is used. Filtering is done with second-order sections.

    Args:
        signal: Signal to be filtered.
        cutoff: Cut-off frequency.
        fs: Sample frequency.
        order: Filter order. Defaults to 4.
        zero_phase: If True, uses filtfilt to prevent phase error. Defaults to False.

    Returns:
        Filtered signal.

    See Also:
        [`scipy.signal.butter`][scipy.signal.butter]: For more details on the Butterworth filter.
    """
    sos = butter(order, cutoff / (fs / 2.0), btype="low", output="sos")
    if zero_phase:
        return _sosfiltfilt(sos, signal)
    else:
        return sosfilt(sos, signal)


def highpass(signal, cutoff, fs, order=4, zero_phase: bool = False):
    """Filter signal with high-pass filter.

    A Butterworth filter is used. Filtering is done with second-order sections.

    Args:
        signal: Signal to be filtered.
        cutoff: Cut-off frequency.
        fs: Sample frequency.
        order: Filter order. Defaults to 4.
        zero_phase: If True, uses filtfilt to prevent phase error. Defaults to False.

    Returns:
        Filtered signal.

    See Also:
        [`scipy.signal.butter`][scipy.signal.butter]: For more details on the Butterworth filter.
    """
    sos = butter(order, cutoff / (fs / 2.0), btype="high", output="sos")
    if zero_phase:
        return _sosfiltfilt(sos, signal)
    else:
        return sosfilt(sos, signal)


def octave_filter(center, fs, fraction, order=8, output: str = "sos"):
    """Fractional-octave band-pass filter.

    A Butterworth filter is used.
    Args:
        center: Center frequency of fractional-octave band.
        fs: Sample frequency.
        fraction: Fraction of fractional-octave band.
        order: Filter order. Defaults to 8.
        output: Output type. {'ba', 'zpk', 'sos'}. Defaults to 'sos'.

    Returns:
        tuple: Filter coefficients depending on `output`.

    See Also:
        [`bandpass_filter`][acoustic_toolbox.signal.bandpass_filter]: Used to create the fractional-octave filter.
    """
    ob = OctaveBand(center=center, fraction=fraction)
    return bandpass_filter(
        ob._get_scalar(ob.lower), ob._get_scalar(ob.upper), fs, order, output=output
    )


def octavepass(signal, center, fs, fraction, order=8, zero_phase: bool = True):
    """Filter signal with fractional-octave bandpass filter.

    A Butterworth filter is used. Filtering is done with second-order sections.

    Args:
        signal: Signal to be filtered.
        center: Center frequency of fractional-octave band.
        fs: Sample frequency.
        fraction: Fraction of fractional-octave band.
        order: Filter order. Defaults to 8.
        zero_phase: If True, uses filtfilt to prevent phase error. Defaults to True.

    Returns:
        Filtered signal.

    See Also:
        [`octave_filter`][acoustic_toolbox.signal.octave_filter]: The filter that is used.
    """
    sos = octave_filter(center, fs, fraction, order)
    if zero_phase:
        return _sosfiltfilt(sos, signal)
    else:
        return sosfilt(sos, signal)


def convolve(signal, ltv: np.ndarray, mode: str = "full"):
    r"""Perform convolution of a signal with a linear time-variant system (`ltv`).

    Notes:
        The convolution of two sequences is given by
        $$
        \mathbf{y} = \mathbf{t} \star \mathbf{u}
        $$

        This can be written as a matrix-vector multiplication
        $$
        \mathbf{y} = \mathbf{T} \cdot \mathbf{u}
        $$

        where $T$ is a Toeplitz matrix in which each column represents an impulse response.
        In the case of a linear time-invariant (LTI) system, each column represents a time-shifted copy of the first column.
        In the time-variant case (LTV), every column can contain a unique impulse response, both in values as in size.

        This function assumes all impulse responses are of the same size.
        The input matrix `ltv` thus represents the non-shifted version of the Toeplitz matrix.

    Args:
        signal: Vector representing the input signal $u$.
        ltv: 2D array where each column represents an impulse response.
        mode: {'full', 'valid', 'same'}. Determines the size of the output.

    Returns:
        The result of the convolution operation.

    Raises:
        AssertionError: If the length of the signal does not match the number of columns in `ltv`.

    See Also:
        For convolution with LTI systems.

        - [`np.convolve`][numpy.convolve]
        - [`scipy.signal.convolve`][scipy.signal.convolve]
        - [`scipy.signal.fftconvolve`][scipy.signal.fftconvolve]
    """
    assert len(signal) == ltv.shape[1]

    n = ltv.shape[0] + len(signal) - 1  # Length of output vector
    un = np.concatenate((signal, np.zeros(ltv.shape[0] - 1)))  # Resize input vector
    offsets = np.arange(0, -ltv.shape[0], -1)  # Offsets for impulse responses
    Cs = spdiags(ltv, offsets, n, n)  # Sparse representation of IR's.
    out = Cs.dot(un)  # Calculate dot product.

    if mode == "full":
        return out
    elif mode == "same":
        start = ltv.shape[0] / 2 - 1 + ltv.shape[0] % 2
        stop = len(signal) + ltv.shape[0] / 2 - 1 + ltv.shape[0] % 2
        return out[start:stop]
    elif mode == "valid":
        # length = len(signal) - ltv.shape[0]
        start = ltv.shape[0] - 1
        stop = len(signal)
        return out[start:stop]


def ir2fr(ir, fs, N: int | None = None):
    """Convert impulse response into frequency response. Returns single-sided RMS spectrum.

    Calculates the positive frequencies using [`np.fft.rfft`][numpy.fft.rfft].
    Corrections are then applied to obtain the single-sided spectrum.

    Note:
        Single-sided spectrum. Therefor the amount of bins returned is either N/2 or N/2+1.

    Args:
        ir: Impulse response.
        fs: Sample frequency.
        N: Blocks

    Returns:
        Frequencies and frequency response.

    See Also:
        [`np.fft.rfft`][numpy.fft.rfft]: Used for calculating the positive frequencies.
    """
    N = N if N else ir.shape[-1]
    fr = rfft(ir, n=N) / N
    f = np.fft.rfftfreq(N, 1.0 / fs)  # / 2.0

    fr *= 2.0
    fr[..., 0] /= 2.0  # DC component should not be doubled.
    if not N % 2:  # if not uneven
        fr[..., -1] /= 2.0  # And neither should fs/2 be.

    return f, fr


def decibel_to_neper(decibel):
    r"""Convert decibel to neper.

    Note:
        The conversion is given by
        $$
        \mathrm{dB} = \frac{\log{10}}{20} \mathrm{Np}
        $$

    Args:
        decibel: Value in decibel (dB).

    Returns:
        Value in neper (Np).

    See Also:
        [`neper_to_decibel`][acoustic_toolbox.signal.neper_to_decibel]: For the reverse conversion.
    """
    return np.log(10.0) / 20.0 * decibel


def neper_to_decibel(neper):
    r"""Convert neper to decibel.

    Note:
        The conversion is given by
        $$
        \mathrm{Np} = \frac{20}{\log{10}} \mathrm{dB}
        $$

    Args:
        neper: Value in neper (Np).

    Returns:
        Value in decibel (dB).

    See Also:
        [`decibel_to_neper`][acoustic_toolbox.signal.decibel_to_neper]: For the reverse conversion.
    """
    return 20.0 / np.log(10.0) * neper


class Frequencies:
    """Object describing frequency bands.

    Attributes:
        center: Center frequencies.
        lower: Lower frequencies.
        upper: Upper frequencies.
        bandwidth: Bandwidth.
    """

    center: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    bandwidth: NDArray[np.float64]

    def __init__(
        self,
        center: NDArray[np.float64] | list[float],
        lower: NDArray[np.float64] | list[float],
        upper: NDArray[np.float64] | list[float],
        bandwidth: NDArray[np.float64] | list[float] | None = None,
    ):
        self.center = np.asarray(center)
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.bandwidth = (
            np.asarray(bandwidth)
            if bandwidth is not None
            else np.asarray(self.upper) - np.asarray(self.lower)
        )

    def __iter__(self):
        for i in range(len(self.center)):
            yield self[i]

    def __len__(self):
        return len(self.center)

    def __str__(self):
        return str(self.center)

    def __repr__(self):
        return "Frequencies({})".format(str(self.center))

    def _get_scalar(self, arr: NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Safely extract a scalar value from a single-element array.

        Args:
            arr: Array to extract scalar from.

        Returns:
            Scalar value if array has one element, otherwise the original array.
        """
        try:
            return arr.item() if arr.size == 1 else arr
        except ValueError:
            return arr  # Return original array if it has multiple elements

    def angular(self):
        """Angular center frequency in radians per second.

        Returns:
            Angular frequencies.
        """
        return 2.0 * np.pi * self.center


class EqualBand(Frequencies):
    """Equal bandwidth spectrum. Generally used for narrowband data.

    Attributes:
        center: Center frequencies.
        fstart: First center frequency.
        fstop: Last center frequency.
        nbands: Amount of frequency bands.
        bandwidth: Bandwidth of bands.
    """

    def __init__(
        self,
        center=None,
        fstart=None,
        fstop=None,
        nbands: int | None = None,
        bandwidth=None,
    ):
        """Equal bandwidth spectrum.

        Raises:
            ValueError: If the center frequencies are not equally spaced.
            ValueError: If insufficient parameters are provided.
        """
        if center is not None:
            try:
                nbands = len(center)
            except TypeError:
                center = [center]
                nbands = 1

            u = np.unique(np.diff(center).round(decimals=3))
            n = len(u)
            if n == 1:
                bandwidth = u
            elif n > 1:
                raise ValueError("Given center frequencies are not equally spaced.")
            else:
                pass
            fstart = center[0]  # - bandwidth/2.0
            fstop = center[-1]  # + bandwidth/2.0
        elif fstart is not None and fstop is not None and nbands:
            bandwidth = (fstop - fstart) / (nbands - 1)
        elif fstart is not None and fstop is not None and bandwidth:
            nbands = round((fstop - fstart) / bandwidth) + 1
        elif fstart is not None and bandwidth and nbands:
            fstop = fstart + nbands * bandwidth
        elif fstop is not None and bandwidth and nbands:
            fstart = fstop - (nbands - 1) * bandwidth
        else:
            raise ValueError(
                "Insufficient parameters. Cannot determine fstart, fstop, bandwidth."
            )

        center = fstart + np.arange(0, nbands) * bandwidth  # + bandwidth/2.0
        upper = fstart + np.arange(0, nbands) * bandwidth + bandwidth / 2.0
        lower = fstart + np.arange(0, nbands) * bandwidth - bandwidth / 2.0

        super(EqualBand, self).__init__(center, lower, upper, bandwidth)

    def __getitem__(self, key):
        return type(self)(center=self.center[key], bandwidth=self.bandwidth)

    def __repr__(self):
        return "EqualBand({})".format(str(self.center))


class OctaveBand(Frequencies):
    """Fractional-octave band spectrum.

    Attributes:
        center: Center frequencies.
        fstart: First center frequency.
        fstop: Last center frequency.
        nbands: Amount of frequency bands.
        bandwidth: Bandwidth.
        fraction: Fraction of fractional-octave filter.
        reference: Reference center frequency.
        nominal: Nominal center frequencies.
    """

    def __init__(
        self,
        center=None,
        fstart=None,
        fstop=None,
        nbands: int | None = None,
        fraction=1,
        reference=acoustic_toolbox.octave.REFERENCE,
    ):
        """Fractional-octave band spectrum.

        Raises:
            ValueError: If insufficient parameters are provided.
        """
        if center is not None:
            try:
                nbands = len(center)
            except TypeError:
                center = [center]
            center = np.asarray(center)
            indices = acoustic_toolbox.octave.index_of_frequency(
                center, fraction=fraction, ref=reference
            )
        elif fstart is not None and fstop is not None:
            nstart = acoustic_toolbox.octave.index_of_frequency(
                fstart, fraction=fraction, ref=reference
            )
            nstop = acoustic_toolbox.octave.index_of_frequency(
                fstop, fraction=fraction, ref=reference
            )
            indices = np.arange(nstart, nstop + 1)
        elif fstart is not None and nbands is not None:
            nstart = acoustic_toolbox.octave.index_of_frequency(
                fstart, fraction=fraction, ref=reference
            )
            indices = np.arange(nstart, nstart + nbands)
        elif fstop is not None and nbands is not None:
            nstop = acoustic_toolbox.octave.index_of_frequency(
                fstop, fraction=fraction, ref=reference
            )
            indices = np.arange(nstop - nbands, nstop)
        else:
            raise ValueError(
                "Insufficient parameters. Cannot determine fstart and/or fstop."
            )

        center = acoustic_toolbox.octave.exact_center_frequency(
            None, fraction=fraction, n=indices, ref=reference
        )
        lower = acoustic_toolbox.octave.lower_frequency(center, fraction=fraction)
        upper = acoustic_toolbox.octave.upper_frequency(center, fraction=fraction)
        bandwidth = upper - lower
        nominal = acoustic_toolbox.octave.nominal_center_frequency(
            None, fraction, indices
        )

        super(OctaveBand, self).__init__(center, lower, upper, bandwidth)

        self.fraction = fraction
        self.reference = reference
        self.nominal = nominal

    def __getitem__(self, key):
        return type(self)(
            center=self.center[key], fraction=self.fraction, reference=self.reference
        )

    def __repr__(self):
        return "OctaveBand({})".format(str(self.center))


def ms(x):
    """Mean value of signal `x` squared.

    Args:
        x: Dynamic quantity.

    Returns:
        Mean squared of `x`.
    """
    return (np.abs(x) ** 2.0).mean()


def rms(x):
    r"""Root mean squared of signal `x`.

    Args:
        x: Dynamic quantity.

    Returns:
        Root mean squared value of `x`.
        $$
        x_{rms} = \lim_{T \to \infty} \sqrt{\frac{1}{T} \int_0^T |f(x)|^2 \mathrm{d} t }
        $$

    See Also:
        [`ms`][acoustic_toolbox.signal.ms]
    """
    return np.sqrt(ms(x))


def normalize(y, x=None):
    r"""Normalize power in `y` to a (standard normal) white noise signal.

    Optionally normalize to power in signal `x`.

    Note:
        The mean power of a Gaussian with $\mu=0$ and $\sigma=1$ is 1.

    Args:
        y: Signal to be normalized.
        x: Reference signal. Defaults to None.

    Returns:
        Normalized signal.
    """
    if x is not None:
        x = ms(x)
    else:
        x = 1.0
    return y * np.sqrt(x / ms(y))
    # return y * np.sqrt( 1.0 / (np.abs(y)**2.0).mean() )

    ## Broken? Caused correlation in auralizations....weird!


def window_scaling_factor(window, axis: int = -1):
    r"""Calculate window scaling factor.

    When analysing broadband (filtered noise) signals, it is common to normalize
    the windowed signal so that it has the same power as the un-windowed signal.
    $$
    S = \sqrt{\frac{\sum_{i=0}^N w_i^2}{N}}
    $$

    Args:
        window: Window.
        axis: Axis along which to calculate.

    Returns:
        Window scaling factor.
    """
    return np.sqrt((window * window).mean(axis=axis))


def apply_window(x, window):
    """Apply window to signal.

    $$
    x_s(t) = x(t) / S
    $$

    where $S$ is the window scaling factor.

    Args:
        x: Instantaneous signal $x(t)$.
        window: Vector representing window.

    Returns:
        Signal with window applied.

    See Also:
        [`window_scaling_factor`][acoustic_toolbox.signal.window_scaling_factor]: For calculating the scaling factor.
    """
    s = window_scaling_factor(window)  # Determine window scaling factor.
    n = len(window)
    windows = x // n  # Amount of windows.
    x = x[0 : windows * n]  # Truncate final part of signal that does not fit.
    # x = x.reshape(-1, len(window)) # Reshape so we can apply window.
    y = np.tile(window, windows)

    return x * y / s


def amplitude_spectrum(x, fs, N: int | None = None):
    r"""Amplitude spectrum of instantaneous signal $x(t)$.

    The amplitude spectrum  gives the amplitudes of the sinusoidal the signal is built
    up from, and the RMS (root-mean-square) amplitudes can easily be found by dividing
    these amplitudes with $\sqrt{2}$

    The amplitude spectrum is double-sided.

    Args:
        x: Instantaneous signal $x(t)$.
        fs: Sample frequency $f_s$.
        N: Number of FFT bins.

    Returns:
        tuple: Frequencies and amplitude spectrum.
    """
    N = N if N else x.shape[-1]
    fr = np.fft.fft(x, n=N) / N
    f = np.fft.fftfreq(N, 1.0 / fs)
    return np.fft.fftshift(f), np.fft.fftshift(fr, axes=[-1])


def auto_spectrum(x, fs, N: int | None = None):
    r"""Auto-spectrum of instantaneous signal $x(t)$.

    The auto-spectrum contains the squared amplitudes of the signal. Squared amplitudes
    are used when presenting data as it is a measure of the power/energy in the signal.

    $$
    S_{xx} (f_n) = \overline{X (f_n)} \cdot X (f_n)
    $$

    The auto-spectrum is double-sided.

    Args:
        x: Instantaneous signal $x(t)$.
        fs: Sample frequency $f_s$.
        N: Number of FFT bins.

    Returns:
        f: Frequencies
        a: Auto-spectrum
    """
    f, a = amplitude_spectrum(x, fs, N=N)
    return f, (a * a.conj()).real


def power_spectrum(x, fs, N: int | None = None):
    r"""Power spectrum of instantaneous signal $x(t)$.


    The power spectrum, or single-sided autospectrum, contains the squared RMS amplitudes of the signal.

    A power spectrum is a spectrum with squared RMS values. The power spectrum is
    calculated from the autospectrum of the signal.

    Warning:
        Does not include scaling to reference value!

    Args:
        x: Instantaneous signal $x(t)$.
        fs: Sample frequency $f_s$.
        N: Number of FFT bins.

    Returns:
        f: Frequencies
        a: Power spectrum

    See Also:
        [`auto_spectrum`][acoustic_toolbox.signal.auto_spectrum]
    """
    N = N if N else x.shape[-1]
    f, a = auto_spectrum(x, fs, N=N)
    a = a[..., N // 2 :]
    f = f[..., N // 2 :]
    a *= 2.0
    a[..., 0] /= 2.0  # DC component should not be doubled.
    if not N % 2:  # if not uneven
        a[..., -1] /= 2.0  # And neither should fs/2 be.
    return f, a


def angle_spectrum(x, fs, N: int | None = None):
    r"""Phase angle spectrum of instantaneous signal $x(t)$.

    This function returns a single-sided phase angle spectrum.

    Args:
        x: Instantaneous signal $x(t)$.
        fs: Sample frequency $f_s$.
        N: Number of FFT bins.

    Returns:
        f: Frequencies
        a: Phase angle spectrum

    See Also:
        [`phase_spectrum`][acoustic_toolbox.signal.phase_spectrum]: For unwrapped phase spectrum.
    """
    N = N if N else x.shape[-1]
    f, a = amplitude_spectrum(x, fs, N)
    a = np.angle(a)
    a = a[..., N // 2 :]
    f = f[..., N // 2 :]
    return f, a


def phase_spectrum(x, fs, N: int | None = None):
    r"""Phase spectrum of instantaneous signal $x(t)$.

    This function returns single-sided unwrapped phase spectrum.

    Args:
        x: Instantaneous signal $x(t)$.
        fs: Sample frequency $f_s$.
        N: Number of FFT bins.

    Returns:
        f: Frequencies
        a: Unwrapped phase spectrum

    See Also:
        [`angle_spectrum`][acoustic_toolbox.signal.angle_spectrum]: For wrapped phase angle.
    """
    f, a = angle_spectrum(x, fs, N=None)
    return f, np.unwrap(a)


def density_spectrum(x, fs, N: int | None = None):
    """Density spectrum of instantaneous signal $x(t)$.

    A density spectrum considers the amplitudes per unit frequency.
    Density spectra are used to compare spectra with different frequency resolution as the
    magnitudes are not influenced by the resolution because it is per Hertz. The amplitude
    spectra on the other hand depend on the chosen frequency resolution.

    Args:
        x: Instantaneous signal $x(t)$.
        fs: Sample frequency $f_s$.
        N: Number of FFT bins.

    Returns:
        f: Frequencies
        a: Density spectrum
    """
    N = N if N else x.shape[-1]
    fr = np.fft.fft(x, n=N) / fs
    f = np.fft.fftfreq(N, 1.0 / fs)
    return np.fft.fftshift(f), np.fft.fftshift(fr)


def integrate_bands(data, a, b):
    """Reduce frequency resolution of power spectrum. Merges frequency bands by integration.


    Args:
        data: Vector with narrowband powers.
        a: Instance of `Frequencies`.
        b: Instance of `Frequencies`.

    Returns:
        Integrated bands.

    Raises:
        NotImplementedError: If the ratio of fractional-octaves is not an integer.

    Todo:
        Needs rewriting so that the summation goes over axis=1.
    """
    try:
        if b.fraction % a.fraction:
            raise NotImplementedError(
                "Non-integer ratio of fractional-octaves are not supported."
            )
    except AttributeError:
        pass

    lower, _ = np.meshgrid(b.lower, a.center)
    upper, _ = np.meshgrid(b.upper, a.center)
    _, center = np.meshgrid(b.center, a.center)

    return ((lower < center) * (center <= upper) * data[..., None]).sum(axis=-2)


def bandpass_frequencies(
    x,
    fs,
    frequencies: Frequencies,
    order: int = 8,
    purge: bool = False,
    zero_phase: bool = False,
) -> tuple[Frequencies, np.ndarray]:
    """Apply bandpass filters for frequencies.

    Args:
        x: Instantaneous signal $x(t)$.
        fs: Sample frequency.
        frequencies: Instance of `Frequencies`.
        order: Filter order.
        purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
        zero_phase: Prevent phase error by filtering in both directions (filtfilt)

    Returns:
        OctaveBand: Instance of `OctaveBand`
        np.ndarray: Filtered array.
    """
    if purge:
        frequencies = frequencies[frequencies.upper < fs / 2.0]
    return frequencies, np.array(
        [
            bandpass(
                x,
                band._get_scalar(band.lower),
                band._get_scalar(band.upper),
                fs,
                order,
                zero_phase=zero_phase,
            )
            for band in frequencies
        ]
    )


def bandpass_octaves(
    x,
    fs,
    frequencies=NOMINAL_OCTAVE_CENTER_FREQUENCIES,
    order=8,
    purge=False,
    zero_phase: bool = False,
) -> tuple[OctaveBand, np.ndarray]:
    """Apply 1/1-octave bandpass filters.

    Args:
        x: Instantaneous signal $x(t)$.
        fs: Sample frequency.
        frequencies: Center frequencies.
        order: Filter order.
        purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
        zero_phase: Prevent phase error by filtering in both directions (filtfilt)

    Returns:
        OctaveBand: Instance of `OctaveBand`
        np.ndarray: Filtered array.

    See Also:
        [`octavepass`][acoustic_toolbox.signal.octavepass]
    """
    return bandpass_fractional_octaves(
        x, fs, frequencies, fraction=1, order=order, purge=purge, zero_phase=zero_phase
    )


def bandpass_third_octaves(
    x,
    fs,
    frequencies=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
    order: int = 8,
    purge: bool = False,
    zero_phase: bool = False,
) -> tuple[OctaveBand, np.ndarray]:
    """Apply 1/3-octave bandpass filters.

    Args:
        x: Instantaneous signal $x(t)$.
        fs: Sample frequency.
        frequencies: Center frequencies.
        order: Filter order.
        purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
        zero_phase: Prevent phase error by filtering in both directions (filtfilt)

    Returns:
        OctaveBand: Instance of `OctaveBand`
        np.ndarray: Filtered array.

    See Also:
        [`octavepass`][acoustic_toolbox.signal.octavepass]
    """
    return bandpass_fractional_octaves(
        x, fs, frequencies, fraction=3, order=order, purge=purge, zero_phase=zero_phase
    )


def bandpass_fractional_octaves(
    x,
    fs,
    frequencies,
    fraction=None,
    order: int = 8,
    purge: bool = False,
    zero_phase: bool = False,
) -> tuple[OctaveBand, np.ndarray]:
    """Apply 1/N-octave bandpass filters.

    Args:
        x: Instantaneous signal $x(t)$.
        fs: Sample frequency.
        frequencies: Center frequencies or instance of `OctaveBand`.
        fraction: Fraction of fractional-octave band.
        order: Filter order.
        purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
        zero_phase: Prevent phase error by filtering in both directions (filtfilt)

    Returns:
        OctaveBand: Instance of `OctaveBand`
        np.ndarray: Filtered array.

    See Also:
        [`octavepass`][acoustic_toolbox.signal.octavepass]
    """
    if not isinstance(frequencies, Frequencies):
        frequencies = OctaveBand(center=frequencies, fraction=fraction)
    return bandpass_frequencies(
        x, fs, frequencies, order=order, purge=purge, zero_phase=zero_phase
    )


def third_octaves(
    p,
    fs,
    density: bool = False,
    frequencies=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
    ref: float = REFERENCE_PRESSURE,
) -> tuple[OctaveBand, np.ndarray]:
    """Calculate level per 1/3-octave in frequency domain using the FFT.

    Note:
        Exact center frequencies are always calculated.

    Args:
        p: Instantaneous signal $x(t)$.
        fs: Sample frequency.
        density: Calculate power density instead of power.
        frequencies: Center frequencies.
        ref: Reference pressure.

    Returns:
        OctaveBand: Instance of `OctaveBand`
        np.ndarray: Level array.

    See Also:
        [`NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES`][acoustic_toolbox.bands.NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES]
    """
    fob = OctaveBand(center=frequencies, fraction=3)
    f, p = power_spectrum(p, fs)
    fnb = EqualBand(f)
    power = integrate_bands(p, fnb, fob)
    if density:
        power /= fob.bandwidth / fnb.bandwidth
    level = 10.0 * np.log10(power / ref**2.0)
    return fob, level


def octaves(
    p,
    fs,
    density=False,
    frequencies=NOMINAL_OCTAVE_CENTER_FREQUENCIES,
    ref=REFERENCE_PRESSURE,
):
    """Calculate level per 1/1-octave in frequency domain using the FFT.

    Notes:
        - Based on power spectrum (FFT)
        - Exact center frequencies are always calculated.

    Args:
        p: Instantaneous signal $x(t)$.
        fs: Sample frequency.
        density: Calculate power density instead of power.
        frequencies: Center frequencies.
        ref: Reference value.

    Returns:
        OctaveBand: Instance of `OctaveBand`
        np.ndarray: Level array.

    See Also:
        [`NOMINAL_OCTAVE_CENTER_FREQUENCIES`][acoustic_toolbox.bands.NOMINAL_OCTAVE_CENTER_FREQUENCIES]
    """
    fob = OctaveBand(center=frequencies, fraction=1)
    f, p = power_spectrum(p, fs)
    fnb = EqualBand(f)
    power = integrate_bands(p, fnb, fob)
    if density:
        power /= fob.bandwidth / fnb.bandwidth
    level = 10.0 * np.log10(power / ref**2.0)
    return fob, level


def fractional_octaves(
    p,
    fs,
    start: float = 5.0,
    stop: float = 16000.0,
    fraction: int = 3,
    density: bool = False,
    ref: float = REFERENCE_PRESSURE,
):
    """Calculate level per 1/N-octave in frequency domain using the FFT. N is `fraction`.

    Notes:
        - Based on power spectrum (FFT)
        - This function does *not* use nominal center frequencies.
        - Exact center frequencies are always calculated.

    Args:
        p: Instantaneous signal $x(t)$.
        fs: Sample frequency.
        start: Start frequency.
        stop: Stop frequency.
        fraction: Fraction of fractional-octave band.
        density: Calculate power density instead of power.
        ref: Reference value.

    Returns:
        OctaveBand: Instance of `OctaveBand`
        np.ndarray: Level array.
    """
    fob = OctaveBand(fstart=start, fstop=stop, fraction=fraction)
    f, p = power_spectrum(p, fs)
    fnb = EqualBand(f)
    power = integrate_bands(p, fnb, fob)
    if density:
        power /= fob.bandwidth / fnb.bandwidth
    level = 10.0 * np.log10(power / ref**2.0)
    return fob, level


class Filterbank:
    """Fractional-Octave filter bank.

    Warning:
        For high frequencies the filter coefficients are wrong for low frequencies.
        Therefore, to improve the response for lower frequencies the signal should be downsampled.
        Currently, there is no easy way to do so within the Filterbank.

    Attributes:
        frequencies: Frequencies object.
            See also [`Frequencies`][acoustic_toolbox.signal.Frequencies] and subclasses.

            **Note:** A frequencies attribute should have the attributes center, lower, and upper.

        order: Filter order of Butterworth filter.
        sample_frequency: Sample frequency.
    """

    def __init__(
        self,
        frequencies: Frequencies,
        sample_frequency: float = 44100,
        order: int = 8,
    ):
        self.frequencies = frequencies
        self.order = order
        self.sample_frequency = sample_frequency

    @property
    def sample_frequency(self):
        """Sample frequency.

        Returns:
            float: Sample frequency.
        """
        return self._sample_frequency

    @sample_frequency.setter
    def sample_frequency(self, x):
        self._sample_frequency = x

    @property
    def filters(self):
        """Filters this filterbank consists of.

        Returns:
            generator: Filter coefficients for each band.
        """
        fs = self.sample_frequency
        return (
            bandpass_filter(
                self.frequencies._get_scalar(lower),
                self.frequencies._get_scalar(upper),
                fs,
                order=self.order,
                output="sos",
            )
            for lower, upper in zip(self.frequencies.lower, self.frequencies.upper)
        )

        # order = self.order
        # filters = list()
        # nyq = self.sample_frequency / 2.0
        # return ( butter(order, [lower/nyq, upper/nyq], btype='band', analog=False) for lower, upper in zip(self.frequencies.lower, self.frequencies.upper) )

    def lfilter(
        self, signal: NDArray[np.float64]
    ) -> Generator[NDArray[np.float64], None, None]:
        """Filter signal with filterbank.

        Note:
            This function uses [`scipy.signal.lfilter`][scipy.signal.lfilter].

        Args:
            signal: Signal to be filtered.

        Returns:
            generator: Filtered signal for each band.
        """
        return (sosfilt(sos, signal) for sos in self.filters)

    def filtfilt(
        self, signal: NDArray[np.float64]
    ) -> Generator[NDArray[np.float64], None, None]:
        """Filter signal with filterbank.

        Note:
            This function uses [`scipy.signal.filtfilt`][scipy.signal.filtfilt] and therefore has a zero-phase response.

        Args:
            signal: Signal to be filtered.

        Returns:
            List consisting of a filtered signal per filter.
        """
        return (_sosfiltfilt(sos, signal) for sos in self.filters)

    def power(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Power per band in signal.

        Args:
            signal: Signal to be analyzed.

        Returns:
            np.ndarray: Power per band.
        """
        filtered = self.filtfilt(signal)
        return np.array(
            [
                (x**2.0).sum() / len(x) / bw
                for x, bw in zip(filtered, self.frequencies.bandwidth)
            ]
        )

    def plot_response(self):
        """Plot frequency response.

        Note:
            The following phase response is obtained in case [`lfilter`][acoustic_toolbox.signal.Filterbank.lfilter] is used.
            The method [`filtfilt`][acoustic_toolbox.signal.Filterbank.filtfilt] has a zero-phase response.

        Returns:
            matplotlib.figure.Figure: Figure with frequency response plot.
        """
        fs = self.sample_frequency
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for f, fc in zip(self.filters, self.frequencies.center):
            w, h = freqz(f[0], f[1], int(fs / 2))  # np.arange(fs/2.0))
            ax1.semilogx(
                w / (2.0 * np.pi) * fs, 20.0 * np.log10(np.abs(h)), label=str(int(fc))
            )
            ax2.semilogx(w / (2.0 * np.pi) * fs, np.angle(h), label=str(int(fc)))
        ax1.set_xlabel(r"$f$ in Hz")
        ax1.set_ylabel(r"$|H|$ in dB re. 1")
        ax2.set_xlabel(r"$f$ in Hz")
        ax2.set_ylabel(r"$\angle H$ in rad")
        ax1.legend(loc=5)
        ax2.legend(loc=5)
        ax1.set_ylim(-60.0, +10.0)

        return fig

    def plot_power(self, signal):
        """Plot power in signal.

        Args:
            signal: Signal to be analyzed.

        Returns:
            matplotlib.figure.Figure: Figure with power plot.
        """
        f = self.frequencies.center
        p = self.power(signal)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        p = ax.bar(f, 20.0 * np.log10(p))
        ax.set_xlabel("$f$ in Hz")
        ax.set_ylabel("$L$ in dB re. 1")
        ax.set_xscale("log")

        return fig


def isolate(signals):
    """Isolate signals using Singular Value Decomposition.

    Args:
        signals: Array of shape N x M where N is the amount of samples and M the amount of signals. Thus, each column is a signal

    Returns:
        Array of isolated signals.
    """
    x = np.asarray(signals)

    W, s, v = np.linalg.svd((np.tile((x * x).sum(axis=0), (len(x), 1)) * x).dot(x.T))
    return v.T


def zero_crossings(data):
    """Determine the positions of zero crossings in `data`.

    Args:
        data: Vector.

    Returns:
        Vector with indices of samples *before* the zero crossing.
    """
    pos = data > 0
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]


def amplitude_envelope(signal: np.ndarray, fs, axis=-1):
    """Instantaneous amplitude of tone.

    The instantaneous amplitude is the magnitude of the analytic signal.

    Args:
        signal: Signal.
        fs: Sample frequency.
        axis: Axis. Defaults to -1.

    Returns:
        Amplitude envelope of `signal`.

    See Also:
        [`hilbert`][scipy.signal.hilbert]
    """
    return np.abs(hilbert(signal, axis=axis))


def instantaneous_phase(signal: np.ndarray, fs, axis=-1):
    """Instantaneous phase of tone.

    The instantaneous phase is the angle of the analytic signal.
    This function returns a wrapped angle.

    Args:
        signal: Signal.
        fs: Sample frequency.
        axis: Axis.

    Returns:
        Instantaneous phase of `signal`.

    See Also:
        [`hilbert`][scipy.signal.hilbert]
    """
    return np.angle(hilbert(signal, axis=axis))


def instantaneous_frequency(signal: np.ndarray, fs, axis=-1):
    """Determine instantaneous frequency of tone.

    The instantaneous frequency can be obtained by differentiating the unwrapped instantaneous phase.

    Args:
        signal: Signal.
        fs: Sample frequency.
        axis: Axis.

    Returns:
        Instantaneous frequency of `signal`.

    See Also:
        [`instantaneous_phase`][acoustic_toolbox.signal.instantaneous_phase]
    """
    return (
        np.diff(
            np.unwrap(instantaneous_phase(signal, fs, axis=axis), axis=axis), axis=axis
        )
        / (2.0 * np.pi)
        * fs
    )


def wvd(signal: np.ndarray, fs, analytic=True):
    r"""Wigner-Ville Distribution.

    $$
    W_z(n, \omega) = 2 \sum_k z^*[n-k]z[n+k] e^{-j\omega 2kT}
    $$

    Includes positive and negative frequencies.

    Args:
        signal: Signal.
        fs: Sample frequency.
        analytic: If True, uses the analytic signal. Defaults to True.

    Returns:
        Frequencies: Instance of `Frequencies`.
        W.T: Wigner-Ville distribution
    """
    signal = np.asarray(signal)

    N = int(len(signal) + len(signal) % 2)
    length_FFT = N  # Take an even value of N

    length_time = len(signal)

    if analytic:
        signal = hilbert(signal)
    s = np.concatenate((np.zeros(length_time), signal, np.zeros(length_time)))
    W = np.zeros((length_FFT, length_time))
    tau = np.arange(0, N // 2)

    R = np.zeros((N, length_time), dtype="float64")

    i = length_time
    for t in range(length_time):
        R[t, tau] = s[i + tau] * s[i - tau].conj()  # In one direction
        R[t, N - (tau + 1)] = R[t, tau + 1].conj()  # And the other direction
        i += 1
    W = np.fft.fft(R, length_FFT) / (2 * length_FFT)

    f = np.fft.fftfreq(N, 1.0 / fs)
    return f, W.T


def _sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=None, method="pad", irlen=None):
    """Filtfilt version using Second Order sections.

    Code is taken from [scipy.signal.sosfiltfilt](https://github.com/scipy/scipy/blob/main/scipy/signal/_filter_design.py#L1100)
    and adapted to make it work with SOS.

    Note that boradcasting does not work.
    """
    from scipy.signal import sosfilt_zi
    from scipy.signal._arraytools import axis_slice, axis_reverse

    x = np.asarray(x)

    if padlen is None:
        edge = 0
    else:
        edge = padlen

    # x's 'axis' dimension must be bigger than edge.
    if x.shape[axis] <= edge:
        raise ValueError(
            "The length of the input vector x must be at least "
            "padlen, which is %d." % edge
        )

    if padtype is not None and edge > 0:
        # Make an extension of length `edge` at each
        # end of the input array.
        if padtype == "even":
            ext = even_ext(x, edge, axis=axis)
        elif padtype == "odd":
            ext = odd_ext(x, edge, axis=axis)
        else:
            ext = const_ext(x, edge, axis=axis)
    else:
        ext = x

    # Get the steady state of the filter's step response.
    zi = sosfilt_zi(sos)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    # zi_shape = [1] * x.ndim
    # zi_shape[axis] = zi.size
    # zi = np.reshape(zi, zi_shape)
    x0 = axis_slice(ext, stop=1, axis=axis)
    # Forward filter.
    (y, zf) = sosfilt(sos, ext, axis=axis, zi=zi * x0)

    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    y0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = sosfilt(sos, axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

    # Reverse y.
    y = axis_reverse(y, axis=axis)

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y


def decimate(
    x: np.ndarray,
    q: int,
    n: int | None = None,
    ftype: str = "iir",
    axis: int = -1,
    zero_phase: bool = False,
) -> np.ndarray:
    """Downsample the signal by using a filter.

    By default, an order 8 Chebyshev type I filter is used. A 30 point FIR
    filter with hamming window is used if `ftype` is 'fir'.

    Notes:
        - The `zero_phase` keyword was added in v0.17.0.
        - The possibility to use instances of `lti` as `ftype` was added in v0.17.0

    Args:
        x: The signal to be downsampled.
        q: The downsampling factor.
        n: The order of the filter.
        ftype: The type of the lowpass filter.
        axis: The axis along which to decimate.
        zero_phase: Prevent phase shift by filtering with `filtfilt` instead of `lfilter`.

    Returns:
        The down-sampled signal.

    See Also:
        `resample`: For resampling the signal.
    """
    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if ftype == "fir":
        if n is None:
            n = 30
        system = lti(firwin(n + 1, 1.0 / q, window="hamming"), 1.0)

    elif ftype == "iir":
        if n is None:
            n = 8
        system = lti(*cheby1(n, 0.05, 0.8 / q))
    else:
        system = ftype

    if zero_phase:
        y = filtfilt(system.num, system.den, x, axis=axis)
    else:
        y = lfilter(system.num, system.den, x, axis=axis)

    sl = [slice(None)] * y.ndim
    sl[axis] = slice(None, None, q)
    return y[tuple(sl)]


def impulse_response_real_even(tf, ntaps):
    """The impulse response of a real and even frequency response is also real and even.

    A symmetric impulse response is needed. The center of symmetry determines the delay
    of the filter and thereby whether the filter is causal (delay>0, linear-phase) or
    non-causal (delay=0, linear-phase, zero-phase).

    Creating linear phase can be done by multiplying the magnitude with a complex
    exponential corresponding to the desired shift. Another method is to rotate the
    impulse response.

    [https://ccrma.stanford.edu/~jos/filters/Zero_Phase_Filters_Even_Impulse.html](https://ccrma.stanford.edu/~jos/filters/Zero_Phase_Filters_Even_Impulse.html)

    Args:
        tf: Real and even frequency response. Only positive frequencies.
        ntaps: Amount of taps.

    Returns:
        A real and even impulse response with length `ntaps`.
    """
    ir = np.fft.ifftshift(np.fft.irfft(tf, n=ntaps)).real
    return ir


def linear_phase(ntaps, steepness=1):
    """Compute linear phase delay for a single-sided spectrum.

    A linear phase delay can be added to an impulse response using the function [`np.fft.ifftshift`][numpy.fft.ifftshift].
    Sometimes, however, you would like to add the linear phase delay to the frequency response instead.
    This function computes the linear phase delay which can be multiplied with a single-sided frequency response.

    Args:
        ntaps: Amount of filter taps.
        steepness: Steepness of phase delay. Default value is 1, corresponding to delay in samples of `ntaps // 2`.

    Returns:
        np.ndarray: Linear phase delay.
    """
    f = np.fft.rfftfreq(ntaps, 1.0)  # Frequencies normalized to Nyquist.
    alpha = ntaps // 2 * steepness
    return np.exp(-1j * 2.0 * np.pi * f * alpha)


__all__ = [
    "bandpass",
    "bandpass_frequencies",
    "bandpass_fractional_octaves",
    "bandpass_octaves",
    "bandpass_third_octaves",
    "lowpass",
    "highpass",
    "octavepass",
    "octave_filter",
    "bandpass_filter",
    "convolve",
    "ir2fr",
    "decibel_to_neper",
    "neper_to_decibel",
    "EqualBand",
    "OctaveBand",
    "ms",
    "rms",
    "normalize",
    "window_scaling_factor",
    "apply_window",
    "amplitude_spectrum",
    "auto_spectrum",
    "power_spectrum",
    "angle_spectrum",
    "phase_spectrum",
    "density_spectrum",
    "integrate_bands",
    "octaves",
    "third_octaves",
    "fractional_octaves",
    "Filterbank",
    "isolate",
    "zero_crossings",
    "amplitude_envelope",
    "instantaneous_phase",
    "instantaneous_frequency",
    "wvd",
    "decimate",
]
