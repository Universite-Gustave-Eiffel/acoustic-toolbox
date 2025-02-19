"""Signal class.

This module contains the `Signal` class, which is a container for time-domain signals.

Attributes:
    fs: Sample frequency.
    samples: Amount of samples in signal.
    channels: Amount of channels in signal.
    duration: Duration of signal in seconds.
    values: Values of signal as instance of [`np.ndarray`][numpy.ndarray].

Methods:
    calibrate_to: Calibrate signal to value `decibel`.
    calibrate_with: Calibrate signal with other signal.
    decimate: Decimate signal by integer `factor`.
    resample: Resample signal.
    upsample: Upsample signal with integer factor.
    gain: Apply gain of `decibel` decibels.
    pick: Get signal from start time to stop time.
    times: Time vector.
    energy: Signal energy.
    power: Signal power.
    ms: Mean value squared of signal.
    rms: Root mean squared of signal.
    weigh: Apply frequency-weighting.
    instantaneous_frequency: Instantaneous frequency.
    instantaneous_phase: Instantaneous phase.
    detrend: Detrend signal.
    unwrap: Unwrap signal.
    complex_cepstrum: Complex cepstrum.
    real_cepstrum: Real cepstrum.
    power_spectrum: Power spectrum.
    angle_spectrum: Angle spectrum.
    phase_spectrum: Phase spectrum.
    peak: Peak value.
    peak_level: Peak level.
    min: Minimum value.
    max: Maximum value.
    max_level: Maximum level.
    sound_exposure: Sound exposure.
    sound_exposure_level: Sound exposure level.
    plot_complex_cepstrum: Plot complex cepstrum.
    plot_real_cepstrum: Plot real cepstrum.
    plot_power_spectrum: Plot power spectrum.
    plot_angle_spectrum: Plot angle spectrum.
    plot_phase_spectrum: Plot phase spectrum.
    spectrogram: Spectrogram.
    plot_spectrogram: Plot spectrogram.
    levels: Levels.
    leq: Leq.
    plot_levels: Plot levels.
    bandpass: Bandpass filter.
    bandstop: Bandstop filter.
    highpass: Highpass filter.
    lowpass: Lowpass filter.
    octavepass: Octavepass filter.
    third_octaves: Third octaves.
    fractional_octaves: Fractional octaves.
    plot_octaves: Plot octaves.
    plot_third_octaves: Plot third octaves.
    plot_fractional_octaves: Plot fractional octaves.
    plot: Plot signal.
    normalize: Normalize signal.
    to_wav: Write signal to WAV file.
    from_wav: Read signal from WAV file.

See Also:
    - [`acoustic_toolbox.signal`](signal.md)
"""

from __future__ import annotations
import itertools
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import (
    detrend,
    lfilter,
    bilinear,
    spectrogram,
    filtfilt,
    resample,
    fftconvolve,
)
from acoustic_toolbox import signal
from acoustic_toolbox.cepstrum import complex_cepstrum, real_cepstrum
from acoustic_toolbox import standards
from acoustic_toolbox.signal import Frequencies

from acoustic_toolbox.standards.iso_tr_25417_2007 import REFERENCE_PRESSURE
from acoustic_toolbox.standards.iec_61672_1_2013 import WEIGHTING_SYSTEMS
from acoustic_toolbox.standards.iec_61672_1_2013 import (
    NOMINAL_OCTAVE_CENTER_FREQUENCIES,
    NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
)


class Signal(np.ndarray):
    """A signal consisting of samples (array) and a sample frequency (float).

    Attributes:
        fs: Sample frequency.
        samples: Amount of samples in signal.
        channels: Amount of channels in signal.
        duration: Duration of signal in seconds.
        values: Values of signal as instance of [`np.ndarray`][numpy.ndarray].
    """

    def __new__(cls, data, fs):
        """Create a new signal.

        Args:
            data: Signal values.
            fs: Sample frequency.
        """
        obj = np.asarray(data).view(cls)
        obj.fs = fs
        return obj

    def __array_prepare__(self, array, context=None):
        try:
            # Best guess: the context here is the existing signal, which may have multiple channels.
            # This is checking for (1) is it two channels and (2) are the sample frequencies the same.
            # Issues: only works for two channels, not n channels (and just seems inelegant).
            a = context[1][0]
            b = context[1][1]
        except IndexError:
            return array

        if hasattr(a, "fs") and hasattr(b, "fs"):
            if a.fs == b.fs:
                return array
            else:
                raise ValueError("Sample frequencies do not match.")
        else:
            return array

    def __array_wrap__(self, out_arr, context=None, return_scalar: bool = False):
        if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
            return np.ndarray.__array_wrap__(self, out_arr, context, return_scalar)
        else:
            return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        self.fs = getattr(obj, "fs", None)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Signal, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.fs,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.fs = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(Signal, self).__setstate__(state[0:-1])

    def __repr__(self):
        return "Signal({})".format(str(self))

    def _construct(self, x):
        """Construct signal like x."""
        return Signal(x, self.fs)

    @property
    def samples(self):
        """Amount of samples in signal."""
        return self.shape[-1]

    @property
    def channels(self):
        """Amount of channels."""
        if self.ndim > 1:
            return self.shape[-2]
        else:
            return 1

    @property
    def duration(self) -> float:
        """Duration of signal in seconds."""
        return float(self.samples / self.fs)

    @property
    def values(self) -> np.ndarray:
        """Return the values of this signal as an instance of [`np.ndarray`][numpy.ndarray]."""
        return np.array(self)

    def calibrate_to(self, decibel, inplace: bool = False) -> Signal:
        """Calibrate signal to value `decibel`.

        Tip:
            Values of `decibel` are broadcasted. To set a value per channel, use `decibel[...,None]`.

        Args:
            decibel: Value to calibrate to.
            inplace: Whether to perform inplace or not.

        Returns:
            Calibrated signal.
        """
        decibel = decibel * np.ones(self.shape)
        gain = decibel - self.leq()[..., None]
        return self.gain(gain, inplace=inplace)

    def calibrate_with(self, other: Signal, decibel, inplace: bool = False) -> Signal:
        """Calibrate signal with other signal.

        Args:
          other: Other signal/array.
          decibel: Signal level of `other`.
          inplace: Whether to perform inplace or not.

        Returns:
          Calibrated signal.
        """
        if not isinstance(other, Signal):
            other = Signal(other, self.fs)
        gain = decibel - other.leq()
        return self.gain(gain, inplace=inplace)

    def decimate(self, factor, zero_phase=False, ftype="iir", order=None) -> Signal:
        """Decimate signal by integer `factor`. Before downsampling a low-pass filter is applied.

        Args:
            factor: Downsampling factor.
            zero_phase: Prevent phase shift by filtering with ``filtfilt`` instead of ``lfilter``.
            ftype: Filter type.
            order: Filter order.

        Returns:
            Decimated signal.

        .. seealso:: :func:`scipy.signal.decimate`
        .. seealso:: :meth:`resample`: Decimated signal.

        """
        return Signal(
            signal.decimate(
                x=self, q=factor, n=order, ftype=ftype, zero_phase=zero_phase
            ),
            self.fs / factor,
        )

    def resample(self, nsamples, times=None, axis=-1, window=None) -> Signal:
        """Resample signal.

        Tip:
            You might want to low-pass filter this signal before resampling.

        Args:
          samples: New amount of samples.
          times: Times corresponding to samples.
          axis: Axis.
          window: Window.

        Returns:
            Resampled signal.
        See Also:
            - [`scipy.signal.resample`][scipy.signal.resample]
            - [`decimate`][acoustic_toolbox._signal.Signal.decimate]

        """
        return Signal(
            resample(self, nsamples, times, axis, window),
            nsamples / self.samples * self.fs,
        )

    def upsample(self, factor: int, axis: int = -1) -> Signal:
        """Upsample signal with integer factor.

        Args:
          factor: Upsample factor.
          axis: Axis.

        See Also:
            - [`resample`][acoustic_toolbox._signal.Signal.resample]
        """
        return self.resample(int(self.samples * factor), axis=axis)

    def gain(self, decibel, inplace: bool = False) -> Signal:
        """Apply gain of `decibel` decibels.

        Args:
          decibel: Decibels
          inplace: In place

        Returns:
          Amplified signal.

        """
        factor = 10.0 ** (decibel / 20.0)
        if inplace:
            self *= factor
            return self
        else:
            return self * factor

    def pick(self, start: float = 0.0, stop: float | None = None) -> Signal:
        """Get signal from start time to stop time.

        Args:
          start: Start time.
          stop: End time.

        Returns:
          Selected part of the signal.

        """
        if start is not None:
            start = int(np.floor(start * self.fs))
        if stop is not None:
            stop = int(np.floor(stop * self.fs))
        return self[..., start:stop]

    def times(self) -> np.ndarray:
        """Time vector.

        Returns:
          A vector with a timestamp for each sample.

        """
        return np.arange(0, self.samples) / self.fs

    def energy(self) -> np.ndarray:
        r"""Signal energy.

        Returns:
            Total energy per channel.

            $$
            E = \sum_{n=0}^{N-1} |x_n|^2
            $$

        """
        return float((self * self).sum())

    def power(self):
        r"""Signal power.

        $$
        P = \frac{1}{N} \sum_{n=0}^{N-1} |x_n|^2
        $$
        """
        return self.energy() / len(self)

    def ms(self):
        """Mean value squared of signal.

        See Also:
            [`acoustic_toolbox.signal.ms`][acoustic_toolbox._signal.Signal.ms]
        """
        return signal.ms(self)

    def rms(self):
        """Root mean squared of signal.

        See Also:
            [`acoustic_toolbox.signal.rms`][acoustic_toolbox._signal.Signal.rms]
        """
        return signal.rms(self)
        # return np.sqrt(self.power())

    def weigh(self, weighting: str = "A", zero_phase: bool = False) -> Signal:
        """Apply frequency-weighting. By default 'A'-weighting is applied.

        Note:
            By default the weighting filter is applied using
            [`scipy.signal.lfilter`][scipy.signal.lfilter] causing a frequency-dependent delay.

            In case a delay is undesired, the filter can be applied using
            [`scipy.signal.filtfilt`][scipy.signal.filtfilt] by setting `zero_phase=True`.

        Args:
            weighting: Frequency-weighting filter to apply.
                       Valid options are 'A', 'C' and 'Z'. Default weighting is 'A'.
            zero_phase: Prevent phase shift by filtering with ``filtfilt`` instead of ``lfilter``.

        Returns:
            Weighted signal.
        """
        num, den = WEIGHTING_SYSTEMS[weighting]()
        b, a = bilinear(num, den, self.fs)
        func = filtfilt if zero_phase else lfilter
        return self._construct(func(b, a, self))

    def correlate(self, other: Signal | None = None, mode: str = "full"):
        """Correlate signal with `other` signal. In case `other==None` this
        method returns the autocorrelation.

        Args:
          other: Other signal.
          mode: Mode.

        Raises:
            ValueError: If sample frequencies are not the same.
            ValueError: If not supported for multichannel signals.

        See Also:
            - [`np.correlate`][numpy.correlate]
            - [`scipy.signal.fftconvolve`][scipy.signal.fftconvolve]
        """
        if other is None:
            other = self
        if self.fs != other.fs:
            raise ValueError("Cannot correlate. Sample frequencies are not the same.")
        if self.channels > 1 or other.channels > 1:
            raise ValueError(
                "Cannot correlate. Not supported for multichannel signals."
            )
        return self._construct(fftconvolve(self, other[::-1], mode=mode))

    def amplitude_envelope(self) -> Signal:
        """Amplitude envelope.

        Returns:
            Amplitude envelope.

        See Also:
            - [`acoustic_toolbox.signal.amplitude_envelope`][acoustic_toolbox._signal.Signal.amplitude_envelope]

        """
        return self._construct(signal.amplitude_envelope(self, self.fs))

    def instantaneous_frequency(self) -> Signal:
        """Instantaneous frequency.

        Returns:
            Instantaneous frequency of signal

        See Also:
            - [`acoustic_toolbox.signal.instantaneous_frequency`][acoustic_toolbox._signal.Signal.instantaneous_frequency]


        """
        return self._construct(signal.instantaneous_frequency(self, self.fs))

    def instantaneous_phase(self) -> Signal:
        """Instantaneous phase.

        Returns:
            Instantaneous phase of signal

        See Also:
            - [`acoustic_toolbox.signal.instantaneous_phase`][acoustic_toolbox._signal.Signal.instantaneous_phase]
        """
        return self._construct(signal.instantaneous_phase(self, self.fs))

    def detrend(self, **kwargs) -> Signal:
        """Detrend signal.

        Args:
            **kwargs:

        Returns:
            Detrended signal.

        See Also:
            - [`scipy.signal.detrend`][scipy.signal.detrend]

        """
        return self._construct(detrend(self, **kwargs))

    def unwrap(self) -> Signal:
        """Unwrap signal in case the signal represents wrapped phase.

        Returns:
            Unwrapped signal.

        See Also:
            - [`np.unwrap`][numpy.unwrap]
        """
        return self._construct(np.unwrap(self))

    def complex_cepstrum(
        self, N: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Complex cepstrum.

        Args:
            N: Amount of bins.

        Returns:
            Quefrency
            Complex cepstrum
            Delay in amount of samples.

        See Also:
            - [`acoustic_toolbox.cepstrum.complex_cepstrum`][acoustic_toolbox._signal.Signal.complex_cepstrum]

        """
        if N is not None:
            times = np.linspace(0.0, self.duration, N, endpoint=False)
        else:
            times = self.times()
        cepstrum, ndelay = complex_cepstrum(self, n=N)
        return times, cepstrum, ndelay

    def real_cepstrum(self, N=None):
        """Real cepstrum.

        Args:
          N: Amount of bins.

        Returns:
          Quefrency
          Real cepstrum.

        See Also:
            - [`acoustic_toolbox.cepstrum.real_cepstrum`][acoustic_toolbox._signal.Signal.real_cepstrum]

        """
        if N is not None:
            times = np.linspace(0.0, self.duration, N, endpoint=False)
        else:
            times = self.times()
        return times, real_cepstrum(self, n=N)

    def power_spectrum(self, N: int | None = None):
        """Power spectrum.

        Args:
            N: Amount of bins.

        See Also:
            - [`acoustic_toolbox.signal.power_spectrum`][acoustic_toolbox._signal.Signal.power_spectrum]
        """
        return signal.power_spectrum(self, self.fs, N=N)

    def angle_spectrum(self, N: int | None = None):
        """Phase angle spectrum. Wrapped.

        Args:
          N: amount of bins.

        See Also:
            - [`acoustic_toolbox.signal.angle_spectrum`][acoustic_toolbox._signal.Signal.angle_spectrum]
            - [`acoustic_toolbox.signal.phase_spectrum`][acoustic_toolbox._signal.Signal.phase_spectrum]
            - [`phase_spectrum`][acoustic_toolbox._signal.Signal.phase_spectrum]
        """
        return signal.angle_spectrum(self, self.fs, N=N)

    def phase_spectrum(self, N=None):
        """Phase spectrum. Unwrapped.

        Args:
          N: Amount of bins.

        See Also:
            - [`acoustic_toolbox.signal.phase_spectrum`][acoustic_toolbox._signal.Signal.phase_spectrum]
            - [`acoustic_toolbox.signal.angle_spectrum`][acoustic_toolbox._signal.Signal.angle_spectrum]
            - [`angle_spectrum`][acoustic_toolbox._signal.Signal.angle_spectrum]
        """
        return signal.phase_spectrum(self, self.fs, N=N)

    def peak(self, axis=-1):
        """Peak sound pressure.

        Args:
          axis: Axis.

        See Also:
            - [`acoustic_toolbox.standards.iso_tr_25417_2007.peak_sound_pressure`][acoustic_toolbox.standards.iso_tr_25417_2007.peak_sound_pressure]
        """
        return standards.iso_tr_25417_2007.peak_sound_pressure(self, axis=axis)

    def peak_level(self, axis=-1):
        """Peak sound pressure level.

        Args:
          axis: Axis.

        See Also:
            - [`acoustic_toolbox.standards.iso_tr_25417_2007.peak_sound_pressure_level`][acoustic_toolbox.standards.iso_tr_25417_2007.peak_sound_pressure_level]
        """
        return standards.iso_tr_25417_2007.peak_sound_pressure_level(self, axis=axis)

    def min(self, axis=-1):
        """Return the minimum along a given axis.

        Args:
          axis: Axis.

        See Also:
            - Refer to [`np.amin`][numpy.amin] for full documentation.
        """
        return np.ndarray.min(self, axis=axis)

    def max(self, axis=-1):
        """Return the maximum along a given axis.

        Args:
          axis: Axis.

        See Also:
            - Refer to [`np.amax`][numpy.amax] for full documentation.
        """
        return np.ndarray.max(self, axis=axis)

    def max_level(self, axis=-1):
        """Maximum sound pressure level.

        Args:
          axis: Axis.

        See Also:
            - [`acoustic_toolbox.standards.iso_tr_25417_2007.max_sound_pressure_level`][acoustic_toolbox.standards.iso_tr_25417_2007.max_sound_pressure_level]
        """
        return standards.iso_tr_25417_2007.max_sound_pressure_level(self, axis=axis)

    def sound_exposure(self, axis=-1):
        """Sound exposure.

        Args:
          axis: Axis.

        See Also:
            - [`acoustic_toolbox.standards.iso_tr_25417_2007.sound_exposure`][acoustic_toolbox.standards.iso_tr_25417_2007.sound_exposure]
        """
        return standards.iso_tr_25417_2007.sound_exposure(self, self.fs, axis=axis)

    def sound_exposure_level(self, axis=-1):
        """Sound exposure level.

        Args:
          axis: Axis.

        See Also:
            - [`acoustic_toolbox.standards.iso_tr_25417_2007.sound_exposure_level`][acoustic_toolbox.standards.iso_tr_25417_2007.sound_exposure_level]
        """
        return standards.iso_tr_25417_2007.sound_exposure_level(
            self, self.fs, axis=axis
        )

    def plot_complex_cepstrum(self, N: int | None = None, **kwargs):
        """Plot complex cepstrum of signal.

        Args:
          N: Amount of bins.
          **kwargs:

        **Valid kwargs:**

        Other Args:
            xscale: X-axis scale.
            yscale: Y-axis scale.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            frequency (bool): Boolean indicating whether the x-axis should show time in seconds or quefrency
            xlabel_frequency: Label in case frequency is shown.
        """
        params = {
            "xscale": "linear",
            "yscale": "linear",
            "xlabel": "$t$ in s",
            "ylabel": "$C$",
            "title": "Complex cepstrum",
            "frequency": False,
            "xlabel_frequency": "$f$ in Hz",
        }
        params.update(kwargs)

        t, ceps, _ = self.complex_cepstrum(N=N)
        if params["frequency"]:
            t = 1.0 / t
            params["xlabel"] = params["xlabel_frequency"]
            t = t[::-1]
            ceps = ceps[::-1]
        return _base_plot(t, ceps, params)

    def plot_real_cepstrum(self, N: int | None = None, **kwargs):
        """Plot real cepstrum of signal.

        Args:
          N: Amount of bins.
          **kwargs:

        **Valid kwargs:**

        Other Args:
            xscale: X-axis scale.
            yscale: Y-axis scale.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            frequency (bool): Boolean indicating whether the x-axis should show time in seconds or quefrency
            xlabel_frequency: Label in case frequency is shown.
        """
        params = {
            "xscale": "linear",
            "yscale": "linear",
            "xlabel": "$t$ in s",
            "ylabel": "$C$",
            "title": "Real cepstrum",
            "frequency": False,
            "xlabel_frequency": "$f$ in Hz",
        }
        params.update(kwargs)

        t, ceps = self.real_cepstrum(N=N)
        if params["frequency"]:
            t = 1.0 / t
            params["xlabel"] = params["xlabel_frequency"]
            t = t[::-1]
            ceps = ceps[::-1]
        return _base_plot(t, ceps, params)

    def plot_power_spectrum(self, N=None, **kwargs):  # filename=None, scale='log'):
        """Plot spectrum of signal.

        Args:
          N: Amount of bins.
          **kwargs:

        **Valid kwargs:**

        Other Args:
            xscale: X-axis scale.
            yscale: Y-axis scale.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            reference: Reference power

        See Also:
            - [`acoustic_toolbox.signal.power_spectrum`][acoustic_toolbox._signal.Signal.power_spectrum]
        """
        params = {
            "xscale": "log",
            "yscale": "linear",
            "xlabel": "$f$ in Hz",
            "ylabel": "$L_{p}$ in dB",
            "title": "SPL",
            "reference": REFERENCE_PRESSURE**2.0,
        }
        params.update(kwargs)

        f, o = self.power_spectrum(N=N)
        return _base_plot(f, 10.0 * np.log10(o / params["reference"]), params)

    def plot_angle_spectrum(self, N: int | None = None, **kwargs):
        """Plot phase angle spectrum of signal. Wrapped.

        Args:
          N: Amount of bins.
          **kwargs:

        **Valid kwargs:**

        Other Args:
            xscale: X-axis scale.
            yscale: Y-axis scale.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            reference: Reference power
        """
        params = {
            "xscale": "linear",
            "yscale": "linear",
            "xlabel": "$f$ in Hz",
            "ylabel": r"$\angle \phi$",
            "title": "Phase response (wrapped)",
        }
        params.update(kwargs)
        f, o = self.angle_spectrum(N=N)
        return _base_plot(f, o, params)

    def plot_phase_spectrum(self, N: int | None = None, **kwargs):
        """Plot phase spectrum of signal. Unwrapped.

        Args:
          N: Amount of bins.
          **kwargs:

        **Valid kwargs:**

        Other Args:
            xscale: X-axis scale.
            yscale: Y-axis scale.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            reference: Reference power
        """
        params = {
            "xscale": "linear",
            "yscale": "linear",
            "xlabel": "$f$ in Hz",
            "ylabel": r"$\angle \phi$",
            "title": "Phase response (unwrapped)",
        }
        params.update(kwargs)
        f, o = self.phase_spectrum(N=N)
        return _base_plot(f, o, params)

    def spectrogram(self, **kwargs):
        """Spectrogram of signal.

        Returns:
          Time
          Frequency
          Power

        See Also:
            See [`scipy.signal.spectrogram`][scipy.signal.spectrogram]. Some of the default values have been changed.
            The generated spectrogram consists by default of complex values.

        """
        params = {
            "nfft": 4096,
            "noverlap": 128,
            "mode": "complex",
        }
        params.update(kwargs)

        t, s, P = spectrogram(self, fs=self.fs, **params)

        return t, s, P

    def plot_spectrogram(self, **kwargs):
        """Plot spectrogram of the signal.

        Note:
            This method only works for a single channel.

        **Valid kwargs:**

        Other Args:
            xlim: X-axis limits.
            ylim: Y-axis limits.
            clim: Color limits.
            NFFT: Amount of FFT bins.
            noverlap: Amount of overlap between FFT bins.
            title: Title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            clabel: Color label.
            colorbar: Whether to show the colorbar.
        """
        # TODO: use `spectrogram`.
        params = {
            "xlim": None,
            "ylim": None,
            "clim": None,
            "NFFT": 4096,
            "noverlap": 128,
            "title": "Spectrogram",
            "xlabel": "$t$ in s",
            "ylabel": "$f$ in Hz",
            "clabel": "SPL in dB",
            "colorbar": True,
        }
        params.update(kwargs)

        if self.channels > 1:
            raise ValueError(
                "Cannot plot spectrogram of multichannel signal. Please select a single channel."
            )

        # Check if an axes object is passed in. Otherwise, create one.
        ax0 = params.get("ax", plt.figure().add_subplot(111))
        ax0.set_title(params["title"])

        data = np.squeeze(self)
        try:
            _, _, _, im = ax0.specgram(
                data,
                Fs=self.fs,
                noverlap=params["noverlap"],
                NFFT=params["NFFT"],
                mode="magnitude",
                scale_by_freq=False,
            )
        except AttributeError:
            raise NotImplementedError(
                "Your version of matplotlib is incompatible due to lack of support of the mode keyword argument to matplotlib.mlab.specgram."
            )

        if params["colorbar"]:
            cb = ax0.get_figure().colorbar(mappable=im)
            cb.set_label(params["clabel"])

        ax0.set_xlim(params["xlim"])
        ax0.set_ylim(params["ylim"])
        im.set_clim(params["clim"])

        ax0.set_xlabel(params["xlabel"])
        ax0.set_ylabel(params["ylabel"])

        return ax0

    def levels(self, time: float = 0.125, method: str = "average"):
        """Calculate sound pressure level as function of time.

        Args:
            time: Averaging time or integration time constant. Default value is 0.125 corresponding to FAST.
            method: Use time `average` or time `weighting`.

        Returns:
            sound pressure level as function of time.

        See Also:
            - [`acoustic_toolbox.standards.iec_61672_1_2013.time_averaged_sound_level`][acoustic_toolbox.standards.iec_61672_1_2013.time_averaged_sound_level]
            - [`acoustic_toolbox.standards.iec_61672_1_2013.time_weighted_sound_level`][acoustic_toolbox.standards.iec_61672_1_2013.time_weighted_sound_level]

        """
        if method == "average":
            return standards.iec_61672_1_2013.time_averaged_sound_level(
                self.values, self.fs, time
            )
        elif method == "weighting":
            return standards.iec_61672_1_2013.time_weighted_sound_level(
                self.values, self.fs, time
            )
        else:
            raise ValueError("Invalid method")

    def leq(self):
        """Equivalent level. Single-value number.

        See Also:
            - [`acoustic_toolbox.standards.iso_tr_25417_2007.equivalent_sound_pressure_level`][acoustic_toolbox.standards.iso_tr_25417_2007.equivalent_sound_pressure_level]
        """
        return standards.iso_tr_25417_2007.equivalent_sound_pressure_level(self.values)

    def plot_levels(self, **kwargs):
        """Plot sound pressure level as function of time.

        See Also:
            - [`levels`][acoustic_toolbox._signal.Signal.levels]
        """
        params = {
            "xscale": "linear",
            "yscale": "linear",
            "xlabel": "$t$ in s",
            "ylabel": "$L_{p,F}$ in dB",
            "title": "SPL",
            "time": 0.125,
            "method": "average",
            "labels": None,
        }
        params.update(kwargs)
        t, L = self.levels(params["time"], params["method"])
        L_masked = np.ma.masked_where(np.isinf(L), L)
        return _base_plot(t, L_masked, params)

    # def octave(self, frequency, fraction=1):
    # """Determine fractional-octave `fraction` at `frequency`.

    # .. seealso:: :func:`acoustic_toolbox.signal.fractional_octaves`

    # """
    # return acoustic_toolbox.signal.fractional_octaves(self, self.fs, frequency,
    # frequency, fraction, False)[1]

    def bandpass(self, lowcut, highcut, order=8, zero_phase=False) -> Signal:
        """Filter signal with band-pass filter.

        Args:
            lowcut: Lower cornerfrequency.
            highcut: Upper cornerfrequency.
            order: Filter order. (Default value = 8)
            zero_phase: Prevent phase error by filtering in both directions (filtfilt). (Default value = False)

        Returns:
            class:`Signal`.

        See Also:
            - [`acoustic_toolbox.signal.bandpass`][acoustic_toolbox.signal.bandpass]

        """
        return type(self)(
            signal.bandpass(
                self, lowcut, highcut, self.fs, order=order, zero_phase=zero_phase
            ),
            self.fs,
        )

    def bandstop(
        self, lowcut, highcut, order: int = 8, zero_phase: bool = False
    ) -> Signal:
        """Filter signal with band-stop filter.

        Args:
            lowcut: Lower cornerfrequency.
            highcut: Upper cornerfrequency.
            order: Filter order.
            zero_phase: Prevent phase error by filtering in both directions (filtfilt).

        Returns:
            Band-pass filtered signal

        See Also:
            - [`acoustic_toolbox.signal.bandstop`][acoustic_toolbox.signal.bandstop]

        """
        return type(self)(
            signal.bandstop(
                self, lowcut, highcut, self.fs, order=order, zero_phase=zero_phase
            ),
            self.fs,
        )

    def highpass(self, cutoff, order: int = 4, zero_phase: bool = False) -> Signal:
        """Filter signal with high-pass filter.

        Args:
            cutoff: Cornerfrequency.
            order: Filter order.
            zero_phase: Prevent phase error by filtering in both directions (filtfilt).

        Returns:
            High-pass filtered signal

        See Also:
            - [`acoustic_toolbox.signal.highpass`][acoustic_toolbox.signal.highpass]

        """
        return type(self)(
            signal.highpass(self, cutoff, self.fs, order=order, zero_phase=zero_phase),
            self.fs,
        )

    def lowpass(self, cutoff, order: int = 4, zero_phase: bool = False) -> Signal:
        """Filter signal with low-pass filter.

        Args:
            cutoff: Cornerfrequency.
            order: Filter order.
            zero_phase: Prevent phase error by filtering in both directions (filtfilt).

        Returns:
            Low-pass filtered signal

        See Also:
            - [`acoustic_toolbox.signal.lowpass`][acoustic_toolbox.signal.lowpass]

        """
        return type(self)(
            signal.lowpass(self, cutoff, self.fs, order=order, zero_phase=zero_phase),
            self.fs,
        )

    def octavepass(
        self, center, fraction, order: int = 8, zero_phase: bool = False
    ) -> Signal:
        """Filter signal with fractional-octave band-pass filter.

        Args:
            center: Center frequency. Any value in the band will suffice.
            fraction: Band designator.
            order: Filter order.
            zero_phase: Prevent phase error by filtering in both directions (filtfilt).

        Returns:
            Band-pass filtered signal

        See Also:
            - [`acoustic_toolbox.signal.octavepass`][acoustic_toolbox.signal.octavepass]

        """
        return type(self)(
            signal.octavepass(
                self,
                center,
                self.fs,
                fraction=fraction,
                order=order,
                zero_phase=zero_phase,
            ),
            self.fs,
        )

    def bandpass_frequencies(
        self,
        frequencies: "Frequencies",
        order: int = 8,
        purge: bool = True,
        zero_phase: bool = False,
    ) -> tuple["Frequencies", Signal]:
        """Apply bandpass filters for frequencies.

        Args:
            frequencies: Instance of :class:`acoustic_toolbox.signal.Frequencies`
            order: Filter order.
            purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
            zero_phase: Prevent phase error by filtering in both directions (filtfilt).

        Returns:
            Frequencies
            Band-pass filtered signal.

        See Also:
            - [`acoustic_toolbox.signal.bandpass_frequencies`][acoustic_toolbox.signal.bandpass_frequencies]

        """
        frequencies, filtered = signal.bandpass_frequencies(
            self, self.fs, frequencies, order, purge, zero_phase=zero_phase
        )
        return frequencies, type(self)(filtered, self.fs)

    def octaves(
        self,
        frequencies: "Frequencies" | np.ndarray = NOMINAL_OCTAVE_CENTER_FREQUENCIES,
        order: int = 8,
        purge: bool = True,
        zero_phase: bool = False,
    ):
        """Apply 1/1-octaves bandpass filters.

        Args:
            frequencies: Band-pass filter frequencies.
            order: Filter order.
            purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
            zero_phase: Prevent phase error by filtering in both directions (filtfilt).

        Returns:
            Frequencies
            Band-pass filtered signal.

        See Also:
            - [`acoustic_toolbox.signal.bandpass_octaves`][acoustic_toolbox.signal.bandpass_octaves]
        """
        frequencies, octaves = signal.bandpass_octaves(
            self, self.fs, frequencies, order, purge, zero_phase=zero_phase
        )
        return frequencies, type(self)(octaves, self.fs)

    def third_octaves(
        self,
        frequencies: "Frequencies" = NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
        order: int = 8,
        purge: bool = True,
        zero_phase: bool = False,
    ):
        """Apply 1/3-octaves bandpass filters.

        Args:
            frequencies: Band-pass filter frequencies.
            order: Filter order.
            purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
            zero_phase: Prevent phase error by filtering in both directions (filtfilt).

        Returns:
          Frequencies and band-pass filtered signal.

        See Also:
            - [`acoustic_toolbox.signal.bandpass_third_octaves`][acoustic_toolbox.signal.bandpass_third_octaves]

        """
        frequencies, octaves = signal.bandpass_third_octaves(
            self, self.fs, frequencies, order, purge, zero_phase=zero_phase
        )
        return frequencies, type(self)(octaves, self.fs)

    def fractional_octaves(
        self,
        frequencies: "Frequencies" | None = None,
        fraction: int = 1,
        order: int = 8,
        purge: bool = True,
        zero_phase: bool = False,
    ):
        """Apply 1/N-octaves bandpass filters.

        Args:
            frequencies: Band-pass filter frequencies.
            fraction: Default band-designator of fractional-octaves.
            order: Filter order.
            purge: Discard bands of which the upper corner frequency is above the Nyquist frequency.
            zero_phase: Prevent phase error by filtering in both directions (filtfilt).

        Returns:
          Frequencies and band-pass filtered signal.

        See Also:
            - [`acoustic_toolbox.signal.bandpass_fractional_octaves`][acoustic_toolbox.signal.bandpass_fractional_octaves]

        """
        if frequencies is None:
            frequencies = signal.OctaveBand(
                fstart=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES[0],
                fstop=self.fs / 2.0,
                fraction=fraction,
            )
        frequencies, octaves = signal.bandpass_fractional_octaves(
            self, self.fs, frequencies, fraction, order, purge, zero_phase=zero_phase
        )
        return frequencies, type(self)(octaves, self.fs)

    def plot_octaves(self, **kwargs):
        """Plot octaves.

        See Also:
            - [`octaves`][acoustic_toolbox._signal.Signal.octaves]
        """
        params = {
            "xscale": "log",
            "yscale": "linear",
            "xlabel": "$f$ in Hz",
            "ylabel": "$L_{p}$ in dB",
            "title": "1/1-Octaves SPL",
        }
        params.update(kwargs)
        f, o = self.octaves()
        print(len(f.center), len(o.leq()))
        return _base_plot(f.center, o.leq().T, params)

    def plot_third_octaves(self, **kwargs):
        """Plot 1/3-octaves.

        See Also:
            - [`third_octaves`][acoustic_toolbox._signal.Signal.third_octaves]
        """
        params = {
            "xscale": "log",
            "yscale": "linear",
            "xlabel": "$f$ in Hz",
            "ylabel": "$L_{p}$ in dB",
            "title": "1/3-Octaves SPL",
        }
        params.update(kwargs)
        f, o = self.third_octaves()
        return _base_plot(f.center, o.leq().T, params)

    def plot_fractional_octaves(
        self,
        frequencies=None,
        fraction=1,
        order=8,
        purge=True,
        zero_phase=False,
        **kwargs,
    ):
        """Plot fractional octaves."""
        title = "1/{}-Octaves SPL".format(fraction)

        params = {
            "xscale": "log",
            "yscale": "linear",
            "xlabel": "$f$ in Hz",
            "ylabel": "$L_p$ in dB",
            "title": title,
        }
        params.update(kwargs)
        f, o = self.fractional_octaves(
            frequencies=frequencies,
            fraction=fraction,
            order=order,
            purge=purge,
            zero_phase=zero_phase,
        )
        return _base_plot(f.center, o.leq().T, params)

    def plot(self, **kwargs):
        """Plot signal as function of time. By default the entire signal is plotted.

        **Valid kwargs:**

        Other Args:
            filename: Name of file.
            start: First sample index.
            stop: Last sample index.
        """
        params = {
            "xscale": "linear",
            "yscale": "linear",
            "xlabel": "$t$ in s",
            "ylabel": "$x$ in -",
            "title": "Signal",
        }
        params.update(kwargs)
        return _base_plot(self.times(), self, params)

    # def plot_scalo(self, filename=None):
    # """
    # Plot scalogram
    # """
    # from scipy.signal import ricker, cwt

    # wavelet = ricker
    # widths = np.logspace(-1, 3.5, 10)
    # x = cwt(self, wavelet, widths)

    # interpolation = 'nearest'

    # from matplotlib.ticker import LinearLocator, AutoLocator, MaxNLocator
    # majorLocator = LinearLocator()
    # majorLocator = MaxNLocator()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_title('Scaleogram')
    ##ax.set_xticks(np.arange(0, x.shape[1])*self.fs)
    ##ax.xaxis.set_major_locator(majorLocator)

    ##ax.imshow(10.0 * np.log10(x**2.0), interpolation=interpolation, aspect='auto', origin='lower')#, extent=[0, 1, 0, len(x)])
    # ax.pcolormesh(np.arange(0.0, x.shape[1])/self.fs, widths, 10.0*np.log(x**2.0))
    # if filename:
    # fig.savefig(filename)
    # else:
    # return fig

    # def plot_scaleogram(self, filename):
    # """
    # Plot scaleogram
    # """
    # import pywt

    # wavelet = 'dmey'
    # level = pywt.dwt_max_level(len(self), pywt.Wavelet(wavelet))
    # print level
    # level = 20
    # order = 'freq'
    # interpolation = 'nearest'

    # wp = pywt.WaveletPacket(self, wavelet, 'sym', maxlevel=level)
    # nodes = wp.get_level(level, order=order)
    # labels = [n.path for n in nodes]
    # values = np.abs(np.array([n.data for n in nodes], 'd'))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_title('Scaleogram')
    # ax.imshow(values, interpolation=interpolation, aspect='auto', origin='lower', extent=[0, 1, 0, len(values)])
    ##ax.set_yticks(np.arange(0.5, len(labels) + 0.5))
    ##ax.set_yticklabels(labels)

    # fig.savefig(filename)

    def normalize(self, gap=6.0, inplace=False):
        """Normalize signal.

        The parameter `gap` can be understood as using `gap` decibels fewer for the dynamic range.

        By default a 6 decibel gap is used.

        Args:
          gap: Gap between maximum value and ceiling in decibel.
          inplace: Normalize signal in place.
        """
        factor = np.abs(self).max() * 10.0 ** (gap / 20.0)
        if inplace:
            self /= factor[..., None]
            return self
        else:
            return self / factor[..., None]

    def to_wav(self, filename, depth=16):
        """Save signal as WAV file.

        By default, this function saves a normalized 16-bit version of the signal with at least 6 dB range till clipping occurs.

        Args:
            filename: Name of file to save to.
            depth: If given, convert to integer with specified depth. Else, try to store using the original data type.

        """
        data = self
        dtype = data.dtype if not depth else "int" + str(depth)
        if depth:
            data = (data * 2 ** (depth - 1) - 1).astype(dtype)
        wavfile.write(filename, int(self.fs), data.T)
        # wavfile.write(filename, int(self.fs), self._data/np.abs(self._data).max() *  0.5)
        # wavfile.write(filename, int(self.fs), np.int16(self._data/(np.abs(self._data).max()) * 32767) )

    @classmethod
    def from_wav(cls, filename, normalize=True) -> Signal:
        """Create an instance of `Signal` from a WAV file.

        Args:
            filename: Filename of WAV file.
            normalize: Whether to normalize the signal.

        Returns:
            Signal
        """
        fs, data = wavfile.read(filename)
        data = data.astype(np.float32, copy=False).T
        if normalize:
            data /= np.max(np.abs(data))
        return cls(data, fs=fs)


_PLOTTING_PARAMS = {
    "title": None,
    "xlabel": None,
    "ylabel": None,
    "xscale": "linear",
    "yscale": "linear",
    "xlim": (None, None),
    "ylim": (None, None),
    "labels": None,
    "linestyles": ["-", "-.", "--", ":"],
}


def _get_plotting_params():
    d = dict()
    d.update(_PLOTTING_PARAMS)
    return d


def _base_plot(x, y, given_params) -> Axes:
    """Common function for creating plots.

    Returns:
        Axes object.

    """
    params = _get_plotting_params()
    params.update(given_params)

    linestyles = itertools.cycle(iter(params["linestyles"]))

    # Check if an axes object is passed in. Otherwise, create one.
    ax0 = params.get("ax", plt.figure().add_subplot(111))

    ax0.set_title(params["title"])
    if y.ndim > 1:
        for channel in y:
            ax0.plot(x, channel, linestyle=next(linestyles))
    else:
        ax0.plot(x, y)
    ax0.set_xlabel(params["xlabel"])
    ax0.set_ylabel(params["ylabel"])
    ax0.set_xscale(params["xscale"])
    ax0.set_yscale(params["yscale"])
    ax0.set_xlim(params["xlim"])
    ax0.set_ylim(params["ylim"])

    if params["labels"] is None and y.ndim > 1:
        params["labels"] = np.arange(y.shape[-2]) + 1
    if params["labels"] is not None:
        ax0.legend(labels=params["labels"])

    return ax0


__all__ = ["Signal"]
