import itertools
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
import acoustic_toolbox

from acoustic_toolbox.standards.iso_tr_25417_2007 import REFERENCE_PRESSURE
from acoustic_toolbox.standards.iec_61672_1_2013 import WEIGHTING_SYSTEMS
from acoustic_toolbox.standards.iec_61672_1_2013 import (
    NOMINAL_OCTAVE_CENTER_FREQUENCIES,
    NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
)


class Signal(np.ndarray):
    """A signal consisting of samples (array) and a sample frequency (float)."""

    def __new__(cls, data, fs):
        obj = np.asarray(data).view(cls)
        obj.fs = fs
        return obj

    def __array_prepare__(self, array, context=None):
        try:
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

    def __array_wrap__(self, out_arr, context=None):
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
    def duration(self):
        """Duration of signal in seconds."""
        return float(self.samples / self.fs)

    @property
    def values(self):
        """Return the values of this signal as an instance of :class:`np.ndarray`."""
        return np.array(self)

    def calibrate_to(self, decibel, inplace=False):
        """Calibrate signal to value `decibel`.

        Args:
                  decibel: Value to calibrate to.
                  inplace: Whether to perform inplace or not. (Default value = False)

        Returns:
                  class:`Signal`

        Values of `decibel` are broadcasted. To set a value per channel, use `decibel[...,None]`.: Calibrated signal.

        """
        decibel = decibel * np.ones(self.shape)
        gain = decibel - self.leq()[..., None]
        return self.gain(gain, inplace=inplace)

    def calibrate_with(self, other, decibel, inplace=False):
        """Calibrate signal with other signal.

        Args:
          other: Other signal/array.
          decibel: Signal level of `other`.
          inplace: Whether to perform inplace or not. (Default value = False)

        Returns:
          class:`Signal`: calibrated signal.

        """
        if not isinstance(other, Signal):
            other = Signal(other, self.fs)
        gain = decibel - other.leq()
        return self.gain(gain, inplace=inplace)

    def decimate(self, factor, zero_phase=False, ftype="iir", order=None):
        """Decimate signal by integer `factor`. Before downsampling a low-pass filter is applied.

        Args:
                  factor: Downsampling factor.
                  zero_phase: Prevent phase shift by filtering with ``filtfilt`` instead of ``lfilter``. (Default value = False)
                  ftype: Filter type. (Default value = "iir")
                  order: Filter order. (Default value = None)

        Returns:
                  class:`Signal`

        .. seealso:: :func:`scipy.signal.decimate`
        .. seealso:: :meth:`resample`: Decimated signal.

        """
        return Signal(
            acoustic_toolbox.signal.decimate(
                x=self, q=factor, n=order, ftype=ftype, zero_phase=zero_phase
            ),
            self.fs / factor,
        )

    def resample(self, nsamples, times=None, axis=-1, window=None):
        """Resample signal.

        Args:
          samples: New amount of samples.
          times: Times corresponding to samples. (Default value = None)
          axis: Axis. (Default value = -1)
          window: Window.

        .. seealso:: :func:`scipy.signal.resample`
        .. seealso:: :meth:`decimate`

        You might want to low-pass filter this signal before resampling. (Default value = None)
          nsamples:

        Returns:

        """
        return Signal(
            resample(self, nsamples, times, axis, window),
            nsamples / self.samples * self.fs,
        )

    def upsample(self, factor, axis=-1):
        """Upsample signal with integer factor.

        Args:
          factor: Upsample factor.
          axis: Axis.

        .. seealso:: :meth:`resample` (Default value = -1)

        Returns:

        """
        return self.resample(int(self.samples * factor), axis=axis)

    def gain(self, decibel, inplace=False):
        """Apply gain of `decibel` decibels.

        Args:
          decibel: Decibels
          inplace: In place (Default value = False)

        Returns:
          class:`Signal`: Amplified signal.

        """
        factor = 10.0 ** (decibel / 20.0)
        if inplace:
            self *= factor
            return self
        else:
            return self * factor

    def pick(self, start=0.0, stop=None):
        """Get signal from start time to stop time.

        Args:
          start(float, optional): Start time. (Default value = 0.0)
          stop(float, optional): End time. (Default value = None)

        Returns:
          class:`Signal`: Selected part of the signal.

        """
        if start is not None:
            start = int(np.floor(start * self.fs))
        if stop is not None:
            stop = int(np.floor(stop * self.fs))
        return self[..., start:stop]

    def times(self):
        """Time vector.

        Args:

        Returns:
          class:`np.ndarray`: A vector with a timestamp for each sample.

        """
        return np.arange(0, self.samples) / self.fs

    def energy(self):
        """Signal energy.

        Args:

        Returns:
                  class:`np.ndarray`

        .. math:: E = \\sum_{n=0}^{N-1} |x_n|^2: Total energy per channel.

        """
        return float((self * self).sum())

    def power(self):
        """Signal power.

        .. math:: P = \\frac{1}{N} \\sum_{n=0}^{N-1} |x_n|^2

        Args:

        Returns:

        """
        return self.energy() / len(self)

    def ms(self):
        """Mean value squared of signal.

        .. seealso:: :func:`acoustic_toolbox.signal.ms`

        Args:

        Returns:

        """
        return acoustic_toolbox.signal.ms(self)

    def rms(self):
        """Root mean squared of signal.

        .. seealso:: :func:`acoustic_toolbox.signal.rms`

        Args:

        Returns:

        """
        return acoustic_toolbox.signal.rms(self)
        # return np.sqrt(self.power())

    def weigh(self, weighting="A", zero_phase=False):
        """Apply frequency-weighting. By default 'A'-weighting is applied.

        Args:
                  weighting: Frequency-weighting filter to apply.
                Valid options are 'A', 'C' and 'Z'. Default weighting is 'A'.
                  zero_phase:  (Default value = False)

        Returns:
                  class:`Signal`.

        By default the weighting filter is applied using
        :func:`scipy.signal.lfilter` causing a frequency-dependent delay. In case a
        delay is undesired, the filter can be applied using :func:`scipy.signal.filtfilt`
        by setting `zero_phase=True`.: Weighted signal.

        """
        num, den = WEIGHTING_SYSTEMS[weighting]()
        b, a = bilinear(num, den, self.fs)
        func = filtfilt if zero_phase else lfilter
        return self._construct(func(b, a, self))

    def correlate(self, other=None, mode="full"):
        """Correlate signal with `other` signal. In case `other==None` this
        method returns the autocorrelation.

        Args:
          other: Other signal. (Default value = None)
          mode: Mode.

        .. seealso:: :func:`np.correlate`, :func:`scipy.signal.fftconvolve` (Default value = "full")

        Returns:

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

    def amplitude_envelope(self):
        """Amplitude envelope.

        Args:

        Returns:
                  class:`Signal`

        .. seealso:: :func:`acoustic_toolbox.signal.amplitude_envelope`: Amplitude envelope of signal.

        """
        return self._construct(
            acoustic_toolbox.signal.amplitude_envelope(self, self.fs)
        )

    def instantaneous_frequency(self):
        """Instantaneous frequency.

        Args:

        Returns:
                  class:`Signal`

        .. seealso:: :func:`acoustic_toolbox.signal.instantaneous_frequency`: Instantaneous frequency of signal.

        """
        return self._construct(
            acoustic_toolbox.signal.instantaneous_frequency(self, self.fs)
        )

    def instantaneous_phase(self):
        """Instantaneous phase.

        Args:

        Returns:
                  class:`Signal`

        .. seealso:: :func:`acoustic_toolbox.signal.instantaneous_phase`: Instantaneous phase of signal.

        """
        return self._construct(
            acoustic_toolbox.signal.instantaneous_phase(self, self.fs)
        )

    def detrend(self, **kwargs):
        """Detrend signal.

        Args:
                  **kwargs:

        Returns:
                  class:`Signal`

        .. seealso:: :func:`scipy.signal.detrend`: Detrended version of signal.

        """
        return self._construct(detrend(self, **kwargs))

    def unwrap(self):
        """Unwrap signal in case the signal represents wrapped phase.

        Args:

        Returns:
                  class:`Signal`

        .. seealso:: :func:`np.unwrap`: Unwrapped signal.

        """
        return self._construct(np.unwrap(self))

    def complex_cepstrum(self, N=None):
        """Complex cepstrum.

        Args:
          N: Amount of bins. (Default value = None)

        Returns:
          Quefrency, complex cepstrum and delay in amount of samples.

          .. seealso:: :func:`acoustic_toolbox.cepstrum.complex_cepstrum`

        """
        if N is not None:
            times = np.linspace(0.0, self.duration, N, endpoint=False)
        else:
            times = self.times()
        cepstrum, ndelay = acoustic_toolbox.cepstrum.complex_cepstrum(self, n=N)
        return times, cepstrum, ndelay

    def real_cepstrum(self, N=None):
        """Real cepstrum.

        Args:
          N: Amount of bins. (Default value = None)

        Returns:
          Quefrency and real cepstrum.

          .. seealso:: :func:`acoustic_toolbox.cepstrum.real_cepstrum`

        """
        if N is not None:
            times = np.linspace(0.0, self.duration, N, endpoint=False)
        else:
            times = self.times()
        return times, acoustic_toolbox.cepstrum.real_cepstrum(self, n=N)

    def power_spectrum(self, N=None):
        """Power spectrum.

        Args:
          N: Amount of bins.

        .. seealso:: :func:`acoustic_toolbox.signal.power_spectrum` (Default value = None)

        Returns:

        """
        return acoustic_toolbox.signal.power_spectrum(self, self.fs, N=N)

    def angle_spectrum(self, N=None):
        """Phase angle spectrum. Wrapped.

        Args:
          N: amount of bins.

        .. seealso::

        :func:`acoustic_toolbox.signal.angle_spectrum`, :func:`acoustic_toolbox.signal.phase_spectrum`
        and :meth:`phase_spectrum`. (Default value = None)

        Returns:

        """
        return acoustic_toolbox.signal.angle_spectrum(self, self.fs, N=N)

    def phase_spectrum(self, N=None):
        """Phase spectrum. Unwrapped.

        Args:
          N: Amount of bins.

        .. seealso::

        :func:`acoustic_toolbox.signal.phase_spectrum`, :func:`acoustic_toolbox.signal.angle_spectrum`
        and :meth:`angle_spectrum`. (Default value = None)

        Returns:

        """
        return acoustic_toolbox.signal.phase_spectrum(self, self.fs, N=N)

    def peak(self, axis=-1):
        """Peak sound pressure.

        Args:
          axis: Axis.

        .. seealso::

        :func:`acoustic.standards.iso_tr_25417_2007.peak_sound_pressure` (Default value = -1)

        Returns:

        """
        return acoustic_toolbox.standards.iso_tr_25417_2007.peak_sound_pressure(
            self, axis=axis
        )

    def peak_level(self, axis=-1):
        """Peak sound pressure level.

        Args:
          axis: Axis.

        .. seealso::

        :func:`acoustic_toolbox.standards.iso_tr_25417_2007.peak_sound_pressure_level` (Default value = -1)

        Returns:

        """
        return acoustic_toolbox.standards.iso_tr_25417_2007.peak_sound_pressure_level(
            self, axis=axis
        )

    def min(self, axis=-1):
        """Args:
          axis:  (Default value = -1)

        Returns:
          Refer to `np.amin` for full documentation.

        """
        return np.ndarray.min(self, axis=axis)

    def max(self, axis=-1):
        """Args:
          axis:  (Default value = -1)

        Returns:
          Refer to `np.amax` for full documentation.

        """
        return np.ndarray.max(self, axis=axis)

    def max_level(self, axis=-1):
        """Maximum sound pressure level.

        Args:
          axis: Axis.

        .. seealso:: :func:`acoustic_toolbox.standards.iso_tr_25417_2007.max_sound_pressure_level` (Default value = -1)

        Returns:

        """
        return acoustic_toolbox.standards.iso_tr_25417_2007.max_sound_pressure_level(
            self, axis=axis
        )

    def sound_exposure(self, axis=-1):
        """Sound exposure.

        Args:
          axis: Axis.

        .. seealso:: :func:`acoustic_toolbox.standards.iso_tr_25417_2007.sound_exposure` (Default value = -1)

        Returns:

        """
        return acoustic_toolbox.standards.iso_tr_25417_2007.sound_exposure(
            self, self.fs, axis=axis
        )

    def sound_exposure_level(self, axis=-1):
        """Sound exposure level.

        Args:
          axis: Axis.

        .. seealso:: :func:`acoustic_toolbox.standards.iso_tr_25417_2007.sound_exposure_level` (Default value = -1)

        Returns:

        """
        return acoustic_toolbox.standards.iso_tr_25417_2007.sound_exposure_level(
            self, self.fs, axis=axis
        )

    def plot_complex_cepstrum(self, N=None, **kwargs):
        """Plot complex cepstrum of signal.

        Valid kwargs:

        * xscale
        * yscale
        * xlim
        * ylim
        * frequency: Boolean indicating whether the x-axis should show time in seconds or quefrency
        * xlabel_frequency: Label in case frequency is shown.

        Args:
          N:  (Default value = None)
          **kwargs:

        Returns:

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

    def plot_real_cepstrum(self, N=None, **kwargs):
        """Plot real cepstrum of signal.

        Valid kwargs:

        * xscale
        * yscale
        * xlim
        * ylim
        * frequency: Boolean indicating whether the x-axis should show time in seconds or quefrency
        * xlabel_frequency: Label in case frequency is shown.

        Args:
          N:  (Default value = None)
          **kwargs:

        Returns:

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

        Valid kwargs:

        * xscale
        * yscale
        * xlim
        * ylim
        * reference: Reference power

        .. seealso:: :meth:`power_spectrum`

        Args:
          N:  (Default value = None)
          **kwargs):  # filename:  (Default value = None)
          scale:  (Default value = 'log')

        Returns:

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

    def plot_angle_spectrum(self, N=None, **kwargs):
        """Plot phase angle spectrum of signal. Wrapped.

        Valid kwargs:

        * xscale
        * yscale
        * xlim
        * ylim
        * reference: Reference power

        Args:
          N:  (Default value = None)
          **kwargs:

        Returns:

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

    def plot_phase_spectrum(self, N=None, **kwargs):
        """Plot phase spectrum of signal. Unwrapped.

        Valid kwargs:

        * xscale
        * yscale
        * xlim
        * ylim
        * reference: Reference power

        Args:
          N:  (Default value = None)
          **kwargs:

        Returns:

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

        Args:
          **kwargs:

        Returns:
          Spectrogram.

          See :func:`scipy.signal.spectrogram`. Some of the default values have been changed.
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

        Valid kwargs:

        * xlim
        * ylim
        * clim
        .. note:: This method only works for a single channel.

        Args:
          **kwargs:

        Returns:

        """
        # To do, use :meth:`spectrogram`.
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

    def levels(self, time=0.125, method="average"):
        """Calculate sound pressure level as function of time.

        Args:
          time: Averaging time or integration time constant. Default value is 0.125 corresponding to FAST.
          method: Use time `average` or time `weighting`. Default option is `average`.

        Returns:
          sound pressure level as function of time.

          .. seealso:: :func:`acoustic_toolbox.standards.iec_61672_1_2013.time_averaged_sound_level`
          .. seealso:: :func:`acoustic_toolbox.standards.iec_61672_1_2013.time_weighted_sound_level`

        """
        if method == "average":
            return (
                acoustic_toolbox.standards.iec_61672_1_2013.time_averaged_sound_level(
                    self.values, self.fs, time
                )
            )
        elif method == "weighting":
            return (
                acoustic_toolbox.standards.iec_61672_1_2013.time_weighted_sound_level(
                    self.values, self.fs, time
                )
            )
        else:
            raise ValueError("Invalid method")

    def leq(self):
        """Equivalent level. Single-value number.

        .. seealso:: :func:`acoustic_toolbox.standards.iso_tr_25417_2007.equivalent_sound_pressure_level`

        Args:

        Returns:

        """
        return acoustic_toolbox.standards.iso_tr_25417_2007.equivalent_sound_pressure_level(
            self.values
        )

    def plot_levels(self, **kwargs):
        """Plot sound pressure level as function of time.

        .. seealso:: :meth:`levels`

        Args:
          **kwargs:

        Returns:

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

    def bandpass(self, lowcut, highcut, order=8, zero_phase=False):
        """Filter signal with band-pass filter.

        Args:
                  lowcut: Lower cornerfrequency.
                  highcut: Upper cornerfrequency.
                  order: Filter order. (Default value = 8)
                  zero_phase: Prevent phase error by filtering in both directions (filtfilt). (Default value = False)

        Returns:
                  class:`Signal`.

        .. seealso:: :func:`acoustic_toolbox.signal.bandpass`: Band-pass filtered signal.

        """
        return type(self)(
            acoustic_toolbox.signal.bandpass(
                self, lowcut, highcut, self.fs, order=order, zero_phase=zero_phase
            ),
            self.fs,
        )

    def bandstop(self, lowcut, highcut, order=8, zero_phase=False):
        """Filter signal with band-stop filter.

        Args:
                  lowcut: Lower cornerfrequency.
                  highcut: Upper cornerfrequency.
                  order: Filter order. (Default value = 8)
                  zero_phase: Prevent phase error by filtering in both directions (filtfilt). (Default value = False)

        Returns:
                  class:`Signal`.

        .. seealso:: :func:`acoustic_toolbox.signal.bandstop`: Band-pass filtered signal.

        """
        return type(self)(
            acoustic_toolbox.signal.bandstop(
                self, lowcut, highcut, self.fs, order=order, zero_phase=zero_phase
            ),
            self.fs,
        )

    def highpass(self, cutoff, order=4, zero_phase=False):
        """Filter signal with high-pass filter.

        Args:
                  cutoff: Cornerfrequency.
                  order: Filter order. (Default value = 4)
                  zero_phase: Prevent phase error by filtering in both directions (filtfilt). (Default value = False)

        Returns:
                  class:`Signal`.

        .. seealso:: :func:`acoustic_toolbox.signal.highpass`: High-pass filtered signal.

        """
        return type(self)(
            acoustic_toolbox.signal.highpass(
                self, cutoff, self.fs, order=order, zero_phase=zero_phase
            ),
            self.fs,
        )

    def lowpass(self, cutoff, order=4, zero_phase=False):
        """Filter signal with low-pass filter.

        Args:
                  cutoff: Cornerfrequency.
                  order: Filter order. (Default value = 4)
                  zero_phase: Prevent phase error by filtering in both directions (filtfilt). (Default value = False)

        Returns:
                  class:`Signal`.

        .. seealso:: :func:`acoustic_toolbox.signal.lowpass`: Low-pass filtered signal.

        """
        return type(self)(
            acoustic_toolbox.signal.lowpass(
                self, cutoff, self.fs, order=order, zero_phase=zero_phase
            ),
            self.fs,
        )

    def octavepass(self, center, fraction, order=8, zero_phase=False):
        """Filter signal with fractional-octave band-pass filter.

        Args:
                  center: Center frequency. Any value in the band will suffice.
                  fraction: Band designator.
                  order: Filter order. (Default value = 8)
                  zero_phase: Prevent phase error by filtering in both directions (filtfilt). (Default value = False)

        Returns:
                  class:`Signal`.

        .. seealso:: :func:`acoustic_toolbox.signal.octavepass`: Band-pass filtered signal.

        """
        return type(self)(
            acoustic_toolbox.signal.octavepass(
                self,
                center,
                self.fs,
                fraction=fraction,
                order=order,
                zero_phase=zero_phase,
            ),
            self.fs,
        )

    def bandpass_frequencies(self, frequencies, order=8, purge=True, zero_phase=False):
        """Apply bandpass filters for frequencies.

        Args:
          frequencies(Instance of :class:`acoustic_toolbox.signal.Frequencies`): Band-pass filter frequencies.
          order: Filter order. (Default value = 8)
          purge: Discard bands of which the upper corner frequency is above the Nyquist frequency. (Default value = True)
          zero_phase: Prevent phase error by filtering in both directions (filtfilt). (Default value = False)

        Returns:
          Frequencies and band-pass filtered signal.

          .. seealso:: :func:`acoustic_toolbox.signal.bandpass_frequencies`

        """
        frequencies, filtered = acoustic_toolbox.signal.bandpass_frequencies(
            self, self.fs, frequencies, order, purge, zero_phase=zero_phase
        )
        return frequencies, type(self)(filtered, self.fs)

    def octaves(
        self,
        frequencies=NOMINAL_OCTAVE_CENTER_FREQUENCIES,
        order=8,
        purge=True,
        zero_phase=False,
    ):
        """Apply 1/1-octaves bandpass filters.

        Args:
          frequencies(class:`np.ndarray` with (approximate) center-frequencies or an instance of :class:`acoustic_toolbox.signal.Frequencies`, optional): Band-pass filter frequencies. (Default value = NOMINAL_OCTAVE_CENTER_FREQUENCIES)
          order: Filter order. (Default value = 8)
          purge: Discard bands of which the upper corner frequency is above the Nyquist frequency. (Default value = True)
          zero_phase: Prevent phase error by filtering in both directions (filtfilt). (Default value = False)

        Returns:
          Frequencies and band-pass filtered signal.

          .. seealso:: :func:`acoustic_toolbox.signal.bandpass_octaves`

        """
        frequencies, octaves = acoustic_toolbox.signal.bandpass_octaves(
            self, self.fs, frequencies, order, purge, zero_phase=zero_phase
        )
        return frequencies, type(self)(octaves, self.fs)

    def third_octaves(
        self,
        frequencies=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
        order=8,
        purge=True,
        zero_phase=False,
    ):
        """Apply 1/3-octaves bandpass filters.

        Args:
          frequencies(class:`np.ndarray` with (approximate) center-frequencies or an instance of :class:`acoustic_toolbox.signal.Frequencies`, optional): Band-pass filter frequencies. (Default value = NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES)
          order: Filter order. (Default value = 8)
          purge: Discard bands of which the upper corner frequency is above the Nyquist frequency. (Default value = True)
          zero_phase: Prevent phase error by filtering in both directions (filtfilt). (Default value = False)

        Returns:
          Frequencies and band-pass filtered signal.

          .. seealso:: :func:`acoustic_toolbox.signal.bandpass_third_octaves`

        """
        frequencies, octaves = acoustic_toolbox.signal.bandpass_third_octaves(
            self, self.fs, frequencies, order, purge, zero_phase=zero_phase
        )
        return frequencies, type(self)(octaves, self.fs)

    def fractional_octaves(
        self, frequencies=None, fraction=1, order=8, purge=True, zero_phase=False
    ):
        """Apply 1/N-octaves bandpass filters.

        Args:
          frequencies(Instance of :class:`acoustic_toolbox.signal.Frequencies`, optional): Band-pass filter frequencies. (Default value = None)
          fraction: Default band-designator of fractional-octaves.
          order: Filter order. (Default value = 8)
          purge: Discard bands of which the upper corner frequency is above the Nyquist frequency. (Default value = True)
          zero_phase: Prevent phase error by filtering in both directions (filtfilt). (Default value = False)

        Returns:
          Frequencies and band-pass filtered signal.

          .. seealso:: :func:`acoustic_toolbox.signal.bandpass_fractional_octaves`

        """
        if frequencies is None:
            frequencies = acoustic_toolbox.signal.OctaveBand(
                fstart=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES[0],
                fstop=self.fs / 2.0,
                fraction=fraction,
            )
        frequencies, octaves = acoustic_toolbox.signal.bandpass_fractional_octaves(
            self, self.fs, frequencies, fraction, order, purge, zero_phase=zero_phase
        )
        return frequencies, type(self)(octaves, self.fs)

    def plot_octaves(self, **kwargs):
        """Plot octaves.

        .. seealso:: :meth:`octaves`

        Args:
          **kwargs:

        Returns:

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

        .. seealso:: :meth:`third_octaves`

        Args:
          **kwargs:

        Returns:

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
        """Plot fractional octaves.

        Args:
          frequencies:  (Default value = None)
          fraction:  (Default value = 1)
          order:  (Default value = 8)
          purge:  (Default value = True)
          zero_phase:  (Default value = False)
          **kwargs:

        Returns:

        """
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

        Args:
          filename: Name of file.
          start(Start time in seconds from start of signal.): First sample index.
          stop(Stop time in seconds. from stop of signal.): Last sample index.
          **kwargs:

        Returns:

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

        Args:
          gap: Gap between maximum value and ceiling in decibel. (Default value = 6.0)
          inplace: Normalize signal in place.

        The parameter `gap` can be understood as using `gap` decibels fewer for the dynamic range.
        By default a 6 decibel gap is used.

        Returns:

        """
        factor = np.abs(self).max() * 10.0 ** (gap / 20.0)
        if inplace:
            self /= factor[..., None]
            return self
        else:
            return self / factor[..., None]

    def to_wav(self, filename, depth=16):
        """Save signal as WAV file.

        Args:
          filename: Name of file to save to.
          depth: If given, convert to integer with specified depth. Else, try to store using the original data type.

        By default, this function saves a normalized 16-bit version of the signal with at least 6 dB range till clipping occurs.

        Returns:

        """
        data = self
        dtype = data.dtype if not depth else "int" + str(depth)
        if depth:
            data = (data * 2 ** (depth - 1) - 1).astype(dtype)
        wavfile.write(filename, int(self.fs), data.T)
        # wavfile.write(filename, int(self.fs), self._data/np.abs(self._data).max() *  0.5)
        # wavfile.write(filename, int(self.fs), np.int16(self._data/(np.abs(self._data).max()) * 32767) )

    @classmethod
    def from_wav(cls, filename, normalize=True):
        """Create an instance of `Signal` from a WAV file.

        Args:
          filename: Filename
          normalize: Whether to normalize the signal. (Default value = True)

        Returns:

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
    """ """
    d = dict()
    d.update(_PLOTTING_PARAMS)
    return d


def _base_plot(x, y, given_params):
    """Common function for creating plots.

    Args:
      x:
      y:
      given_params:

    Returns:
      class:`matplotlib.Axes`: Axes object.

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
