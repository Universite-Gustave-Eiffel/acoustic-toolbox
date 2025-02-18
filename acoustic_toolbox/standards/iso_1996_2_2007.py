"""ISO 1996-2:2007

ISO 1996-2:2007 describes how sound pressure levels can be determined by direct measurement,
by extrapolation of measurement results by means of calculation, or exclusively by calculation,
intended as a basis for assessing environmental noise.

Reference:
    ISO 1996-2:2007: Description, measurement and assessment of environmental noise
"""

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import linregress
import matplotlib.pyplot as plt
from acoustic_toolbox.decibel import dbsum
from acoustic_toolbox.standards.iso_tr_25417_2007 import REFERENCE_PRESSURE
import weakref
from tabulate import tabulate

TONE_WITHIN_PAUSE_CRITERION_DB = 6.0
"""A tone may exist when the level of any line in the noise pause is 6 dB or more about...."""

TONE_BANDWIDTH_CRITERION_DB = 3.0
"""Bandwidth of the detected peak."""

TONE_LINES_CRITERION_DB = 6.0
"""All lines with levels within 6 dB of the maximum level are classified as tones."""

TONE_SEEK_CRITERION = 1.0
"""Tone seek criterion."""

REGRESSION_RANGE_FACTOR = 0.75
"""Range of regression is usually +/- 0.75 critical bandwidth."""

_WINDOW_CORRECTION = {
    "hann": -1.8,
}
"""Window correction factors for different window types."""


def window_correction(window):
    """Get correction factor to be applied to $L_{pt}$ due to window type.

    Args:
        window: Window type (e.g., 'hann').

    Returns:
        float: Correction factor in dB.

    Raises:
        ValueError: If window correction is not available for specified window.

    See Also:
        Tonality: Class that uses window correction in tonal analysis
    """
    try:
        return _WINDOW_CORRECTION[window]
    except KeyError:
        raise ValueError("Window correction is not available for specified window.")


def critical_band(frequency):
    """Bandwidth of critical band of frequency.

    Args:
        frequency: Center frequency of tone in Hz.

    Returns:
        tuple: A tuple containing:
            - float: Center frequency (minimum 50 Hz)
            - float: Lower band-edge frequency
            - float: Upper band-edge frequency
            - float: Bandwidth (100 Hz below 500 Hz, 20% of center frequency above)

    See Also:
        tonal_audibility: Function that uses critical band parameters
    """
    if isinstance(frequency, np.ndarray):
        center = frequency.copy()
        center[frequency < 50.0] = 50.0
    else:
        center = 50.0 if frequency < 50 else frequency

    bandwidth = (center > 500.0) * (center * 0.20) + (center <= 500.0) * 100.0

    upper = center + bandwidth / 2.0
    lower = center - bandwidth / 2.0

    return center, lower, upper, bandwidth


def tones_level(tone_levels):
    r"""Total sound pressure level of the tones in a critical band given the level of each of the tones.

    Returns:
        float: Total sound pressure level Lpt calculated as:
            $$
            L_{pt} = 10 \log_{10}{\sum 10^{L_{pti}/10}}
            $$

    Note:
        Implementation of equation C.1 from section C.2.3.1 of the standard.

    See Also:
        masking_noise_level: Function to calculate masking noise level
    """
    return dbsum(tone_levels)


def masking_noise_level(
    noise_lines, frequency_resolution, effective_analysis_bandwidth
):
    r"""Masking noise level $L_{pn}$.

    Args:
        noise_lines: Array of masking noise lines Ln in dB.
        frequency_resolution: Frequency resolution Δf in Hz.
        effective_analysis_bandwidth: Effective analysis bandwidth B in Hz.

    Returns:
        float: Masking noise level calculated as:
            $$
            L_{pn} = 10 \log_{10}{\sum 10^{L_n/10}} + 10 \log_{10}{\frac{\Delta f}{B}}
            $$

    Note:
        Implementation of equation C.11 from section C.4.4 of the standard.

    See Also:
        tones_level: Function to calculate total sound pressure level of tones
    """
    return dbsum(noise_lines) + 10.0 * np.log10(
        frequency_resolution / effective_analysis_bandwidth
    )


def masking_noise_lines(
    levels: pd.Series,
    line_classifier,
    center: float,
    bandwidth: float,
    regression_range_factor,
) -> tuple[np.ndarray, float, float]:
    """Determine masking noise level lines using regression line. Returns array of $L_n$.

    Args:
        levels: Levels as function of frequency
        line_classifier: Categorical indicating line types.
        center: Center frequency in Hz.
        bandwidth: Critical band bandwidth in Hz.
        regression_range_factor: Range factor for regression analysis.

    Returns:
        (ndarray): Array of masking noise lines Ln
        (float): Regression slope
        (float): Regression intercept
    """
    slicer = slice(
        center - bandwidth * regression_range_factor,
        center + bandwidth * regression_range_factor,
    )
    levels = levels[slicer]
    frequencies = levels.index
    regression_levels = levels[line_classifier == "noise"]
    slope, intercept = linregress(x=regression_levels.index, y=regression_levels)[0:2]
    levels_from_regression = slope * frequencies + intercept
    return levels_from_regression, slope, intercept


def tonal_audibility(tones_level, masking_noise_level, center) -> float:
    r"""Tonal audibility.

    Args:
        tones_level: Total sound pressure level of tones in critical band Lpt.
        masking_noise_level: Total sound pressure level of masking noise Lpn.
        center: Center frequency of critical band fc.

    Returns:
        Tonal audibility ΔLta calculated as:
            $$
            \Delta L_{ta} = L_{pt} - L_{pn} + 2 + \log_{10}{1 + \left(\frac{f_c}{502}\right)^{2.5}}
            $$


    Note:
        Implementation of equation C.3 from section C.2.4 of the standard.

    See Also:
        critical_band: Function to calculate critical band parameters
        tonal_adjustment: Function to calculate tonal adjustment
    """
    return (
        tones_level
        - masking_noise_level
        + 2.0
        + np.log10(1.0 + (center / 502.0) ** (2.5))
    )


def tonal_adjustment(tonal_audibility):
    """Calculate tonal adjustment Kt.

    Args:
        tonal_audibility: Tonal audibility $L_{ta}$ in dB.

    Returns:
        float: Adjustment Kt in dB:
            - 6.0 dB if ΔLta > 10 dB
            - 0.0 dB if ΔLta < 4 dB
            - ΔLta - 4 dB otherwise

    Note:
        Implementation of equations C.4-C.6 from section C.2.4 of the standard.

    See Also:
        tonal_audibility: Function to calculate tonal audibility
    """
    if tonal_audibility > 10.0:
        return 6.0
    elif tonal_audibility < 4.0:
        return 0.0
    else:
        return tonal_audibility - 4.0


class Tonality:
    """Perform assessment of audibility of tones in noise.

    Objective method for assessing the audibility of tones in noise.

    Args:
        signal: Time-domain signal samples.
        sample_frequency: Sample frequency in Hz.
        window: Window type for spectral analysis. Defaults to 'hann'.
        reference_pressure: Reference pressure. Defaults to REFERENCE_PRESSURE.
        tsc: Tone seeking criterion in dB. Defaults to TONE_SEEK_CRITERION.
        regression_range_factor: Regression range factor. Defaults to REGRESSION_RANGE_FACTOR.
        nbins: Number of frequency bins for FFT. Defaults to sample_frequency.
        force_tone_without_pause: Force tone detection without noise pause. Defaults to False.
        force_bandwidth_criterion: Force bandwidth criterion. Defaults to False.
    """

    def __init__(  # pylint: disable=too-many-instance-attributes
        self,
        signal,
        sample_frequency,
        window="hann",
        reference_pressure=REFERENCE_PRESSURE,
        tsc=TONE_SEEK_CRITERION,
        regression_range_factor=REGRESSION_RANGE_FACTOR,
        nbins=None,
        force_tone_without_pause=False,
        force_bandwidth_criterion=False,
    ):
        self.signal = signal
        """Samples in time-domain."""
        self.sample_frequency = sample_frequency
        """Sample frequency."""
        self.window = window
        """Window to be used."""
        self.reference_pressure = reference_pressure
        """Reference sound pressure."""
        self.tsc = tsc
        """Tone seeking criterium."""
        self.regression_range_factor = regression_range_factor
        """Regression range factor."""
        self.nbins = nbins
        """Amount of frequency nbins to use. See attribute `nperseg` of :func:`scipy.signal.welch`."""

        self._noise_pauses = list()
        """Private list of noise pauses that were determined or assigned."""
        self._spectrum = None
        """Power spectrum as function of frequency."""

        self.force_tone_without_pause = force_tone_without_pause
        self.force_bandwidth_criterion = force_bandwidth_criterion

    @property
    def noise_pauses(self):
        """Get determined noise pauses.

        Yields:
            NoisePause: Each noise pause found in the signal.
        """
        for noise_pause in self._noise_pauses:
            yield noise_pause

    @property
    def tones(self):
        """Get determined tones.

        Yields:
            Tone: Each tone found in noise pauses.
        """
        for noise_pause in self.noise_pauses:
            if noise_pause.tone is not None:
                yield noise_pause.tone

    @property
    def critical_bands(self):
        """Get critical bands.

        A critical band is determined for each detected tone.

        Yields:
            CriticalBand: Each critical band around detected tones.
        """
        for tone in self.tones:
            yield tone.critical_band

    @property
    def spectrum(self):
        """Get power spectrum of the input signal.

        Returns:
            pandas.Series: Power spectrum in dB re reference_pressure.
        """
        if self._spectrum is None:
            nbins = self.nbins
            if nbins is None:
                nbins = self.sample_frequency
            nbins //= 1  # Fix because of bug in welch with uneven nbins
            f, p = welch(
                self.signal,
                fs=self.sample_frequency,
                nperseg=nbins,
                window=self.window,
                detrend=False,
                scaling="spectrum",
            )
            self._spectrum = pd.Series(
                10.0 * np.log10(p / self.reference_pressure**2.0), index=f
            )
        return self._spectrum

    @property
    def frequency_resolution(self):
        """Frequency resolution.

        Returns:
            float: Frequency resolution Δf in Hz.
        """
        df = np.diff(np.array(self.spectrum.index)).mean()
        return df

    @property
    def effective_analysis_bandwidth(self):
        r"""Effective analysis bandwidth.

        In the case of the Hanning window
        $$
        B_{eff} = 1.5 \Delta f
        $$

        with $\Delta f$ the `frequency_resolution`.

        Returns:
            float: Effective analysis bandwidth in Hz.

        Raises:
            ValueError: If window type is not supported.

        Note:
            See section C.2.2 Note 1 of the standard.
        """
        if self.window == "hann":
            return 1.5 * self.frequency_resolution
        else:
            raise ValueError()

    def _set_noise_pauses(self, noise_pauses):
        """Manually set noise pauses. Expects iterable of tuples.

        Args:
            noise_pauses: Iterable of (start, end) tuples.

        Returns:
            self: For method chaining.
        """
        self._noise_pauses = [NoisePause(start, end) for start, end in noise_pauses]
        return self

    def determine_noise_pauses(self, end=None):
        """Find noise pauses in the spectrum.

        Uses noise_pause_seeker to find potential noise pauses.

        Args:
            end: Optional end index to limit search range.

        Returns:
            self: For method chaining.
        """
        self._set_noise_pauses(
            noise_pause_seeker(np.array(self.spectrum[:end]), self.tsc)
        )
        return self

    def _construct_line_classifier(self):
        """Set values of line classifier.

        Returns:
            self: For method chaining.
        """
        levels = self.spectrum

        categories = ["noise", "start", "end", "neither", "tone"]
        self.line_classifier = pd.Series(
            pd.Categorical(["noise"] * len(levels), categories=categories),
            index=levels.index,
        )

        # Add noise pauses
        for noise_pause in self.noise_pauses:
            # Mark noise pause start and end.
            self.line_classifier.iloc[noise_pause.start] = "start"
            self.line_classifier.iloc[noise_pause.end] = "end"
            # Mark all other lines within noise pause as neither tone nor noise.
            self.line_classifier.iloc[noise_pause.start + 1 : noise_pause.end] = (
                "neither"  # Half-open interval
            )

        # Add tone lines
        for tone in self.tones:
            self.line_classifier.iloc[tone._tone_lines] = "tone"

        return self

    def _determine_tones(self):
        """Analyze noise pauses for tones.

        Examines each noise pause for potential tones using determine_tone_lines.
        Creates Tone objects for any detected tones.

        Returns:
            self: For method chaining.
        """
        levels = self.spectrum

        # First we need to check for the tones.
        for noise_pause in self.noise_pauses:
            # Determine the indices of the tones in a noise pause
            tone_indices, bandwidth_for_tone_criterion = determine_tone_lines(
                levels,
                self.frequency_resolution,
                noise_pause.start,
                noise_pause.end,
                self.force_tone_without_pause,
                self.force_bandwidth_criterion,
            )
            # If we have indices, ...
            if np.any(tone_indices):
                # ...then we create a tone object.
                noise_pause.tone = create_tone(
                    levels,
                    tone_indices,
                    bandwidth_for_tone_criterion,
                    weakref.proxy(noise_pause),
                )
        return self

    def _determine_critical_bands(self):
        """Put a critical band around each of the determined tones.

        Returns:
            self: For method chaining.
        """
        for tone in self.tones:
            critical_band = self.critical_band_at(tone.center)
            tone.critical_band = critical_band
            critical_band.tone = weakref.proxy(tone)
        return self

    def analyse(self):
        """Analyse the noise pauses for tones and put critical bands around each of these tones.

        The tones are available via `tones` and the critical bands via `critical_bands`.
        Per frequency line results are available via `line_classifier`.

        Returns:
            self: For method chaining.
        """
        # Determine tones. Puts noise pause starts/ends in classier as well as tone lines
        # and lines that are neither tone nor noise.
        self._determine_tones()
        # Construct line classifier
        self._construct_line_classifier()
        # Determine critical bands.
        self._determine_critical_bands()
        return self

    def critical_band_at(self, frequency):
        """Put at a critical band at `frequency`.

        In order to use this function `line_classifier` needs to be available,
        which means `analyse` needs to be used first.

        Args:
            frequency: Center frequency in Hz.

        Returns:
            CriticalBand: Critical band object.

        Note:
            Requires line_classifier to be available (call analyse first).
        """
        return create_critical_band(
            self.spectrum,
            self.line_classifier,
            frequency,
            self.frequency_resolution,
            self.effective_analysis_bandwidth,
            self.regression_range_factor,
            self.window,
        )

    def plot_spectrum(self):
        """Plot power spectrum.

        Returns:
            matplotlib.figure.Figure: Figure with spectrum plot.
        """
        spectrum = self.spectrum
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(spectrum.index, spectrum)
        ax.set_xlabel("f in Hz")
        ax.set_ylabel("L in dB")
        return fig

    @property
    def dominant_tone(self):
        """Get most dominant tone.

        The most dominant tone is the one with highest tonal audibility $L_{ta}$.

        Returns:
            Tone: Most dominant tone, or None if no tones found.
        """
        try:
            return sorted(
                self.tones, key=lambda x: x.critical_band.tonal_audibility, reverse=True
            )[0]
        except IndexError:
            return None

    def plot_results(self, noise_pauses=False, tones=True, critical_bands=True):
        """Plot analysis results.

        Args:
            noise_pauses: Whether to show noise pauses. Defaults to False.
            tones: Whether to show tones. Defaults to True.
            critical_bands: Whether to show critical bands. Defaults to True.

        Returns:
            matplotlib.figure.Figure: Figure with results plot.
        """
        df = self.frequency_resolution
        levels = self.spectrum

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(levels.index, levels)
        ax.set_xlabel("$f$ in Hz")
        ax.set_ylabel("$L$ in dB")

        if noise_pauses:
            for pause in self.noise_pauses:
                ax.axvspan(pause.start * df, pause.end * df, color="green", alpha=0.05)

        if tones:
            for tone in self.tones:
                ax.axvline(tone.center, color="black", alpha=0.05)

        if critical_bands:
            for band in self.critical_bands:
                ax.axvspan(band.start, band.end, color="yellow", alpha=0.05)

        band = self.dominant_tone.critical_band
        ax.axvline(band.start, color="red", linewidth=0.1)
        ax.axvline(band.end, color="red", linewidth=0.1)

        # Limit xrange
        if noise_pauses:
            _items = list(self.noise_pauses)
        elif critical_bands:
            _items = list(self.critical_bands)
        ax.set_xlim(
            min(item.start for item in _items), max(item.end for item in _items)
        )
        return fig

    def overview(self):
        """Print overview of analysis results.

        Returns:
            str: Tabulated overview of results.

        Raises:
            ValueError: If no tones have been determined yet.
        """
        try:
            cb = self.dominant_tone.critical_band
        except AttributeError:
            raise ValueError(
                "Cannot show overview (yet). No tones have been determined."
            )

        tones = [
            ("Tone", "{:4.1f} Hz: {:4.1f} dB".format(tone.center, tone.tone_level))
            for tone in self.tones
        ]

        table = [
            ("Critical band", "{:4.1f} to {:4.1f} Hz".format(cb.start, cb.end)),
            (
                "Masking noise level $L_{pn}$",
                "{:4.1f} dB".format(cb.masking_noise_level),
            ),
            ("Tonal level $L_{pt}$", "{:4.1f} dB".format(cb.total_tone_level)),
            ("Dominant tone", "{:4.1f} Hz".format(cb.tone.center)),
            (
                "3 dB bandwidth of tone",
                "{:2.1f}% of {:4.1f}".format(
                    cb.tone.bandwidth_3db / cb.bandwidth * 100.0, cb.bandwidth
                ),
            ),
            ("Tonal audibility $L_{ta}$", "{:4.1f} dB".format(cb.tonal_audibility)),
            ("Adjustment $K_{t}$", "{:4.1f} dB".format(cb.adjustment)),
            ("Frequency resolution", "{:4.1f} Hz".format(self.frequency_resolution)),
            (
                "Effective analysis bandwidth",
                "{:4.1f} Hz".format(self.effective_analysis_bandwidth),
            ),
        ]
        table += tones
        return tabulate(table)

    def results_as_dataframe(self):
        """Get analysis results as pandas DataFrame.

        Returns:
            pandas.DataFrame: DataFrame containing results for each tone.
        """
        data = (
            (
                tone.center,
                tone.tone_level,
                tone.bandwidth_3db,
                tone.critical_band.start,
                tone.critical_band.end,
                tone.critical_band.bandwidth,
                tone.critical_band.regression_slope,
                tone.critical_band.regression_intercept,
                tone.critical_band.masking_noise_level,
                tone.critical_band.total_tone_level,
                tone.critical_band.tonal_audibility,
                tone.critical_band.adjustment,
            )
            for tone in self.tones
        )
        columns = [
            "center",
            "tone_level",
            "bandwidth_3db",
            "critical_band_start",
            "critical_band_end",
            "critical_band_bandwidth",
            "regression_slope",
            "regression_intercept",
            "masking_noise_level",
            "total_tone_level",
            "tonal_audibility",
            "adjustment",
        ]
        return pd.DataFrame(list(data), columns=columns)


class NoisePause:
    """Container for noise pause information.

    A noise pause is a section of the spectrum that may contain a tone.

    Args:
        start: Start index of noise pause.
        end: End index of noise pause.
        tone: Optional Tone object if a tone is found in this pause.
    """

    def __init__(self, start, end, tone=None):
        self.start = start
        self.end = end
        self.tone = tone

    def __str__(self):
        return "(start={},end={})".format(self.start, self.end)

    def __repr__(self):
        return "NoisePause{}".format(str(self))

    def __iter__(self):
        yield self.start
        yield self.stop

    def _repr_html_(self):
        """HTML representation for Jupyter notebooks."""
        table = [("Start", self.start), ("End", self.end)]
        return tabulate(table, tablefmt="html")


def create_tone(levels, tone_lines, bandwidth_for_tone_criterion, noise_pause):
    """Create an instance of Tone."""
    center = levels.iloc[tone_lines].idxmax()
    tone_level = tones_level(levels.iloc[tone_lines])
    return Tone(
        center, tone_lines, tone_level, noise_pause, bandwidth_for_tone_criterion
    )


class Tone:
    """Container for tone information.

    Args:
        center: Center frequency in Hz.
        tone_lines: Indices of spectral lines belonging to tone.
        tone_level: Total level of tone in dB.
        noise_pause: Parent NoisePause object.
        bandwidth_3db: -3 dB bandwidth in Hz.
        critical_band: Optional CriticalBand object.
    """

    def __init__(
        self,
        center,
        tone_lines,
        tone_level,
        noise_pause,
        bandwidth_3db,
        critical_band=None,
    ):
        self.center = center
        self._tone_lines = tone_lines
        self.tone_level = tone_level
        self.noise_pause = noise_pause
        self.bandwidth_3db = bandwidth_3db
        self.critical_band = critical_band

    def __str__(self):
        return "(center={:4.1f}, levels={:4.1f})".format(self.center, self.tone_level)

    def __repr__(self):
        return "Tone{}".format(str(self))

    def _repr_html_(self):
        """HTML representation for Jupyter notebooks."""
        table = [
            ("Center frequency", "{:4.1f} Hz".format(self.center)),
            ("Tone level", "{:4.1f} dB".format(self.tone_level)),
        ]
        return tabulate(table, tablefmt="html")


def create_critical_band(
    levels,
    line_classifier,
    frequency,
    frequency_resolution,
    effective_analysis_bandwidth,
    regression_range_factor,
    window,
    tone=None,
):
    """Create an instance of CriticalBand."""
    center, start, end, bandwidth = critical_band(frequency)

    # Masking noise lines
    noise_lines, regression_slope, regression_intercept = masking_noise_lines(
        levels, line_classifier, center, bandwidth, regression_range_factor
    )
    # Masking noise level
    noise_level = masking_noise_level(
        noise_lines, frequency_resolution, effective_analysis_bandwidth
    )
    # Total tone level
    tone_lines = levels[line_classifier == "tone"][start:end]
    tone_level = tones_level(tone_lines) - window_correction(window)
    # Tonal audibility
    audibility = tonal_audibility(tone_level, noise_level, center)
    # Adjustment Kt
    adjustment = tonal_adjustment(audibility)

    return CriticalBand(
        center,
        start,
        end,
        bandwidth,
        regression_range_factor,
        regression_slope,
        regression_intercept,
        noise_level,
        tone_level,
        audibility,
        adjustment,
        tone,
    )


class CriticalBand:
    """Container for critical band information.

    A critical band is a frequency band around a tone used for masking analysis.

    Args:
        center: Center frequency in Hz.
        start: Lower band-edge frequency in Hz.
        end: Upper band-edge frequency in Hz.
        bandwidth: Bandwidth in Hz.
        regression_range_factor: Factor for regression range.
        regression_slope: Slope from linear regression.
        regression_intercept: Intercept from linear regression.
        noise_level: Masking noise level in dB.
        tone_level: Total tone level in dB.
        audibility: Tonal audibility in dB.
        adjustment: Tonal adjustment in dB.
        tone: Optional Tone object.
    """

    def __init__(  # pylint: disable=too-many-instance-attributes
        self,
        center,
        start,
        end,
        bandwidth,
        regression_range_factor,
        regression_slope,
        regression_intercept,
        noise_level,
        tone_level,
        audibility,
        adjustment,
        tone=None,
    ):
        self.center = center
        """Center frequency of the critical band."""
        self.start = start
        """Lower band-edge frequency of the critical band."""
        self.end = end
        """Upper band-edge frequency of the critical band."""
        self.bandwidth = bandwidth
        """Bandwidth of the critical band."""
        self.regression_range_factor = regression_range_factor
        """Range of regression factor. See also :attr:`REGRESSION_RANGE_FACTOR`."""
        self.regression_slope = regression_slope
        """Linear regression slope."""
        self.regression_intercept = regression_intercept
        """Linear regression intercept."""
        self.masking_noise_level = noise_level
        """Masking noise level $L_{pn}$."""
        self.total_tone_level = tone_level
        """Total tone level $L_{pt}$."""
        self.tonal_audibility = audibility
        """Tonal audibility $L_{ta}$."""
        self.adjustment = adjustment
        """Adjustment $K_{t}$."""
        self.tone = tone

    def __str__(self):
        return "(center={:4.1f}, bandwidth={:4.1f}, tonal_audibility={:4.1f}, adjustment={:4.1f}".format(
            self.center, self.bandwidth, self.tonal_audibility, self.adjustment
        )

    def __repr__(self):
        return "CriticalBand{}".format(str(self))

    def _repr_html_(self):
        """HTML representation for Jupyter notebooks."""
        table = [
            ("Center frequency", "{:4.1f} Hz".format(self.center)),
            ("Start frequency", "{:4.1f} Hz".format(self.start)),
            ("End frequency", "{:4.1f} Hz".format(self.end)),
            ("Bandwidth", "{:4.1f} Hz".format(self.bandwidth)),
            ("Regression factor", "{:4.1f}".format(self.regression_range_factor)),
            ("Regression slope", "{:4.1f}".format(self.regression_slope)),
            ("Regression intercept", "{:4.1f}".format(self.regression_intercept)),
            ("Masking noise level", "{:4.1f} dB".format(self.masking_noise_level)),
            ("Total tone level", "{:4.1f} dB".format(self.total_tone_level)),
            ("Tonal audibility $L_{ta}$", "{:4.1f} dB".format(self.tonal_audibility)),
            ("Adjustment $K_{t}$", "{:4.1f} dB".format(self.adjustment)),
        ]

        return tabulate(table, tablefmt="html")


# ----------Noise pauses----------------------------


def _search_noise_pauses(levels, tsc):
    """Search for noise pauses in a level sequence.

    Args:
        levels: Array of level values.
        tsc: Tone seeking criterion in dB.

    Returns:
        list: List of tuples containing (start_index, end_index) for each noise pause.
    """
    pauses = list()
    possible_start = None
    for i in range(2, len(levels) - 2):
        if (levels[i] - levels[i - 1]) >= tsc and (levels[i - 1] - levels[i - 2]) < tsc:
            possible_start = i
        if (levels[i] - levels[i + 1]) >= tsc and (levels[i + 1] - levels[i + 2]) < tsc:
            if possible_start:
                pauses.append((possible_start, i))
                possible_start = None
    return pauses


def noise_pause_seeker(levels, tsc):
    """Given the levels of a spectrum and a tone seeking criterium
    this top level function seeks possible noise pauses.

    Args:
        levels: Spectral levels in dB.
        tsc: Tone seeking criterion in dB.

    Returns:
        list: List of tuples containing (start_index, end_index) for each noise pause,
            sorted and filtered to avoid overlapping intervals.

    Note:
    Possible start and end indices of noise pauses are determined using `possible_noise_pauses.

    Then, only those that correspond to the smallest intervals that do not overlap other intervals are kept.
    """
    n = len(levels)
    forward_pauses = _search_noise_pauses(levels, tsc)
    backward_pauses = _search_noise_pauses(levels[::-1], tsc)
    backward_pauses = [
        (n - 1 - start, n - 1 - end) for end, start in reversed(backward_pauses)
    ]
    possible_pauses = sorted(list(set(forward_pauses) & set(backward_pauses)))
    return possible_pauses


# ------------------- Tone seeking--------------------


def determine_tone_lines(
    levels,
    df,
    start,
    end,
    force_tone_without_pause=False,
    force_bandwidth_criterion=False,
):
    """Determine tone lines in a noise pause.

    Args:
        levels: Series with levels as function of frequency.
        df: Frequency resolution in Hz.
        start: Index of noise pause start.
        end: Index of noise pause end.
        force_tone_without_pause: Force tone detection without pause. Defaults to False.
        force_bandwidth_criterion: Force bandwidth criterion. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - ndarray: Indices of tone lines
            - float: -3 dB bandwidth in Hz
    """
    # Noise pause range object
    npr = slice(start, end + 1)

    # Return values
    tone_indices = np.array([])
    bandwidth_for_tone_criterion = None

    # Levels but with integeres as indices instead of frequencies.
    # Benefit over np.array is that the index is maintained when the object is sliced.
    levels_int = levels.reset_index(drop=True)

    # If any of the lines is six 6 dB above. See section C.4.3.
    if (
        np.any(
            (
                levels.iloc[npr]
                >= TONE_WITHIN_PAUSE_CRITERION_DB + levels.iloc[start - 1]
            )
            & (
                levels.iloc[npr]
                >= TONE_WITHIN_PAUSE_CRITERION_DB + levels.iloc[end + 1]
            )
        )
        or force_tone_without_pause
    ):
        # Indices of values that are within -3 dB point.
        indices_3db = (
            (levels.iloc[npr] >= levels.iloc[npr].max() - TONE_BANDWIDTH_CRITERION_DB)
            .to_numpy()
            .nonzero()[0]
        )
        # -3 dB bandwidth
        bandwidth_for_tone_criterion = (indices_3db.max() - indices_3db.min()) * df
        # Frequency of tone.
        tone_center_frequency = levels.iloc[npr].idxmax()
        # tone_center_index = levels.reset_index(drop=True).iloc[npr].idxmax()
        # Critical band
        _, _, _, critical_band_bandwidth = critical_band(tone_center_frequency)

        # Fullfill bandwidth criterion? See section C.4.3
        if (
            bandwidth_for_tone_criterion < 0.10 * critical_band_bandwidth
        ) or force_bandwidth_criterion:
            # All values within 6 decibel are designated as tones.
            tone_indices = (
                levels_int.iloc[npr][
                    levels_int.iloc[npr]
                    >= levels_int.iloc[npr].max() - TONE_LINES_CRITERION_DB
                ]
            ).index.values

    return tone_indices, bandwidth_for_tone_criterion
