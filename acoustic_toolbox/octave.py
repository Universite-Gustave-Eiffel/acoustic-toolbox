"""Module for working with octaves.

The following is an example on how to use [acoustic_toolbox.octave.Octave][acoustic_toolbox.octave.Octave].

.. literalinclude:: ../../examples/example_octave.py

"""

import numpy as np
from acoustic_toolbox.standards import iec_61260_1_2014
from acoustic_toolbox.standards.iec_61260_1_2014 import index_of_frequency
from acoustic_toolbox.standards.iec_61260_1_2014 import REFERENCE_FREQUENCY as REFERENCE

# REFERENCE = 1000.0
# """
# Reference frequency.
# """

# from acoustic_toolbox.standards.iec_61260_1_2014 import index_of_frequency
# from acoustic_toolbox.standards.iec_61260_1_2014 import REFERENCE_FREQUENCY as REFERENCE


def exact_center_frequency(frequency=None, fraction=1, n=None, ref=REFERENCE):
    """Exact center frequency.

    Args:
      frequency: Frequency within the band.
      fraction: Band designator.
      n: Index of band.
      ref: Reference frequency.

    Returns:
      Exact center frequency for the given frequency or band index.

    See Also:
      - [acoustic_toolbox.standards.iec_61260_1_2014.exact_center_frequency][acoustic_toolbox.standards.iec_61260_1_2014.exact_center_frequency]
      - [acoustic_toolbox.standards.iec_61260_1_2014.index_of_frequency][acoustic_toolbox.standards.iec_61260_1_2014.index_of_frequency]
    """
    if frequency is not None:
        n = iec_61260_1_2014.index_of_frequency(frequency, fraction=fraction, ref=ref)
    return iec_61260_1_2014.exact_center_frequency(n, fraction=fraction, ref=ref)


def nominal_center_frequency(frequency=None, fraction=1, n=None):
    """Nominal center frequency.

    Note:
      Contrary to the other functions this function silently assumes 1000 Hz reference frequency.

    Args:
      frequency: Frequency within the band.
      fraction: Band designator.
      n: Index of band.

    Returns:
      The nominal center frequency for the given frequency or band index.

    See Also:
      - [acoustic_toolbox.standards.iec_61260_1_2014.exact_center_frequency][acoustic_toolbox.standards.iec_61260_1_2014.exact_center_frequency]
      - [acoustic_toolbox.standards.iec_61260_1_2014.nominal_center_frequency][acoustic_toolbox.standards.iec_61260_1_2014.nominal_center_frequency]


    """
    center = exact_center_frequency(frequency, fraction, n)
    return iec_61260_1_2014.nominal_center_frequency(center, fraction)


def lower_frequency(
    frequency=None, fraction=1, n=None, ref=REFERENCE
) -> float | np.ndarray:
    """Lower band-edge frequency.

    Args:
      frequency: Frequency within the band.
      fraction: Band designator.
      n: Index of band.
      ref: Reference frequency.

    Returns:
      Lower band-edge frequency for the given frequency or band index.

    See Also:
      - [acoustic_toolbox.standards.iec_61260_1_2014.exact_center_frequency][acoustic_toolbox.standards.iec_61260_1_2014.exact_center_frequency]
      - [acoustic_toolbox.standards.iec_61260_1_2014.lower_frequency][acoustic_toolbox.standards.iec_61260_1_2014.lower_frequency]
    """
    center = exact_center_frequency(frequency, fraction, n, ref=ref)
    return iec_61260_1_2014.lower_frequency(center, fraction)


def upper_frequency(
    frequency=None, fraction=1, n=None, ref=REFERENCE
) -> float | np.ndarray:
    """Upper band-edge frequency.

    Args:
      frequency: Frequency within the band.
      fraction: Band designator.
      n: Index of band.
      ref: Reference frequency.

    Returns:
      Upper band-edge frequency for the given frequency or band index.

    See Also:
      - [acoustic_toolbox.standards.iec_61260_1_2014.exact_center_frequency][acoustic_toolbox.standards.iec_61260_1_2014.exact_center_frequency]
      - [acoustic_toolbox.standards.iec_61260_1_2014.upper_frequency][acoustic_toolbox.standards.iec_61260_1_2014.upper_frequency]
    """
    center = exact_center_frequency(frequency, fraction, n, ref=ref)
    return iec_61260_1_2014.upper_frequency(center, fraction)


# -- things below will be deprecated?---#

frequency_of_band = iec_61260_1_2014.exact_center_frequency
"""Calculate the center frequency for a given band index."""

band_of_frequency = iec_61260_1_2014.index_of_frequency
"""Calculate the band index for a given frequency."""


class Octave:
    """Class to calculate octave center frequencies.

    Attributes:
      reference: Reference center frequency $f_{c,0}$.
      fraction: Fraction of octave.
      interval: Interval.
      fmin: Minimum frequency of a range.
      fmax: Maximum frequency of a range.
      unique: Whether or not to calculate the requested values for every value of ``interval``.
      reference: Reference frequency.
    """

    def __init__(
        self,
        fraction=1,
        interval=None,
        fmin=None,
        fmax=None,
        unique: bool = False,
        reference=REFERENCE,
    ):
        """Initialize the Octave class.

        Raises:
          AttributeError: If ``interval`` is not ``None`` and ``fmin`` or ``fmax`` is not ``None``.
        """
        self.reference = reference
        self.fraction = fraction

        if (interval is not None) and (fmin is not None or fmax is not None):
            raise AttributeError("Cannot specify both interval and fmin/fmax")
        self._interval = np.asarray(interval)
        self._fmin = fmin
        self._fmax = fmax
        self.unique = unique

    @property
    def fmin(self):
        """Minimum frequency of an interval."""
        if self._fmin is not None:
            return self._fmin
        elif self._interval is not None:
            return self.interval.min()
        else:
            raise ValueError("Incorrect fmin/interval")

    @fmin.setter
    def fmin(self, x):
        if self.interval is not None:
            pass  # Warning, remove interval first.
        else:
            self._fmin = x

    @property
    def fmax(self):
        """Maximum frequency of an interval."""
        if self._fmax is not None:
            return self._fmax
        elif self._interval is not None:
            return self.interval.max()
        else:
            raise ValueError("Incorrect fmax/interval")

    @fmax.setter
    def fmax(self, x):
        if self.interval is not None:
            pass
        else:
            self._fmax = x

    @property
    def interval(self):
        """Interval."""
        return self._interval

    @interval.setter
    def interval(self, x):
        if self._fmin or self._fmax:
            pass
        else:
            self._interval = np.asarray(x)

    def _n(self, f):
        """Calculate the band ``n`` from a given frequency.

        Args:
          f: Frequency

        See also [band_of_frequency][acoustic_toolbox.octave.band_of_frequency].
        """
        return band_of_frequency(f, fraction=self.fraction, ref=self.reference)

    def _fc(self, n):
        """Calculate center frequency of band ``n``.

        Args:
          n: band ``n`.

        See also [frequency_of_band][acoustic_toolbox.octave.frequency_of_band].
        """
        return frequency_of_band(n, fraction=self.fraction, ref=self.reference)

    @property
    def n(self):
        """Return band ``n`` for a given frequency."""
        if self.interval is not None and self.unique:
            return self._n(self.interval)
        else:
            return np.arange(self._n(self.fmin), self._n(self.fmax) + 1)

    @property
    def center(self) -> float:
        r"""Return center frequency $f_c$.

        Returns:
          Center frequency calculated as:
            $$
            f_c = f_{ref} \cdot 2^{n/N} \cdot 10^{\frac{3}{10N}}
            $$
        """
        n = self.n
        return self._fc(n)

    @property
    def bandwidth(self):
        """Bandwidth of bands.

        $$
        B = f_u - f_l
        $$
        """
        return self.upper - self.lower

    @property
    def lower(self) -> float:
        r"""Lower frequency limits of bands.

        Returns:
          Lower frequency calculated as:
            $$
            f_l = f_c \cdot 2^{\frac{-1}{2N}}
            $$

        See also [lower_frequency][acoustic_toolbox.octave.lower_frequency].
        """
        return lower_frequency(self.center, self.fraction)

    @property
    def upper(self):
        r"""Upper frequency limits of bands.

        Returns:
          float: Upper frequency calculated as:
            $$
            f_u = f_c \cdot 2^{\frac{1}{2N}}
            $$

        See Also:
          [upper_frequency][acoustic_toolbox.octave.upper_frequency].
        """
        return upper_frequency(self.center, self.fraction)


__all__ = [
    "exact_center_frequency",
    "nominal_center_frequency",
    "lower_frequency",
    "upper_frequency",
    "index_of_frequency",
    "Octave",
    "frequency_of_band",
    "band_of_frequency",  # These three will be deprecated?
]
