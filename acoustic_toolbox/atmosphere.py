"""
The atmosphere module contains functions and classes related to atmospheric acoustics and is based on
[ISO 9613-1:1993](standards/iso_9613_1_1993.md).

Classes:
    Atmosphere: Class describing atmospheric conditions.

Functions:
    soundspeed: Calculate the speed of sound.
    saturation_pressure: Calculate the saturation pressure.
    molar_concentration_water_vapour: Calculate the molar concentration of water vapour.
    relaxation_frequency_nitrogen: Calculate the relaxation frequency of nitrogen.
    relaxation_frequency_oxygen: Calculate the relaxation frequency of oxygen.
    attenuation_coefficient: Calculate the attenuation coefficient.
"""

import numpy as np
import matplotlib.pyplot as plt

import acoustic_toolbox
from acoustic_toolbox.standards.iso_9613_1_1993 import (
    SOUNDSPEED,
    REFERENCE_TEMPERATURE,
    REFERENCE_PRESSURE,
    TRIPLE_TEMPERATURE,
    soundspeed,
    saturation_pressure,
    molar_concentration_water_vapour,
    relaxation_frequency_oxygen,
    relaxation_frequency_nitrogen,
    attenuation_coefficient,
)


class Atmosphere:
    """Class describing atmospheric conditions."""

    REF_TEMP = 293.15
    """Reference temperature"""

    REF_PRESSURE = 101.325
    """International Standard Atmosphere in kilopascal"""

    TRIPLE_TEMP = 273.16
    """Triple point isotherm temperature."""

    def __init__(
        self,
        temperature=REFERENCE_TEMPERATURE,
        pressure=REFERENCE_PRESSURE,
        relative_humidity=0.0,
        reference_temperature=REFERENCE_TEMPERATURE,
        reference_pressure=REFERENCE_PRESSURE,
        triple_temperature=TRIPLE_TEMPERATURE,
    ):
        """
        Initialize the atmosphere.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.
            relative_humidity: Relative humidity.
            reference_temperature: Reference temperature.
            reference_pressure: Reference pressure.
            triple_temperature: Triple temperature.
        """

        self.temperature = temperature
        """Ambient temperature $T$."""

        self.pressure = pressure
        """Ambient pressure $p_a$."""

        self.relative_humidity = relative_humidity
        """Relative humidity"""

        self.reference_temperature = reference_temperature
        """Reference temperature."""

        self.reference_pressure = reference_pressure
        """Reference pressure."""

        self.triple_temperature = triple_temperature
        """Triple temperature."""

    def __repr__(self):
        return "Atmosphere{}".format(self.__str__())

    def __str__(self):
        attributes = [
            "temperature",
            "pressure",
            "relative_humidity",
            "reference_temperature",
            "reference_pressure",
            "triple_temperature",
        ]
        return "({})".format(
            ", ".join(
                map(lambda attr: "{}={}".format(attr, getattr(self, attr)), attributes)
            )
        )

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.__class__ == other.__class__

    @property
    def soundspeed(self):
        """Speed of sound $c$.

        The speed of sound is calculated using [soundspeed][acoustic_toolbox.standards.iso_9613_1_1993.soundspeed].

        Returns:
            float: The speed of sound.
        """
        return soundspeed(
            self.temperature,
            self.reference_temperature,
        )

    @property
    def saturation_pressure(self):
        """Saturation pressure $p_{sat}$.

        The saturation pressure is calculated using [saturation_pressure][acoustic_toolbox.standards.iso_9613_1_1993.saturation_pressure].

        Returns:
            float: The saturation pressure.
        """
        return saturation_pressure(
            self.temperature,
            self.reference_pressure,
            self.triple_temperature,
        )

    @property
    def molar_concentration_water_vapour(self):
        """Molar concentration of water vapour $h$.

        The molar concentration of water vapour is calculated using
        [molar_concentration_water_vapour][acoustic_toolbox.standards.iso_9613_1_1993.molar_concentration_water_vapour].

        Returns:
            float: The molar concentration of water vapour.
        """
        return molar_concentration_water_vapour(
            self.relative_humidity,
            self.saturation_pressure,
            self.pressure,
        )

    @property
    def relaxation_frequency_nitrogen(self):
        """Resonance frequency of nitrogen $f_{r,N}$.

        The resonance frequency is calculated using
        [relaxation_frequency_nitrogen][acoustic_toolbox.standards.iso_9613_1_1993.relaxation_frequency_nitrogen].

        Returns:
            float: The resonance frequency of nitrogen.
        """
        return relaxation_frequency_nitrogen(
            self.pressure,
            self.temperature,
            self.molar_concentration_water_vapour,
            self.reference_pressure,
            self.reference_temperature,
        )

    @property
    def relaxation_frequency_oxygen(self):
        """Resonance frequency of oxygen $f_{r,O}$.

        The resonance frequency is calculated using
        [relaxation_frequency_oxygen][acoustic_toolbox.standards.iso_9613_1_1993.relaxation_frequency_oxygen].

        Returns:
            float: The resonance frequency of oxygen.
        """
        return relaxation_frequency_oxygen(
            self.pressure,
            self.molar_concentration_water_vapour,
            self.reference_pressure,
        )

    def attenuation_coefficient(self, frequency):
        """Attenuation coefficient $\\alpha$ describing atmospheric absorption in dB/m.

        The attenuation coefficient is calculated using
        [attenuation_coefficient][acoustic_toolbox.standards.iso_9613_1_1993.attenuation_coefficient].

        Args:
            frequency: Frequencies to be considered.

        Returns:
            float: The attenuation coefficient.
        """
        return attenuation_coefficient(
            self.pressure,
            self.temperature,
            self.reference_pressure,
            self.reference_temperature,
            self.relaxation_frequency_nitrogen,
            self.relaxation_frequency_oxygen,
            frequency,
        )

    def frequency_response(self, distance, frequencies, inverse=False):
        """Calculate the frequency response.

        Args:
            distance: Distance between source and receiver.
            frequencies: Frequencies for which to compute the response.
            inverse: Whether the attenuation should be undone. Defaults to False.

        Returns:
            array: The frequency response.
        """
        return frequency_response(
            self,
            distance,
            frequencies,
            inverse,
        )

    def impulse_response(self, distance, fs, ntaps=None, inverse=False):
        """Calculate the impulse response of sound travelling through atmosphere.

        Args:
            distance: Distance between source and receiver.
            fs: Sample frequency.
            ntaps: Amount of taps. Defaults to None.
            inverse: Whether the attenuation should be undone. Defaults to False.

        Returns:
            array: The impulse response.

        See Also:
            [impulse_response][acoustic_toolbox.atmosphere.impulse_response]
        """
        return impulse_response(
            self,
            distance,
            fs,
            ntaps,
            inverse,
        )

    def plot_attenuation_coefficient(self, frequency):
        """Plot the attenuation coefficient $\\alpha$ as function of frequency.

        Args:
            frequency: Frequencies.

        Note:
            The attenuation coefficient is plotted in dB/km!

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.plot(frequency, self.attenuation_coefficient(frequency) * 1000.0)
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlabel(r"$f$ in Hz")
        ax0.set_ylabel(r"$\alpha$ in dB/km")
        ax0.legend()

        return fig


def frequency_response(atmosphere, distance, frequencies, inverse=False):
    """Calculate the single-sided frequency response.

    Args:
        atmosphere: Atmosphere instance.
        distance: Distance between source and receiver.
        frequencies: Frequencies for which to compute the response.
        inverse: Whether the attenuation should be undone. Defaults to False.

    Returns:
        array: The frequency response.
    """
    sign = +1 if inverse else -1
    tf = 10.0 ** (
        float(sign) * distance * atmosphere.attenuation_coefficient(frequencies) / 20.0
    )
    return tf


def impulse_response(atmosphere, distance, fs, ntaps, inverse=False):
    """Calculate the impulse response of sound travelling through atmosphere.

    The attenuation is calculated for a set of positive frequencies. Because the
    attenuation is the same for the negative frequencies, we have Hermitian
    symmetry. The attenuation is entirely real-valued. We like to have a constant
    group delay and therefore we need a linear-phase filter.

    This function creates a zero-phase filter, which is the special case of a
    linear-phase filter with zero phase slope. The type of filter is non-causal. The
    impulse response of the filter is made causal by rotating it by M/2 samples and
    discarding the imaginary parts. A real, even impulse response corresponds to a
    real, even frequency response.

    Args:
        atmosphere: Atmosphere instance.
        distance: Distance between source and receiver.
        fs: Sample frequency.
        ntaps: Amount of taps.
        inverse: Whether the attenuation should be undone. Defaults to False.

    Returns:
        array: The impulse response.
    """
    # Frequencies vector with positive frequencies only.
    frequencies = np.fft.rfftfreq(ntaps, 1.0 / fs)
    # Single-sided spectrum. Negative frequencies have the same values.
    tf = frequency_response(atmosphere, distance, frequencies, inverse)
    # Impulse response. We design a zero-phase filter (linear-phase with zero slope).
    # We need to shift the IR to make it even. Taking the real part should not be necessary, see above.
    # ir = np.fft.ifftshift(np.fft.irfft(tf, n=ntaps)).real
    ir = acoustic_toolbox.signal.impulse_response_real_even(tf, ntaps=ntaps)
    return ir


__all__ = [
    "Atmosphere",
    "SOUNDSPEED",
    "REFERENCE_TEMPERATURE",
    "REFERENCE_TEMPERATURE",
    "TRIPLE_TEMPERATURE",
    "soundspeed",
    "saturation_pressure",
    "molar_concentration_water_vapour",
    "relaxation_frequency_oxygen",
    "relaxation_frequency_nitrogen",
    "attenuation_coefficient",
    "impulse_response",
    "frequency_response",
]
