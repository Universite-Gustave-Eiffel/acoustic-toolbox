"""# ISO 9613-1:1993

This module implements ISO 9613-1:1993 which provides analytical methods for calculating
sound attenuation due to atmospheric absorption under various meteorological conditions.

The module provides functions for calculating:
- Sound speed in different temperatures
- Saturation vapor pressure
- Molar concentration of water vapor
- Relaxation frequencies of oxygen and nitrogen
- Atmospheric absorption coefficients

Constants:
    [SOUNDSPEED][acoustic_toolbox.standards.iso_9613_1_1993.SOUNDSPEED]: Speed of sound constant.

    [REFERENCE_TEMPERATURE][acoustic_toolbox.standards.iso_9613_1_1993.REFERENCE_TEMPERATURE]: Reference temperature constant.

    [REFERENCE_PRESSURE][acoustic_toolbox.standards.iso_9613_1_1993.REFERENCE_PRESSURE]: Reference pressure constant.

    [TRIPLE_TEMPERATURE][acoustic_toolbox.standards.iso_9613_1_1993.TRIPLE_TEMPERATURE]: Triple point isotherm temperature constant.

Reference:
    ISO 9613-1:1993: Acoustics — Attenuation of sound during propagation outdoors
"""

import numpy as np
from numpy.typing import NDArray

SOUNDSPEED = 343.2
"""Speed of sound."""

REFERENCE_TEMPERATURE = 293.15
"""Reference temperature."""

REFERENCE_PRESSURE = 101.325
"""International Standard Atmosphere pressure in kilopascal."""

TRIPLE_TEMPERATURE = 273.16
"""Triple point isotherm temperature."""


def soundspeed(
    temperature: float, reference_temperature: float = REFERENCE_TEMPERATURE
) -> float:
    r"""Calculate speed of sound $c$ in air.

    Notes:
        The speed of sound is calculated using the formula:
        $$
        c = 343.2 \left( \frac{T}{T_0} \right)
        $$

    Args:
        temperature: Ambient temperature $T_0$.
        reference_temperature: Reference temperature $T$.

    Returns:
        float: Speed of sound in m/s.
    """
    return 343.2 * np.sqrt(temperature / reference_temperature)


def saturation_pressure(
    temperature: float,
    reference_pressure: float = REFERENCE_PRESSURE,
    triple_temperature: float = TRIPLE_TEMPERATURE,
) -> float:
    r"""Calculate saturation vapor pressure $p_{sat}$.

    Notes:
        The saturation vapor pressure is calculated using the formula:
        $$
        p_{sat} = 10^C \cdot p_r
        $$

        with exponent $C$ given by:
        $$
        C = -6.8346 \cdot \left( \frac{T_{01}}{T} \right)^{1.261}  + 4.6151
        $$

    Args:
        temperature: Ambient temperature $T$.
        reference_pressure: Reference pressure $p_r$.
        triple_temperature: Triple point temperature $T_{01}$.

    Returns:
        float: Saturation vapor pressure
    """
    return reference_pressure * 10.0 ** (
        -6.8346 * (triple_temperature / temperature) ** (1.261) + 4.6151
    )


def molar_concentration_water_vapour(
    relative_humidity: float, saturation_pressure: float, pressure: float
) -> float:
    r"""Calculate molar concentration of water vapour $h$.

    Notes:
        The molar concentration of water vapour is calculated using the formula:
        $$
        h = h_r  \frac{p_{sat}}{p_a}
        $$

    Args:
        relative_humidity: Relative humidity $h_r$
        saturation_pressure: Saturation pressure $p_{sat}$
        pressure: Ambient pressure $p_a$

    Returns:
        float: Molar concentration of water vapour
    """
    return relative_humidity * saturation_pressure / pressure


def relaxation_frequency_oxygen(
    pressure: float, h: float, reference_pressure: float = REFERENCE_PRESSURE
) -> float:
    r"""Calculate relaxation frequency of oxygen $f_{r,O}$.

    Notes:
        The relaxation frequency of oxygen is calculated using the formula:
        $$
        f_{r,O} = \frac{p_a}{p_r} \left( 24 + 4.04 \cdot 10^4 h \frac{0.02 + h}{0.391 + h}  \right)
        $$

    Args:
        pressure: Ambient pressure $p_a$
        h: Molar concentration of water vapour $h$
        reference_pressure: Reference pressure $p_r$.

    Returns:
        float: Relaxation frequency of oxygen in Hz

    See Also:
        [molar_concentration_water_vapour][acoustic_toolbox.standards.iso_9613_1_1993.molar_concentration_water_vapour]: Function to calculate molar concentration of water vapour
    """
    return (
        pressure
        / reference_pressure
        * (24.0 + 4.04 * 10.0**4.0 * h * (0.02 + h) / (0.391 + h))
    )


def relaxation_frequency_nitrogen(
    pressure: float,
    temperature: float,
    h: float,
    reference_pressure: float = REFERENCE_PRESSURE,
    reference_temperature: float = REFERENCE_TEMPERATURE,
) -> float:
    r"""Calculate relaxation frequency of nitrogen $f_{r,N}$.

    Notes:
        The relaxation frequency of nitrogen is calculated by:
        $$
        f_{r,N} = \frac{p_a}{p_r} \left( \frac{T}{T_0} \right)^{-1/2} \cdot \left( 9 + 280 \cdot h \cdot \exp \left( -4.170 \left[ \left(\frac{T}{T_0} \right)^{-1/3} -1 \right] \right) \right)
        $$

    Args:
        pressure: Ambient pressure $p_a$
        temperature: Ambient temperature $T$
        h: Molar concentration of water vapour $h$
        reference_pressure: Reference pressure $p_{ref}$.
        reference_temperature: Reference temperature $T_{ref}$.

    Returns:
        float: Relaxation frequency of nitrogen in Hz
    """
    return (
        pressure
        / reference_pressure
        * (temperature / reference_temperature) ** (-0.5)
        * (
            9.0
            + 280.0
            * h
            * np.exp(
                -4.170 * ((temperature / reference_temperature) ** (-1.0 / 3.0) - 1.0)
            )
        )
    )


def attenuation_coefficient(
    pressure: float,
    temperature: float,
    reference_pressure: float,
    reference_temperature: float,
    relaxation_frequency_nitrogen: float,
    relaxation_frequency_oxygen: float,
    frequency: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    r"""Calculate attenuation coefficient $\alpha$ describing atmospheric absorption in dB/m for the specified ``frequency``.

    Args:
        pressure: Ambient pressure $p_a$
        temperature: Ambient temperature $T$
        reference_pressure: Reference pressure $p_{ref}$
        reference_temperature: Reference temperature $T_{ref}$
        relaxation_frequency_nitrogen: Relaxation frequency of nitrogen $f_{r,N}$
        relaxation_frequency_oxygen: Relaxation frequency of oxygen $f_{r,O}$
        frequency: Frequencies $f$ to calculate $\alpha$ for

    Returns:
        float | NDArray[np.float64]: Attenuation coefficient in dB/m
    """
    return (
        8.686
        * frequency**2.0
        * (
            (
                1.84
                * 10.0 ** (-11.0)
                * (reference_pressure / pressure)
                * (temperature / reference_temperature) ** (0.5)
            )
            + (temperature / reference_temperature) ** (-2.5)
            * (
                0.01275
                * np.exp(-2239.1 / temperature)
                * (
                    relaxation_frequency_oxygen
                    + (frequency**2.0 / relaxation_frequency_oxygen)
                )
                ** (-1.0)
                + 0.1068
                * np.exp(-3352.0 / temperature)
                * (
                    relaxation_frequency_nitrogen
                    + (frequency**2.0 / relaxation_frequency_nitrogen)
                )
                ** (-1.0)
            )
        )
    )
