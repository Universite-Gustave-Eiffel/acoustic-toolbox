"""The descriptors module offers all kinds of acoustics related descriptors.

Descriptors from [ISO/TR 25417:2007](https://www.iso.org/standard/51150.html).

Attributes:
    REFERENCE_PRESSURE: Reference pressure for sound pressure level calculations.
    REFERENCE_SOUND_EXPOSURE: Reference sound exposure for sound exposure level calculations.
    REFERENCE_POWER: Reference power for sound power level calculations.
    REFERENCE_ENERGY: Reference energy for sound energy level calculations.
    REFERENCE_INTENSITY: Reference intensity for sound intensity level calculations.

Functions:
    sound_pressure_level: Calculate sound pressure level.
    equivalent_sound_pressure_level: Calculate equivalent sound pressure level.
    peak_sound_pressure: Calculate peak sound pressure.
    peak_sound_pressure_level: Calculate peak sound pressure level.
    sound_exposure: Calculate sound exposure.
    sound_exposure_level: Calculate sound exposure level.
    sound_power_level: Calculate sound power level.
    sound_energy: Calculate sound energy.
    sound_energy_level: Calculate sound energy level.
    sound_intensity: Calculate sound intensity.
    time_averaged_sound_intensity: Calculate time-averaged sound intensity.
    time_averaged_sound_intensity_level: Calculate time-averaged sound intensity level.
    normal_time_averaged_sound_intensity: Calculate normal time-averaged sound intensity.
    normal_time_averaged_sound_intensity_level: Calculate normal time-averaged sound intensity level.

Other descriptors
*****************

"""

import numpy as np

from acoustic_toolbox.standards.iso_tr_25417_2007 import (
    REFERENCE_PRESSURE,
    sound_pressure_level,
    equivalent_sound_pressure_level,
    peak_sound_pressure,
    peak_sound_pressure_level,
    REFERENCE_SOUND_EXPOSURE,
    sound_exposure,
    sound_exposure_level,
    REFERENCE_POWER,
    sound_power_level,
    sound_energy,
    REFERENCE_ENERGY,
    sound_energy_level,
    sound_intensity,
    time_averaged_sound_intensity,
    REFERENCE_INTENSITY,
    time_averaged_sound_intensity_level,
    normal_time_averaged_sound_intensity,
    normal_time_averaged_sound_intensity_level,
)

from acoustic_toolbox.standards.iso_1996_1_2003 import composite_rating_level


def _leq(levels, time):
    levels = np.asarray(levels)
    return 10.0 * np.log10((1.0 / time) * np.sum(10.0 ** (levels / 10.0)))


def leq(levels, int_time=1.0) -> float:
    """Equivalent level $L_{eq}$.

    Args:
      levels: Levels as function of time.
      int_time: Integration time in seconds.

    Returns:
      Equivalent level $L_{eq}$.
    """
    levels = np.asarray(levels)
    time = levels.size * int_time
    return _leq(levels, time)


def sel(levels: np.ndarray) -> float:
    """Sound Exposure Level from ``levels``."""
    levels = np.asarray(levels)
    return _leq(levels, 1.0)


def lw(W, Wref=1.0e-12) -> float:
    """Sound power level $L_{w}$ for sound power $W$ and reference power $W_{ref}$.

    Args:
      W: Sound power $W$.
      Wref: Reference power $W_{ref}$. Default value is $10^{12}$ watt.

    Returns:
      Sound power level $L_{w}$.
    """
    W = np.asarray(W)
    return 10.0 * np.log10(W / Wref)


def lden(
    lday,
    levening,
    lnight,
    hours: tuple[float, float, float] = (12.0, 4.0, 8.0),
    adjustment: tuple[float, float, float] = (0.0, 5.0, 10.0),
) -> float:
    """Calculate $L_{den}$ from $L_{day}$, $L_{evening}$ and $L_{night}$.

    Args:
      lday: Equivalent level during day period $L_{day}$.
      levening: Equivalent level during evening period $L_{evening}$.
      lnight: Equivalent level during night period $L_{night}$.
      hours: Hours per period.
      adjustment: Correction factor per period.

    Returns:
      $L_{den}$

    See Also:
      [composite_rating_level][acoustic_toolbox.standards.iso_1996_1_2003.composite_rating_level]

    """
    lday = np.asarray(lday)
    levening = np.asarray(levening)
    lnight = np.asarray(lnight)
    return composite_rating_level(
        np.vstack((lday, levening, lnight)).T, hours, adjustment
    )


def ldn(
    lday,
    lnight,
    hours: tuple[float, float] = (15.0, 9.0),
    adjustment: tuple[float, float] = (0.0, 10.0),
) -> float:
    """Calculate $L_{dn}$ from $L_{day}$ and $L_{night}$.

    Args:
      lday: Equivalent level during day period $L_{day}$.
      lnight: Equivalent level during night period $L_{night}$.
      hours: Hours per period.
      adjustment: Correction factor per period.

    Returns:
      $L_{dn}$

    See Also:
      [composite_rating_level][acoustic_toolbox.standards.iso_1996_1_2003.composite_rating_level]
    """
    lday = np.asarray(lday)
    lnight = np.asarray(lnight)
    return composite_rating_level(np.vstack((lday, lnight)).T, hours, adjustment)


__all__ = [
    # Following we take from another module
    "REFERENCE_PRESSURE",
    "sound_pressure_level",
    "equivalent_sound_pressure_level",
    "peak_sound_pressure",
    "peak_sound_pressure_level",
    "REFERENCE_SOUND_EXPOSURE",
    "sound_exposure",
    "sound_exposure_level",
    "REFERENCE_POWER",
    "sound_power_level",
    "sound_energy",
    "REFERENCE_ENERGY",
    "sound_energy_level",
    "sound_intensity",
    "time_averaged_sound_intensity",
    "REFERENCE_INTENSITY",
    "time_averaged_sound_intensity_level",
    "normal_time_averaged_sound_intensity",
    "normal_time_averaged_sound_intensity_level",
    # Following are locally defined
    "leq",
    "sel",
    "lw",
    "lden",
    "ldn",
]
