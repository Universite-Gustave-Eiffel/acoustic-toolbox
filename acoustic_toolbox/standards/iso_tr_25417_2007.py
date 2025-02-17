"""ISO/TR 25417:2007 specifies definitions of acoustical quantities and terms used
in noise measurement documents prepared by ISO Technical Committee TC 43,
Acoustics, Subcommittee SC 1, Noise, together with their symbols and units, with
the principal aim of harmonizing the terminology used in [ISO/TR 25417](http://www.iso.org/iso/home/store/catalogue_tc/catalogue_detail.htm?csnumber=42915).
"""  # noqa: D205

import numpy as np

REFERENCE_PRESSURE = 2.0e-5
r"""Reference value of the sound pressure $p_0$ is $2 \cdot 10^{-5}$ Pa."""


def sound_pressure_level(
    pressure, reference_pressure=REFERENCE_PRESSURE
) -> float | np.ndarray:
    r"""Sound pressure level $L_p$ in dB.

    The sound pressure level is calculated as:
    $$
    L_p = 10 \log_{10}{ \left( \frac{p^2}{p_0^2} \right)}
    $$

    See section 2.2 of the standard.

    Args:
        pressure: Instantaneous sound pressure $p$.
        reference_pressure: Reference value $p_0$.

    Returns:
        Sound pressure level
    """
    return 10.0 * np.log10(pressure**2.0 / reference_pressure**2.0)


def equivalent_sound_pressure_level(
    pressure, reference_pressure=REFERENCE_PRESSURE, axis=-1
) -> float | np.ndarray:
    r"""Time-averaged sound pressure level $L_{p,T}$ or equivalent-continious sound pressure level $L_{p,eqT}$ in dB.

    The time-averaged sound pressure level is calculated as:
    $$
    L_{p,T} = L_{p,eqT} = 10.0 \log_{10}{ \left( \frac{\frac{1}{T} \int_{t_1}^{t_2} p^2 (t) \mathrm{d} t  }{p_0^2} \right)}
    $$

    See section 2.3 of the standard.

    Args:
        pressure: Instantaneous sound pressure $p$.
        reference_pressure: Reference value $p_0$.
        axis: Axis

    Returns:
        Time-averaged sound pressure level
    """
    return 10.0 * np.log10((pressure**2.0).mean(axis=axis) / reference_pressure**2.0)


def max_sound_pressure_level(
    pressure, reference_pressure=REFERENCE_PRESSURE, axis=-1
) -> float | np.ndarray:
    r"""Maximum time-averaged sound pressure level $L_{F,max}$ in dB.

    Args:
        pressure: Instantaneous sound pressure $p$.
        reference_pressure: Reference value $p_0$.
        axis: Axis

    Returns:
        Maximum sound pressure level $\mathrm{max}{(L_{p})}$
    """
    return sound_pressure_level(pressure, reference_pressure=reference_pressure).max(
        axis=axis
    )


def peak_sound_pressure(pressure, axis=-1) -> float | np.ndarray:
    r"""Peak sound pressure $p_{peak}$ is the greatest absolute sound pressure during a certain time interval.

    Args:
        pressure: Instantaneous sound pressure $p$.
        axis: Axis

    Returns:
        Peak sound pressure $p_{peak} = \mathrm{max}(|p|)$
    """
    return np.abs(pressure).max(axis=axis)


def peak_sound_pressure_level(
    pressure, reference_pressure=REFERENCE_PRESSURE, axis=-1
) -> float | np.ndarray:
    r"""Peak sound pressure level $L_{p,peak}$ in dB.

    Args:
        pressure: Instantaneous sound pressure $p$.
        reference_pressure: Reference value $p_0$.
        axis: Axis

    Returns:
        Peak sound pressure level
            $$
            L_{p,peak} = 10.0 \log \frac{p_{peak}^2.0}{p_0^2}
            $$
    """
    return 10.0 * np.log10(
        peak_sound_pressure(pressure, axis=axis) ** 2.0 / reference_pressure**2.0
    )


REFERENCE_SOUND_EXPOSURE = 4.0e-10
r"""
Reference value of the sound exposure $E_0$ is $4 \cdot 10^{-12} \mathrm{Pa}^2\mathrm{s}$.
"""


def sound_exposure(pressure, fs, axis=-1) -> float | np.ndarray:
    r"""Calculate sound exposure $E_T$.

    Args:
        pressure: Instantaneous sound pressure $p$.
        fs: Sample frequency $f_s$.
        axis: Axis

    Returns:
        Sound exposure
            $$
            E_T = \int_{t_1}^{t_2} p^2(t) \mathrm{d}t
            $$
    """
    return (pressure**2.0 / fs).sum(axis=axis)


def sound_exposure_level(
    pressure, fs, reference_sound_exposure=REFERENCE_SOUND_EXPOSURE, axis=-1
) -> float | np.ndarray:
    r"""Sound exposure level $L_{E,T}$ in dB.

    Args:
        pressure: Instantaneous sound pressure $p$.
        fs: Sample frequency $f_s$.
        reference_sound_exposure: Reference sound exposure $E_T$.
        axis: Axis

    Returns:
        Sound exposure level
            $$
            L_{E,T} = 10 \log_{10}{ \frac{E_T}{E_0}  }
            $$
    """
    return 10.0 * np.log10(
        sound_exposure(pressure, fs, axis=axis) / reference_sound_exposure
    )


REFERENCE_POWER = 1.0e-12
"""
Reference value of the sound power $P_0$ is 1 pW.
"""


def sound_power_level(power, reference_power=REFERENCE_POWER) -> float | np.ndarray:
    r"""Sound power level $L_W$.

    Args:
        power: Sound power $P$.
        reference_power: Reference sound power $P_0$.

    Returns:
        Sound power level calculated as:
            $$
            10 \log_{10}{ \frac{P}{P_0}  }
            $$
    """
    return 10.0 * np.log10(power / reference_power)


def sound_energy(power, axis=-1) -> float | np.ndarray:
    r"""Sound energy $J$.

    Args:
        power: Sound power $P$.
        axis: Axis

    Returns:
        Sound energy
            $$
            J = \int_{t_1}^{t_2} P(t) \mathrm{d} t
            $$
    """
    return power.sum(axis=axis)


REFERENCE_ENERGY = 1.0e-12
"""Reference value of the sound energy $J_0$ is 1 pJ."""


def sound_energy_level(energy, reference_energy=REFERENCE_ENERGY) -> np.ndarray:
    r"""Sound energy level $L_J$ in dB.

    Args:
        energy: Sound energy $J$.
        reference_energy: Reference sound energy $J_0$.

    Returns:
        Sound energy level
            $$
            L_{J} = 10 \log_{10}{ \frac{J}{J_0} }
            $$
    """
    return np.log10(energy / reference_energy)


def sound_intensity(pressure, velocity) -> np.ndarray:
    r"""Sound intensity $\mathbf{i}$.

    Args:
        pressure: Sound pressure $p(t)$.
        velocity: Particle velocity $\mathbf{u}(t)$.

    Returns:
        Sound intensity
            $$
            \mathbf{i} = p(t) \cdot \mathbf{u}(t)
            $$
    """
    return pressure * velocity


REFERENCE_INTENSITY = 1.0e-12
r"""Reference value of the sound intensity $I_0$ is $\mathrm{1 pW/m^2}$."""


def time_averaged_sound_intensity(intensity, axis=-1) -> float:
    r"""Time-averaged sound intensity $\mathbf{I}_T$.

    Args:
        intensity: Sound intensity $\mathbf{i}$.
        axis: Axis

    Returns:
        Time-averaged sound intensity
            $$
            I_T = \frac{1}{T} \int_{t_1}^{t_2} \mathbf{i}(t)
            $$
    """
    return intensity.mean(axis=axis)


def time_averaged_sound_intensity_level(
    time_averaged_sound_intensity, reference_intensity=REFERENCE_INTENSITY, axis=-1
) -> float:
    r"""Time-averaged sound intensity level $L_{I,T}$.

    Args:
        time_averaged_sound_intensity: Time-averaged sound intensity $\mathbf{I}_T$.
        reference_intensity: Reference sound intensity $I_0$.
        axis: Axis along which to calculate norm.

    Returns:
        Time-averaged sound intensity level calculated as:
            $$
            L_{I,T} = 10 \log_{10} { \frac{|\mathbf{I}_T|}{I_0} }
            $$
    """
    return 10.0 * np.log10(
        np.linalg.norm(time_averaged_sound_intensity, axis=axis) / reference_intensity
    )


def normal_time_averaged_sound_intensity(
    time_averaged_sound_intensity, unit_normal_vector
) -> float:
    r"""Normal time-averaged sound intensity $\mathbf{I}_{n,T}$.

    Args:
        time_averaged_sound_intensity: Time-averaged sound intensity $\mathbf{I}_T$.
        unit_normal_vector: Unit normal vector $\mathbf{n}$.

    Returns:
        Normal time-averaged sound intensity
            $$
            I_{n,T} = \mathbf{I}_T \cdot \mathbf{n}
            $$
    """
    return time_averaged_sound_intensity.dot(unit_normal_vector)


def normal_time_averaged_sound_intensity_level(
    normal_time_averaged_sound_intensity, reference_intensity=REFERENCE_INTENSITY
) -> float:
    r"""Normal time-averaged sound intensity level $L_{I_{n,T}}$ in dB.

    Args:
        normal_time_averaged_sound_intensity: Normal time-averaged sound intensity $I_{n,T}$.
        reference_intensity: Reference sound intensity $I_0$.

    Returns:
        Normal time-averaged sound intensity level calculated as:
            $$
            I_{n,T} = 10 \log_{10} { \frac{|I_{n,T}|}{I_0}}
            $$
    """
    return 10.0 * np.log10(
        np.abs(normal_time_averaged_sound_intensity / reference_intensity)
    )
