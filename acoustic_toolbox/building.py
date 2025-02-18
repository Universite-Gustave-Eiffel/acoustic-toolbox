"""The building module contains functions for calculating acoustic properties of building elements, including
sound transmission class (STC) and weighted sound reduction index ($R_w$) calculations.

Functions:
    rw_curve: Calculate the reference curve for weighted sound reduction index.
    rw: Calculate the weighted sound reduction index ($R_w$).
    rw_c: Calculate the weighted sound reduction index with spectrum adaptation term C ($R_w + C$).
    rw_ctr: Calculate the weighted sound reduction index with spectrum adaptation term $C_{tr}$ ($R_w + C_{tr}$).
    stc_curve: Calculate the Sound Transmission Class (STC) reference curve.
    stc: Calculate the Sound Transmission Class (STC).
    mass_law: Calculate transmission loss according to mass law.

"""

import numpy as np
from numpy.typing import NDArray


def rw_curve(tl: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the curve of $Rw$ from a NumPy array `tl` with third
    octave data between 100 Hz and 3.15 kHz.

    Args:
        tl: Transmission Loss

    Returns:
        NDArray[np.float64]: Reference curve values.
    """
    ref_curve = np.array([0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 23, 23, 23, 23, 23])
    residuals = 0
    while residuals > -32:
        ref_curve += 1
        diff = tl - ref_curve
        residuals = np.sum(np.clip(diff, np.min(diff), 0))
    ref_curve -= 1
    return ref_curve


def rw(tl: NDArray[np.float64]) -> float:
    """Calculate $R_W$ from a NumPy array `tl` with third octave data
    between 100 Hz and 3.15 kHz.

    Args:
        tl: Transmission Loss.

    Returns:
        float: The weighted sound reduction index.
    """
    return rw_curve(tl)[7]


def rw_c(tl: NDArray[np.float64]) -> float:
    """Calculate $R_W + C$ from a NumPy array `tl` with third octave data
    between 100 Hz and 3.15 kHz.

    Args:
        tl: Transmission Loss.

    Returns:
        float: The weighted sound reduction index with spectrum adaptation term C.
    """
    k = np.array(
        [-29, -26, -23, -21, -19, -17, -15, -13, -12, -11, -10, -9, -9, -9, -9, -9]
    )
    a = -10 * np.log10(np.sum(10 ** ((k - tl) / 10)))
    return a


def rw_ctr(tl: NDArray[np.float64]) -> float:
    """Calculate $R_W + C_{tr}$ from a NumPy array `tl` with third octave
    data between 100 Hz and 3.15 kHz.

    Args:
        tl: Transmission Loss.

    Returns:
        float: The weighted sound reduction index with spectrum adaptation term $C_{tr}$.
    """
    k_tr = np.array(
        [-20, -20, -18, -16, -15, -14, -13, -12, -11, -9, -8, -9, -10, -11, -13, -15]
    )
    a_tr = -10 * np.log10(np.sum(10 ** ((k_tr - tl) / 10)))
    return a_tr


def stc_curve(tl: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the Sound Transmission Class (STC) curve from a NumPy array `tl`
    with third octave data between 125 Hz and 4 kHz.

    Args:
        tl: Transmission Loss.

    Returns:
        NDArray[np.float64]: The STC reference curve values.
    """
    ref_curve = np.array([0, 3, 6, 9, 12, 15, 16, 17, 18, 19, 20, 20, 20, 20, 20, 20])
    top_curve = ref_curve
    res_sum = 0
    while True:
        diff = tl - top_curve
        residuals = np.clip(diff, np.min(diff), 0)
        res_sum = np.sum(residuals)
        if res_sum < -32:
            if np.any(residuals > -8):
                top_curve -= 1
                break
        top_curve += 1
    return top_curve


def stc(tl: NDArray[np.float64]) -> float:
    """Calculate the Sound Transmission Class (STC) from a NumPy array `tl` with
    third octave data between 125 Hz and 4 kHz.

    Args:
        tl: Transmission Loss.

    Returns:
        float: The Sound Transmission Class value.
    """
    return stc_curve(tl)[6]


def mass_law(
    freq: float | NDArray[np.float64],
    vol_density: float,
    thickness: float,
    theta: float = 0,
    c: float = 343,
    rho0: float = 1.225,
) -> float | NDArray[np.float64]:
    r"""Calculate transmission loss according to mass law.

    Args:
        freq: Frequency of interest in Hz.
        vol_density: Volumetric density of material in $frac{kg}{m^3}$.
        thickness: Thickness of wall in m.
        theta: Angle of incidence in degrees. Default value is 0 (normal incidence).
        c: Speed of sound in m/s. Defaults to 343.
        rho0: Density of air in $\frac{kg}{m^3}$. Defaults to 1.225.

    Returns:
        float | NDArray[np.float64]: Transmission loss value(s) in dB.
    """
    rad_freq = 2.0 * np.pi * freq
    surface_density = vol_density * thickness
    theta_rad = np.deg2rad(theta)
    a = rad_freq * surface_density * np.cos(theta_rad) / (2 * rho0 * c)
    tl_theta = 10 * np.log10(1 + a**2)
    return tl_theta


__all__ = ["rw_curve", "rw", "rw_c", "rw_ctr", "stc_curve", "stc", "mass_law"]
