"""ISO 1996-1:2003 defines the basic quantities to be used for the description of
noise in community environments and describes basic assessment procedures. It
also specifies methods to assess environmental noise and gives guidance on
predicting the potential annoyance response of a community to long-term exposure
from various types of environmental noises. The sound sources can be separate or
in various combinations. Application of the method to predict annoyance response
is limited to areas where people reside and to related long-term land uses.

Reference:
    ISO 1996-1:2003: Description, measurement and assessment of environmental noise
"""

import numpy as np


def composite_rating_level(
    levels: np.ndarray, hours: np.ndarray, adjustment: np.ndarray
) -> float | np.ndarray:
    r"""Composite rating level.

    The composite rating level is calculated as:

    $$
    L_R = 10 \log{\left[ \sum_i \frac{d_i}{24} 10^{(L_i+K_i)/10}  \right]}
    $$

    where $i$ is a period. See equation 6 and 7 of the standard.

    Note:
      Summation is done over the last axis

    Args:
        levels: Level per period
        hours: Amount of hours per period
        adjustment: Adjustment per period

    Returns:
        The composite rating level in dB as a float if a scalar result is obtained, or as
        an ndarray if multiple periods are processed.

    """
    levels = np.asarray(levels)
    hours = np.asarray(hours)
    adjustment = np.asarray(adjustment)

    return 10.0 * np.log10(
        (hours / 24.0 * 10.0 ** ((levels + adjustment) / 10.0)).sum(axis=-1)
    )
