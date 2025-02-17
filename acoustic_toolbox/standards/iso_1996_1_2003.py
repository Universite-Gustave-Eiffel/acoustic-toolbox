"""
This module implements ISO 1996-1:2003 which defines basic quantities and procedures
for describing and assessing environmental noise in communities.

The standard provides:
- Basic quantities for noise description
- Assessment procedures for environmental noise
- Methods for predicting community annoyance response
- Guidance for various types of environmental noise sources

Note:
    - Sound sources can be assessed separately or in combination
    - Annoyance prediction is limited to residential areas and related long-term land uses

Reference:
    ISO 1996-1:2003: Description, measurement and assessment of environmental noise
"""

import numpy as np


def composite_rating_level(
    levels: np.ndarray, hours: np.ndarray, adjustment: np.ndarray
) -> float | np.ndarray:
    """Calculate composite rating level (LR) for a whole day.

    Combines noise levels from different time periods, accounting for
    the duration of each period and any adjustments. The composite rating level is
    calculated as:
        $$
        LR = 10 \\cdot \\log_{10} \\left[\\sum \\left(\\frac{d_i}{24} \\cdot 10^{((L_i + K_i) / 10)}\\right)\\right]
        $$

    where:

      - $L_i$ is the noise level (in dB) for period $i$,
      - $d_i$ is the duration (in hours) for period $i$, and
      - $K_i$ is the adjustment (in dB) for period $i$.

    Note:
        - Implementation of equations 6 and 7 from the standard
        - Summation is performed over the last axis of input arrays

    Args:
        levels: Level $L_i$ per period in dB.
        hours: Duration $d_i$ per period in hours.
        adjustment: Adjustment $K_i$ per period in dB.

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
