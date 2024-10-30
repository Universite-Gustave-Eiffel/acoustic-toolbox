"""
Ambisonics
==========

"""

import numpy as np
import scipy.special
from typing import Generator, Tuple


def acn(order: int = 1) -> Generator[Tuple[int, int], None, None]:
    """Spherical harmonic degree `n` and order `m` for ambisonics order `order`.

    Follows ACN.

    | ACN | n | m  | letter |
    |-----|---|----|--------|
    | 0   | 0 | 0  | W      |
    | 1   | 1 | -1 | Y      |
    | 2   | 1 | 0  | Z      |
    | 3   | 1 | +1 | X      |

    Args:
      order: Ambisonics order.

    Yields:
        Degree `n` and order `m`.
    """
    for n in range(order + 1):
        for m in range(-n, n + 1):
            yield (n, m)


def sn3d(m, n):
    """SN3D or Schmidt semi-normalisation

    - [http://en.wikipedia.org/wiki/Ambisonic_data_exchange_formats#SN3D](http://en.wikipedia.org/wiki/Ambisonic_data_exchange_formats#SN3D)

    Args:
      m: order `n`
      n: degree `m`

    Returns:

    """
    m = np.atleast_1d(m)
    n = np.atleast_1d(n)

    d = np.logical_not(m.astype(bool))
    out = np.sqrt(
        (2.0 - d)
        / (4.0 * np.pi)
        * scipy.special.factorial(n - np.abs(m))
        / scipy.special.factorial.factorial(n + np.abs(m))
    )
    return out


def n3d(m, n):
    """N3D or full three-D normalisation

    - [http://en.wikipedia.org/wiki/Ambisonic_data_exchange_formats#N3D](http://en.wikipedia.org/wiki/Ambisonic_data_exchange_formats#N3D)

    Args:
      m: order `n`
      n: degree `m`

    Returns:

    """
    n = np.atleast_1d(n)
    return sn3d(m, n) * np.sqrt(2 * n + 1)


__all__ = ["acn", "sn3d", "n3d"]
