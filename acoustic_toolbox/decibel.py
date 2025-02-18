"""The `decibel` module contains basic functions for decibel arithmetic."""

import numpy as np


def dbsum(levels, axis=None) -> float:
    r"""Energetic summation of levels.

    Args:
      levels: Sequence of levels.
      axis: Axis over which to perform the operation.
        $$
        L_{sum} = 10 \log_{10}{\sum_{i=0}^n{10^{L/10}}}
        $$

    Returns:
        Energetic summation of levels.
    """
    levels = np.asanyarray(levels)
    return 10.0 * np.log10((10.0 ** (levels / 10.0)).sum(axis=axis))


def dbmean(levels, axis=None) -> float:
    r"""Energetic average of levels.

    Args:
      levels: Sequence of levels.
      axis: Axis over which to perform the operation.
        $$
        L_{mean} = 10 \log_{10}{\frac{1}{n}\sum_{i=0}^n{10^{L/10}}}
        $$

    Returns:
        Energetic average of levels.
    """
    levels = np.asanyarray(levels)
    return 10.0 * np.log10((10.0 ** (levels / 10.0)).mean(axis=axis))


def dbadd(a: float | np.ndarray, b: float | np.ndarray) -> float:
    r"""Energetic addition.

    Energetically adds b to a.

      $$
      L_{a+b} = 10 \log_{10}{10^{L_b/10}+10^{L_a/10}}
      $$

    Args:
      a: Single level or sequence of levels.
      b: Single level or sequence of levels.


    Returns:
        Energetic addition of a and b.
    """
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    return 10.0 * np.log10(10.0 ** (a / 10.0) + 10.0 ** (b / 10.0))


def dbsub(a: float | np.ndarray, b: float | np.ndarray) -> float:
    r"""Energetic subtraction.

    Energitally subtract b from a.

    $$
    L_{a-b} = 10 \log_{10}{10^{L_a/10}-10^{L_b/10}}
    $$

    Args:
      a: Single level or sequence of levels.
      b: Single level or sequence of levels.

    Returns:
        Energetic subtraction of a and b.
    """
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    return 10.0 * np.log10(10.0 ** (a / 10.0) - 10.0 ** (b / 10.0))


def dbmul(levels, f, axis=None) -> np.ndarray:
    r"""Energetically add `levels` `f` times.

    $$
    L_{sum} = 10 \log_{10}{\sum_{i=0}^n{10^{L/10} \cdot f}}
    $$

    Args:
      levels: Sequence of levels.
      f: Multiplication factor `f`.
      axis: Axis over which to perform the operation.

    Returns:
        Resulting levels after energetic addition.
    """
    levels = np.asanyarray(levels)
    return 10.0 * np.log10((10.0 ** (levels / 10.0) * f).sum(axis=axis))


def dbdiv(levels: float | np.ndarray, f: float, axis: int | None = None) -> np.ndarray:
    r"""Energetically divide `levels` `f` times.

    $$
    L_{sum} = 10 \log_{10}{\sum_{i=0}^n{10^{L/10} / f}}
    $$

    Args:
      levels: Sequence of levels.
      f: Divider `f`.
      axis: Axis over which to perform the operation.

    Returns:
        Resulting levels after energetic division.
    """
    levels = np.asanyarray(levels)
    return 10.0 * np.log10((10.0 ** (levels / 10.0) / f).sum(axis=axis))


__all__ = ["dbsum", "dbmean", "dbadd", "dbsub", "dbmul", "dbdiv"]
