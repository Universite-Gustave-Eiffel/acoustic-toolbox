r"""The directivity module provides tools to work with directivity.

The following conventions are used within this module:

* The inclination angle $\theta$ has a range $[0, \pi]$.
* The azimuth angle $\phi$ has a range $[0 , 2 \pi]$.

Functions:
    cardioid: Generate a cardioid pattern.
    figure_eight: Generate a figure-of-eight pattern.
    spherical_harmonic: Calculate spherical harmonic of order `m` and degree `n`.
    spherical_to_cartesian: Convert spherical coordinates to cartesian coordinates.
    cartesian_to_spherical: Convert cartesian coordinates to spherical coordinates.

Classes:
    Directivity: Abstract class for directivity.
    Omni: Class for omni-directional directivity.
    Cardioid: Class for cardioid directivity.
    FigureEight: Class for figure-of-eight directivity.
    SphericalHarmonic: Class for spherical harmonic directivity.
    Custom: Class for custom directivity.
"""

import abc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np
from scipy.interpolate import interp2d as interpolate
from scipy.special import sph_harm  # pylint: disable=no-name-in-module


def cardioid(theta, a=1.0, k=1.0):
    r"""A cardioid pattern.

    Args:
      a: a
      k: k
      theta: angle $\theta$

    Returns:
      Cardioid pattern.
    """
    return np.abs(a + a * np.cos(k * theta))


def figure_eight(theta, phi=0.0):
    r"""A figure-of-eight pattern.

    Args:
      theta: angle $\theta$
      phi: angle $\phi$

    Returns:
      Figure-of-eight pattern.
    """
    del phi
    # return spherical_harmonic(theta, phi, m=0, n=1)
    return np.abs(np.cos(theta))


def spherical_harmonic(theta, phi, m=0, n=0):
    """Spherical harmonic of order `m` and degree `n`.

    Note:
      The degree `n` is often denoted `l`.

    See Also:
      [`scipy.special.sph_harm`][scipy.special.sph_harm]
    """
    # TODO: `scipy.special.sph_harm` is deprecated since v1.15. Should use `sph_harm_y` instead.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html
    # TODO: `scipy` expects `m` and `n` to be integers. Will need to adapt typehints
    return sph_harm(m, n, phi, theta).real


def spherical_to_cartesian(r, theta, phi):
    r"""Convert spherical coordinates to cartesian coordinates.

    Args:
      r: norm
      theta: angle $\theta$
      phi: angle $\phi$

    Returns:
       x: $x = r \sin{\theta}\cos{\phi}$
       y: $y = r \sin{\theta}\sin{\phi}$
       z: $z = r \cos{\theta}$
    """
    r = np.asanyarray(r)
    theta = np.asanyarray(theta)
    phi = np.asanyarray(phi)
    return (
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    )


def cartesian_to_spherical(x, y, z):
    r"""Convert cartesian coordinates to spherical coordinates.

    Args:
      x: x
      y: y
      z: z

    Returns:
      r: $r = \sqrt{\left( x^2 + y^2 + z^2 \right)}$
      theta: $\theta = \arccos{\frac{z}{r}}$
      phi: $\phi = \arccos{\frac{y}{x}}$
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    z = np.asanyarray(z)
    r = np.linalg.norm(np.vstack((x, y, z)), axis=0)
    return r, np.arccos(z / r), np.arctan(y / x)


class Directivity:
    """Abstract directivity class.

    This class defines several methods to be implemented by subclasses.
    """

    def __init__(self, rotation=None):
        """Rotation of the directivity pattern.

        Args:
          rotation: Rotation of the directivity pattern.
        """
        self.rotation = (
            rotation if rotation else np.array([1.0, 0.0, 0.0])
        )  # X, Y, Z rotation

    @abc.abstractmethod
    def _directivity(self, theta, phi):
        r"""This function should return the directivity as function of $\theta$ and $\phi$."""

    def _undo_rotation(self, theta, phi):
        """Undo rotation."""

    def using_spherical(self, theta, phi, r=None, include_rotation: bool = True):
        r"""Return the directivity for given spherical coordinates.

        Args:
          theta: angle $\theta$
          phi: angle $\phi$
          r: norm (optional)
          include_rotation: Apply the rotation to the directivity.

        Returns:
          Directivity.

        Todo:
          Correct for rotation!!!!
        """
        # TODO: Correct for rotation!!!! use 'r' if needed in rotation logic
        return self._directivity(theta, phi)

    def using_cartesian(self, x, y, z, include_rotation: bool = True):
        """Return the directivity for given cartesian coordinates.

        Args:
          x: x
          y: y
          z: z
          include_rotation: Apply the rotation to the directivity.

        Returns:
          Directivity.

        Todo:
          Correct for rotation!!!!
        """
        r, theta, phi = cartesian_to_spherical(x, y, z)
        return self.using_spherical(theta, phi, r, include_rotation)

    def plot(
        self, filename: str | None = None, include_rotation: bool = True
    ) -> Figure:
        """Directivity plot. Plot to ``filename`` when given.

        Args:
          filename: Filename
          include_rotation: Apply the rotation to the directivity.
            By default the rotation is applied in this figure.

        Returns:
          Figure.

        Todo:
          filename
        """
        raise NotImplementedError(
            "`Directivity.plot` needs to be re-implemented, due to underlying issues from python-acoustics."
        )
        # I'm not sure how the example notebook ever ran before...
        # The method needs to be re-implemented to match the `plot` function.
        del filename
        return plot(self, include_rotation)


class Omni(Directivity):
    """Class to work with omni-directional directivity."""

    def _directivity(self, theta, phi):
        """Directivity."""
        return np.ones_like(theta)


class Cardioid(Directivity):
    """Cardioid directivity."""

    def _directivity(self, theta, phi):
        """Directivity."""
        return cardioid(theta)


class FigureEight(Directivity):
    """Directivity of a figure of eight."""

    def _directivity(self, theta, phi):
        """Directivity."""
        return figure_eight(theta, phi)


class SphericalHarmonic(Directivity):
    """Directivity of a spherical harmonic of degree `n` and order `m`."""

    def __init__(self, rotation=None, m=None, n=None):
        """Constructor."""
        super().__init__(rotation=rotation)
        self.m = m
        """Order `m`."""
        self.n = n
        """Degree `n`."""

    def _directivity(self, theta, phi):
        """Directivity."""
        return spherical_harmonic(theta, phi, self.m, self.n)


class Custom(Directivity):
    """A class to work with directivity."""

    def __init__(self, theta=None, phi=None, r=None):
        """Constructor."""
        self.theta = theta
        """Latitude. 1-D array."""

        self.phi = phi
        """Longitude. 1-D array."""
        self.r = r
        """Magnitude or radius. 2-D array."""

    def _directivity(self, theta, phi):
        """Custom directivity.

        Interpolate the directivity given longitude and latitude vectors.
        """
        f = interpolate(self.theta, self.phi, self.r)

        return f(theta, phi)


def plot(d: Directivity, sphere: bool = False) -> Figure:
    """Plot directivity `d`.

    Args:
      d: Directivity
      sphere: Plot on a sphere.

    Returns:
      Figure
    """
    theta, phi = np.meshgrid(
        np.linspace(0.0, np.pi, 50), np.linspace(0.0, +2.0 * np.pi, 50)
    )

    # Directivity strength. Real-valued. Can be positive and negative.
    dr = d.using_spherical(theta, phi)

    if sphere:
        x, y, z = spherical_to_cartesian(1.0, theta, phi)

    else:
        x, y, z = spherical_to_cartesian(np.abs(dr), theta, phi)
    # R, theta, phi = cartesian_to_spherical(x, y, z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # p = ax.plot_surface(x, y, z, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)

    norm = Normalize()
    norm.autoscale(dr)
    colors = cm.jet(norm(dr))
    m = cm.ScalarMappable(cmap=cm.jet, norm=norm)
    m.set_array(dr)
    ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, linewidth=0)
    plt.colorbar(m, ax=ax)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    return fig
