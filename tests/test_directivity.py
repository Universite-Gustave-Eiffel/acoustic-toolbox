from acoustic_toolbox.directivity import figure_eight, SphericalHarmonic
import numpy as np
import pytest


class TestDirectivity:
    """Test :mod:`acoustic_toolbox.directivity`"""

    @pytest.mark.parametrize(
        "given, expected, uncertainty",
        [
            (0.0, 1.0, 0.0),
            (1.0 / 2.0 * np.pi, 0.0, 0.0),
            (np.pi, +1.0, 0.0),
            (3.0 / 2.0 * np.pi, 0.0, 0.0),
            (2.0 * np.pi, +1.0, 0.0),
        ],
    )
    def test_figure_eight(self, given, expected, uncertainty):
        assert figure_eight(given) == pytest.approx(expected, uncertainty)


class TestSphericalHarmonic:
    """Test :class:`acoustic_toolbox.directivity.SphericalHarmonic`"""

    def test_init(self):
        """Test initialization."""
        m = 1
        n = 2
        sh = SphericalHarmonic(m=m, n=n)
        assert sh.m == m
        assert sh.n == n
        assert sh.rotation is not None

    @pytest.mark.parametrize(
        "m, n, theta, phi, expected",
        [
            # Y_0^0 = 0.5 * sqrt(1/pi)
            (0, 0, 0.0, 0.0, 0.5 * np.sqrt(1/np.pi)),
            (0, 0, np.pi/2, 0.0, 0.5 * np.sqrt(1/np.pi)),
            (0, 0, np.pi/2, np.pi, 0.5 * np.sqrt(1/np.pi)),

            # Y_1^0 = 0.5 * sqrt(3/pi) * cos(theta)
            (0, 1, 0.0, 0.0, 0.5 * np.sqrt(3/np.pi)),
            (0, 1, np.pi, 0.0, -0.5 * np.sqrt(3/np.pi)),
            (0, 1, np.pi/2, 0.0, 0.0),
        ]
    )
    def test_values(self, m, n, theta, phi, expected):
        """Test specific values."""
        sh = SphericalHarmonic(m=m, n=n)
        # using_spherical calls _directivity
        assert sh.using_spherical(theta, phi) == pytest.approx(expected)

    def test_shapes(self):
        """Test shapes of output."""
        sh = SphericalHarmonic(m=0, n=0)
        theta = np.linspace(0, np.pi, 10)
        phi = np.linspace(0, 2*np.pi, 10)

        # Test with arrays
        res = sh.using_spherical(theta, phi)
        assert res.shape == theta.shape
        assert np.allclose(res, 0.5 * np.sqrt(1/np.pi))

        # Test with meshgrid
        theta_mesh, phi_mesh = np.meshgrid(theta, phi)
        res_mesh = sh.using_spherical(theta_mesh, phi_mesh)
        assert res_mesh.shape == theta_mesh.shape
