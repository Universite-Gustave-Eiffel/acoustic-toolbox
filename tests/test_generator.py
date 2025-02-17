import pytest
import numpy as np
from acoustic_toolbox.generator import noise
from acoustic_toolbox.signal import octaves

parameters = [
    ("white", +3.0, 0.0),
    ("pink", 0.0, -3.0),
    ("blue", +6.0, +3.0),
    ("brown", -3.0, -6.0),
    ("violet", +9.0, +6.0),
]
# color, power_change, power_density_change

ERROR = 1.0
"""Permitted error for power_change and power_density_change.
This value is relatively large for `violet` because of its steep
inclination. On other cases the error is mostly < 0.1
"""


@pytest.fixture(params=parameters)
def parameters(request):
    return request.param


@pytest.fixture
def color(parameters):
    return parameters[0]


@pytest.fixture
def power_change(parameters):
    return parameters[1]


@pytest.fixture
def power_density_change(parameters):
    return parameters[2]


@pytest.fixture(params=[48000 * 10, 48000 * 10 + 1])
def samples(request):
    return request.param


class TestNoise:
    """Test :func:`acoustic_toolbox.generator.noise`."""

    def test_length(self, color, samples):
        assert len(noise(samples, color)) == samples

    def test_power(self, color, samples, power_change):
        fs = 48000
        _, L = octaves(noise(samples, color), fs)
        change = np.diff(L).mean()
        assert np.abs(change - power_change) < ERROR

    def test_power_density(self, color, samples, power_density_change):
        fs = 48000
        _, L = octaves(noise(samples, color), fs, density=True)
        change = np.diff(L).mean()
        assert np.abs(change - power_density_change) < ERROR
