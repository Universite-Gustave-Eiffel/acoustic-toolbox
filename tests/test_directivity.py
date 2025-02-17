from acoustic_toolbox.directivity import figure_eight
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
