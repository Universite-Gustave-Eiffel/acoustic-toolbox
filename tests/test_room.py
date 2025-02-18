import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import pytest

from acoustic_toolbox.room import (
    mean_alpha,
    nrc,
    t60_sabine,
    t60_eyring,
    t60_millington,
    t60_fitzroy,
    t60_arau,
    t60_impulse,
    c50_from_file,
    c80_from_file,
)
from acoustic_toolbox.bands import octave, third

from tests.get_data_path import data_path


class TestRoom:
    """Test :mod:`acoustic_toolbox.room`"""

    surfaces = np.array([240, 600, 500])
    alpha = np.array([0.1, 0.25, 0.45])
    alpha_bands = np.array(
        [[0.1, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25], [0.45, 0.45, 0.45, 0.45]]
    )
    volume = 3000

    def test_t60_sabine(self):
        calculated = t60_sabine(self.surfaces, self.alpha, self.volume)
        real = 1.211382149
        assert_almost_equal(calculated, real)

    def test_t60_sabine_bands(self):
        calculated = t60_sabine(self.surfaces, self.alpha_bands, self.volume)
        real = np.array([1.211382149, 1.211382149, 1.211382149, 1.211382149])
        assert_array_almost_equal(calculated, real)

    def test_t60_eyring(self):
        calculated = t60_eyring(self.surfaces, self.alpha, self.volume)
        real = 1.020427763
        assert_almost_equal(calculated, real)

    def test_t60_eyring_bands(self):
        calculated = t60_eyring(self.surfaces, self.alpha_bands, self.volume)
        real = np.array([1.020427763, 1.020427763, 1.020427763, 1.020427763])
        assert_array_almost_equal(calculated, real)

    def test_t60_millington(self):
        calculated = t60_millington(self.surfaces, self.alpha, self.volume)
        real = 1.020427763
        assert_almost_equal(calculated, real)

    def test_t60_millington_bands(self):
        calculated = t60_millington(self.surfaces, self.alpha_bands, self.volume)
        real = np.array([1.020427763, 1.020427763, 1.020427763, 1.020427763])
        assert_array_almost_equal(calculated, real)

    def test_t60_fitzroy(self):
        surfaces_fitzroy = np.array([240, 240, 600, 600, 500, 500])
        alpha_fitzroy = np.array([0.1, 0.1, 0.25, 0.25, 0.45, 0.45])
        calculated = t60_fitzroy(surfaces_fitzroy, alpha_fitzroy, self.volume)
        real = 0.699854185
        assert_almost_equal(calculated, real)

    def test_t60_fitzroy_bands(self):
        surfaces_fitzroy = np.array([240, 240, 600, 600, 500, 500])
        alpha_bands_f = np.array(
            [
                [0.1, 0.1, 0.25, 0.25, 0.45, 0.45],
                [0.1, 0.1, 0.25, 0.25, 0.45, 0.45],
                [0.1, 0.1, 0.25, 0.25, 0.45, 0.45],
            ]
        )
        calculated = t60_fitzroy(surfaces_fitzroy, alpha_bands_f, self.volume)
        real = np.array([0.699854185, 0.699854185, 0.699854185])
        assert_array_almost_equal(calculated, real)

    def test_t60_arau(self):
        Sx = self.surfaces[0]
        Sy = self.surfaces[1]
        Sz = self.surfaces[2]
        calculated = t60_arau(Sx, Sy, Sz, self.alpha, self.volume)
        real = 1.142442931
        assert_almost_equal(calculated, real)

    def test_t60_arau_bands(self):
        Sx = self.surfaces[0]
        Sy = self.surfaces[1]
        Sz = self.surfaces[2]
        calculated = t60_arau(Sx, Sy, Sz, self.alpha_bands, self.volume)
        real = np.array([1.142442931, 1.142442931, 1.142442931, 1.142442931])
        assert_array_almost_equal(calculated, real)

    def test_mean_alpha_float(self):
        alpha = 0.1
        surface = 10
        calculated = mean_alpha(alpha, surface)
        real = 0.1
        assert_almost_equal(calculated, real)

    def test_mean_alpha_1d(self):
        alpha = np.array([0.1, 0.2, 0.3])
        surfaces = np.array([20, 30, 40])
        calculated = mean_alpha(alpha, surfaces)
        real = 0.222222222
        assert_almost_equal(calculated, real)

    def test_mean_alpha_bands(self):
        alpha = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
        surfaces = np.array([20, 30, 40])
        calculated = mean_alpha(alpha, surfaces)
        real = np.array([0.222222222, 0.222222222, 0.222222222])
        assert_array_almost_equal(calculated, real)

    def test_nrc_1d(self):
        alpha = np.array([0.1, 0.25, 0.5, 0.9])
        calculated = nrc(alpha)
        real = 0.4375
        assert_almost_equal(calculated, real)

    def test_nrc_2d(self):
        alphas = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]])
        calculated = nrc(alphas)
        real = np.array([0.25, 0.55])
        assert_array_almost_equal(calculated, real)

    @pytest.mark.parametrize(
        "file_name, bands, rt, expected",
        [
            (
                data_path() / "ir_sportscentre_omni.wav",
                octave(125, 4000),
                "t30",
                np.array([7.388, 8.472, 6.795, 6.518, 4.797, 4.089]),
            ),
            (
                data_path() / "ir_sportscentre_omni.wav",
                octave(125, 4000),
                "edt",
                np.array([4.667, 5.942, 6.007, 5.941, 5.038, 3.735]),
            ),
            (
                data_path() / "living_room_1.wav",
                octave(63, 8000),
                "t30",
                np.array([0.274, 0.365, 0.303, 0.259, 0.227, 0.211, 0.204, 0.181]),
            ),
            (
                data_path() / "living_room_1.wav",
                octave(63, 8000),
                "t20",
                np.array([0.300, 0.365, 0.151, 0.156, 0.102, 0.076, 0.146, 0.152]),
            ),
            (
                data_path() / "living_room_1.wav",
                octave(63, 8000),
                "t10",
                np.array([0.185, 0.061, 0.109, 0.024, 0.039, 0.023, 0.105, 0.071]),
            ),
            (
                data_path() / "living_room_1.wav",
                octave(63, 8000),
                "edt",
                np.array([0.267, 0.159, 0.080, 0.037, 0.021, 0.010, 0.022, 0.020]),
            ),
            (
                data_path() / "living_room_1.wav",
                third(100, 5000),
                "t30",
                np.array(
                    [
                        0.318,
                        0.340,
                        0.259,
                        0.311,
                        0.267,
                        0.376,
                        0.342,
                        0.268,
                        0.212,
                        0.246,
                        0.211,
                        0.232,
                        0.192,
                        0.231,
                        0.252,
                        0.202,
                        0.184,
                        0.216,
                    ]
                ),
            ),
            (
                data_path() / "living_room_1.wav",
                third(100, 5000),
                "t20",
                np.array(
                    [
                        0.202,
                        0.383,
                        0.189,
                        0.173,
                        0.141,
                        0.208,
                        0.323,
                        0.221,
                        0.102,
                        0.110,
                        0.081,
                        0.128,
                        0.072,
                        0.074,
                        0.087,
                        0.129,
                        0.137,
                        0.171,
                    ]
                ),
            ),
            (
                data_path() / "living_room_1.wav",
                third(100, 5000),
                "t10",
                np.array(
                    [
                        0.110,
                        0.104,
                        0.132,
                        0.166,
                        0.135,
                        0.040,
                        0.119,
                        0.223,
                        0.025,
                        0.023,
                        0.047,
                        0.050,
                        0.010,
                        0.017,
                        0.039,
                        0.084,
                        0.154,
                        0.093,
                    ]
                ),
            ),
            (
                data_path() / "living_room_1.wav",
                third(100, 5000),
                "edt",
                np.array(
                    [
                        0.354,
                        0.328,
                        0.284,
                        0.210,
                        0.132,
                        0.116,
                        0.085,
                        0.114,
                        0.064,
                        0.045,
                        0.047,
                        0.047,
                        0.024,
                        0.017,
                        0.016,
                        0.022,
                        0.020,
                        0.036,
                    ]
                ),
            ),
        ],
    )
    def test_t60_impulse(self, file_name, bands, rt, expected):
        calculated = t60_impulse(file_name, bands, rt)
        assert_array_almost_equal(calculated, expected, decimal=0)

    @pytest.mark.parametrize(
        "file_name, bands, expected",
        [
            (
                data_path() / "living_room_1.wav",
                octave(63, 8000),
                np.array([8.0, 18.0, 23.0, 26.0, 30.0, 31.0, 27.0, 29.0]),
            ),
            (
                data_path() / "living_room_1.wav",
                third(100, 5000),
                np.array(
                    [
                        3.0,
                        6.0,
                        7.0,
                        13.0,
                        18.0,
                        23.0,
                        20.0,
                        19.0,
                        28.0,
                        30.0,
                        30.0,
                        27.0,
                        32.0,
                        31.0,
                        30.0,
                        28.0,
                        29.0,
                        25.0,
                    ]
                ),
            ),
        ],
    )
    def test_c50_from_file(self, file_name, bands, expected):
        calculated = c50_from_file(file_name, bands)
        assert_array_almost_equal(calculated, expected, decimal=0)

    # TODO: Test clarity, and/or test c50/c80 with bands=None
    @pytest.mark.parametrize(
        "file_name, bands, expected",
        [
            (
                data_path() / "living_room_1.wav",
                octave(63, 8000),
                np.array(
                    [18.542, 23.077, 27.015, 31.743, 35.469, 36.836, 33.463, 36.062]
                ),
            ),
            (
                data_path() / "living_room_1.wav",
                third(100, 5000),
                np.array(
                    [
                        17.0,
                        14.0,
                        17.0,
                        24.0,
                        26.0,
                        27.0,
                        22.0,
                        26.0,
                        34.0,
                        35.0,
                        34.0,
                        32.0,
                        38.0,
                        38.0,
                        34.0,
                        34.0,
                        35.0,
                        32.0,
                    ]
                ),
            ),
        ],
    )
    def test_c80_from_file(self, file_name, bands, expected):
        calculated = c80_from_file(file_name, bands)
        assert_array_almost_equal(calculated, expected, decimal=0)
