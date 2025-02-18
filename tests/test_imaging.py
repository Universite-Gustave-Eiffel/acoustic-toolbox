import numpy as np
from acoustic_toolbox.bands import octave, third
from acoustic_toolbox.imaging import plot_octave, plot_third, plot_bands


class TestImaging:
    """Test :mod:`acoustic_toolbox.imaging`"""

    octaves = octave(16, 16000)
    thirds = third(63, 8000)
    tl_oct = np.array([3, 4, 5, 12, 15, 24, 28, 23, 35, 45, 55])
    tl_third = np.array(
        [0, 0, 0, 1, 1, 2, 3, 5, 8, 13, 21, 32, 41, 47, 46, 44, 58, 77, 61, 75, 56, 54]
    )
    title = "Title"
    label = "Label"

    def test_plot_octave(self):
        plot_octave(self.tl_oct, self.octaves)

    def test_plot_octave_kHz(self):
        plot_octave(
            self.tl_oct,
            self.octaves,
            kHz=True,
            xlabel=self.label,
            ylabel=self.label,
            title=self.title,
            separator=".",
        )

    def test_plot_third_octave(self):
        plot_third(self.tl_third, self.thirds, marker="s", separator=",")

    def test_plot_third_octave_kHz(self):
        plot_third(
            self.tl_third,
            self.thirds,
            marker="s",
            kHz=True,
            xlabel=self.label,
            ylabel=self.label,
            title=self.title,
        )

    def test_plot_band_oct(self):
        plot_bands(self.tl_oct, self.octaves, axes=None, band_type="octave")
