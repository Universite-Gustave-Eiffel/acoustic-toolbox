import numpy as np

from acoustic_toolbox.standards.iec_61672_1_2013 import time_weighted_level, time_averaged_level


def signal_fs():
    fs = 4000.0
    f = 400.0
    duration = 3.0
    samples = int(duration * fs)
    t = np.arange(samples) / fs
    x = np.sin(2.0 * np.pi * f * t)
    return x, fs


def test_fast_level():
    r"""Test whether integration with time-constant FAST gives the correct level.

    Note that the reference sound pressure is used.

    In this test the amplitude of the sine is 1, which means the mean squared $MS$ is 0.5
    With a reference pressure $p_r$ of 2.0e-5 the level should be 91 decibel

    $$
    L = 10 \cdot \log_{10}{\left(\frac{MS}{p_r^2} \right)}
    $$

    $$
    L = 10 \cdot \log_{10}{\left(\frac{0.5}{(2e-5)^2} \right)} = 91
    $$
    """
    x, fs = signal_fs()

    times, levels = time_weighted_level(x, fs, time_mode="fast")
    assert abs(levels[-1] - 91) < 0.05

    x *= 4.0
    times, levels = time_weighted_level(x, fs, time_mode="fast")
    assert abs(levels[-1] - 103) < 0.05

def test_big_signal_fast_level():
    """Test whether integration with time-constant FAST gives the correct level for a big signal."""
    fs = 48000.0
    f = 400.0
    duration = 350
    samples = int(duration * fs)
    t = np.arange(samples) / fs
    x = np.sin(2.0 * np.pi * f * t)

    times, levels = time_weighted_level(x, fs, time_mode="fast")
    assert abs(levels[-1] - 91) < 0.05

def test_slow_level():
    """Test whether integration with time-constant SLOW gives the correct level."""
    x, fs = signal_fs()
    x = np.concatenate([x, x]) # Make signal longer for SLOW weighting

    times, levels = time_weighted_level(x, fs, time_mode="slow")
    assert abs(levels[-1] - 91) < 0.05

    x *= 4.0
    times, levels = time_weighted_level(x, fs, time_mode="slow")
    assert abs(levels[-1] - 103) < 0.05


def test_time_averaged_level():
    """Test whether time-averaged level gives the correct level."""
    x, fs = signal_fs()

    # Test with integration time of 1.0 second
    integration_time = 1.0
    times, levels = time_averaged_level(x, fs, integration_time)

    # Expected level for sine wave with amp 1 is approx 91 dB relative to 2e-5
    # 10 * log10(0.5 / (2e-5)**2) = 90.969...
    assert abs(levels[-1] - 91) < 0.05

    # Test with scaling
    x_scaled = x * 4.0
    times, levels = time_averaged_level(x_scaled, fs, integration_time)
    # 20 * log10(4) = 12.04 dB increase
    assert abs(levels[-1] - 103) < 0.05

    # Test with fractional integration time to exercise the variable chunking
    # fs = 4000. integration_time 0.1001s -> 400.4 samples.
    integration_time = 0.1001
    times, levels = time_averaged_level(x, fs, integration_time)

    # Should still be approximately correct
    assert abs(np.mean(levels) - 91) < 0.5


def test_frequency_weighting_A():
    """Test A-weighting frequency response."""
    from acoustic_toolbox.standards.iec_61672_1_2013 import (
        frequency_weighting,
        NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
        WEIGHTING_A,
    )

    fs = 48000
    duration = 0.5
    t = np.arange(int(fs * duration)) / fs

    # Tolerances (dB)
    tolerance = 0.6

    for f_nom, expected_db in zip(NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, WEIGHTING_A):
        if f_nom < 10:
            continue

        if f_nom > 8000:
            # Due to using bilinear transform, the accuracy degrades near Nyquist so let's not test those
            # other techniques (e.g., improved matched z-transform) would be needed for higher accuracy
            continue

        # Generate sine wave at this frequency
        signal = np.sin(2 * np.pi * f_nom * t)

        # Apply weighting
        weighted = frequency_weighting(signal, fs, weighting='A')

        # Calculate RMS levels - use last part to avoid transient
        valid_idx = int(fs * 0.2)
        valid_part = weighted[valid_idx:]
        rms_in = np.sqrt(np.mean(signal[valid_idx:]**2))
        rms_out = np.sqrt(np.mean(valid_part**2))

        db_in = 20 * np.log10(rms_in)
        db_out = 20 * np.log10(rms_out)

        attenuation = db_out - db_in

        # Check if attenuation matches expected A-weighting
        assert abs(attenuation - expected_db) < tolerance, \
            f"Frequency {f_nom} Hz: Expected {expected_db} dB, got {attenuation:.2f} dB"


def test_frequency_weighting_C():
    """Test C-weighting frequency response."""
    from acoustic_toolbox.standards.iec_61672_1_2013 import (
        frequency_weighting,
        NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
        WEIGHTING_C,
    )

    fs = 48000
    duration = 0.5
    t = np.arange(int(fs * duration)) / fs
    # Tolerances (dB)
    tolerance = 0.6

    for f_nom, expected_db in zip(NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, WEIGHTING_C):
        if f_nom < 10:
            continue

        if f_nom > 8000:
            # Due to using bilinear transform, the accuracy degrades near Nyquist so let's not test those
            # other techniques (e.g., improved matched z-transform?) would be needed for higher accuracy
            continue

        # Generate sine wave at this frequency
        signal = np.sin(2 * np.pi * f_nom * t)

        # Apply weighting
        weighted = frequency_weighting(signal, fs, weighting='C')

        # Calculate RMS levels - use last part to avoid transient
        valid_idx = int(fs * 0.2)
        valid_part = weighted[valid_idx:]
        rms_in = np.sqrt(np.mean(signal[valid_idx:]**2))
        rms_out = np.sqrt(np.mean(valid_part**2))

        db_in = 20 * np.log10(rms_in)
        db_out = 20 * np.log10(rms_out)

        attenuation = db_out - db_in

        # Check if attenuation matches expected C-weighting
        assert abs(attenuation - expected_db) < tolerance, \
            f"Frequency {f_nom} Hz: Expected {expected_db} dB, got {attenuation:.2f} dB"


def test_frequency_weighting_Z():
    """Test Z-weighting frequency response (should be flat)."""
    from acoustic_toolbox.standards.iec_61672_1_2013 import frequency_weighting

    fs = 48000
    signal = np.random.randn(fs) # 1 second of noise

    weighted = frequency_weighting(signal, fs, weighting='Z')

    # Z-weighting should return the signal as-is
    np.testing.assert_array_equal(signal, weighted)


def test_frequency_weighting_zero_phase():
    """Test zero-phase filtering."""
    from acoustic_toolbox.standards.iec_61672_1_2013 import frequency_weighting

    fs = 1000
    # Create an impulse
    signal = np.zeros(1000)
    signal[500] = 1.0

    # Apply A-weighting with zero phase
    weighted = frequency_weighting(signal, fs, weighting='A', zero_phase=True)

    # Check for symmetry around the impulse to confirm zero phase behavior
    # The impulse response should be symmetric
    max_idx = np.argmax(np.abs(weighted))
    assert max_idx == 500, "Peak should remain at center"

    # Check symmetry of a small window around peak
    window = 10
    left = weighted[max_idx-window:max_idx]
    right = weighted[max_idx+1:max_idx+window+1][::-1]

    np.testing.assert_allclose(left, right, atol=1e-10)
