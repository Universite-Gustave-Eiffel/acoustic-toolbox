"""Doppler shift module."""

SOUNDSPEED = 343.0
"""Speed of sound"""


def velocity_from_doppler_shift(f1, f2, c=SOUNDSPEED) -> float:
    r"""Calculate velocity based on measured frequency shifts due to Doppler shift.

    $$
    v = c \cdot \left( \frac{f_2 - f_1}{f_2 + f_1} \right)
    $$

    The assumption is made that the velocity is constant between the observation times.

    Args:
      f1: Lower frequency $f_1$.
      f2: Upper frequency $f_2$.
      c: Speed of sound $c$.

    Returns:
        Calculated velocity.
    """
    return c * (f2 - f1) / (f2 + f1)


def frequency_shift(
    frequency, velocity_source, velocity_receiver, soundspeed=SOUNDSPEED
) -> float:
    r"""Frequency shift due to Doppler effect.

    Args:
      frequency: Emitted frequency $f$.
      velocity_source: Velocity of source $v_s$.
        Positive if the source is moving away from the receiver (and negative in the other direction).
      velocity_receiver: Velocity of receiver $v_r$.
        Positive if the receiver is moving towards the source (and negative in the other direction);
      soundspeed: Speed of sound $c$.
        $$
        f = \frac{c + v_r}{c + v_s} f_0
        $$

    Returns:
        Frequency after shift.
    """
    return (soundspeed + velocity_receiver) / (soundspeed + velocity_source) * frequency
