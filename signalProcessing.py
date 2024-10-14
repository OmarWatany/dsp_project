from enum import Enum
import math


class Signal_type(Enum):
    Time = 0
    Freq = 1


def generate_signal(
    sin_flag: bool,
    periodic: bool,
    signal_type: Signal_type,
    amp,
    sampling_freq,
    analog_freq,
    phase_shift,
):
    # TODO: Handle Division by 0
    w = 2 * math.pi * (analog_freq / sampling_freq)
    if sin_flag:
        s = [amp * math.sin(w * i + phase_shift) for i in range(0, sampling_freq)]
    else:
        s = [amp * math.cos(w * i + phase_shift) for i in range(0, sampling_freq)]
    return s, [i for i in range(0, sampling_freq)]
