from enum import Enum
import math


class Signal_type(Enum):
    Time = 0
    Freq = 1


class Signal:
    periodic: bool
    signal_type: Signal_type
    amplitudes: []
    indices: []

    def __init__(
        self,
        periodic: bool = False,
        signal_type: Signal_type = Signal_type.Time,
        amps=[],
        indices=[],
    ):
        self.periodic = periodic
        self.signal_type = signal_type
        self.amplitudes = amps
        self.indices = indices


def generate_signal(
    sin_flag: bool,
    periodic: bool,
    signal_type: Signal_type,
    amp,
    sampling_freq,
    analog_freq,
    phase_shift,
) -> Signal:
    # TODO: Handle Division by 0
    w = 2 * math.pi * (analog_freq / sampling_freq)
    if sin_flag:
        s = [amp * math.sin(w * i + phase_shift) for i in range(0, sampling_freq)]
    else:
        s = [amp * math.cos(w * i + phase_shift) for i in range(0, sampling_freq)]
    return Signal(periodic, signal_type, s, [i for i in range(0, sampling_freq)])
    # return s, [i for i in range(0, sampling_freq)]


def read_file(uploaded_file) -> Signal:
    file_content = uploaded_file.read().decode("utf-8").splitlines()

    timeFlag = bool(file_content[0])  # First line
    periodic = bool(file_content[1])  # Second line
    nOfSamples = int(file_content[2])  # Third line

    indices = []
    amplitudes = []
    for line in file_content[3 : 3 + nOfSamples]:
        values = line.strip().split(" ")
        indices.append(int(values[0]))
        amplitudes.append(float(values[1]))

    return Signal(
        periodic,
        Signal_type.Time if timeFlag else Signal_type.Freq,
        amplitudes,
        indices,
    )


def sig_add(signal_1, signal_2) -> Signal:
    return signal_1


def sig_sub(signal_1, signal_2) -> Signal:
    return signal_1


def sig_mul(signal, value) -> Signal:
    return signal


def sig_norm(signal, _range: bool) -> Signal:
    if not signal:
        return None
    # _range 0 -> [0,1] , 1 -> [-1,1]
    mx = max(signal.amplitudes)
    mn = min(signal.amplitudes)
    r = mx - mn
    if _range:
        signal.amplitudes = [(i - mn) / r for i in signal.amplitudes]
    else:
        signal.amplitudes = [(i - mn) / r * 2 - 1 for i in signal.amplitudes]
    return signal


def sig_square(signal) -> Signal:
    if not signal:
        return None
    for i in range(len(signal.amplitudes)):
        signal.amplitudes[i] *= signal.amplitudes[i]
    return signal


def sig_acc(signal) -> Signal:
    if not signal:
        return None
    for i in range(1, len(signal.amplitudes)):
        signal.amplitudes[i] += signal.amplitudes[i - 1]
    return signal
