from enum import Enum
import math
from signal import signal

from numpy.ma.core import subtract


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


def sig_sub(signal_1, signal_2) -> Signal:

    # Determine the maximum length based on the longest signal
    # Create a list to hold the subtracted amplitudes

    subtracted_amplitudes = [ y- x for x , y in zip(signal_1.amplitudes , signal_2.amplitudes)]

    return Signal(False , Signal_type.Time,subtracted_amplitudes,[i for i in range(len(subtracted_amplitudes))])


def sig_add(signal_1, signal_2) -> Signal:
    # Determine the maximum length based on the longest signal
    max_length = max(signal_1.indices[-1], signal_2.indices[-1]) + 1

    # Create a list to hold the summed amplitudes
    added_amplitudes = [0 for i in range(max_length)]

    # Add amplitudes from signal_1
    if signal_1:
        for i in range(len(signal_1.amplitudes)):
            added_amplitudes[i] += signal_1.amplitudes[i]

    # Add amplitudes from signal_2
    if signal_2:
        for i in range(len(signal_2.amplitudes)):
            added_amplitudes[i] += signal_2.amplitudes[i]

    return Signal(False , Signal_type.Time,added_amplitudes,[i for i in range(len(added_amplitudes))])


def sig_mul(signal, value) -> Signal:
    if signal is None:
        return None

    # Initialize the multiplied amplitudes list
    multiplied_amplitudes = [0 for i in range(len(signal.amplitudes))]

    # Multiply each amplitude by the given value
    for i in range(len(signal.amplitudes)):
        multiplied_amplitudes[i] = signal.amplitudes[i] * value

    return Signal(False , Signal_type.Time,multiplied_amplitudes,[i for i in range(len(multiplied_amplitudes))])


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


def SignalSamplesAreEqual(file, indices, samples):
    sig = read_file(file)
    exiected_indices = sig.indices
    expected_samples = sig.amplitudes

    if len(expected_samples) != len(samples):
        return (
            "Test case failed, your signal have different length from the expected one"
        )
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            return "Test case failed, your signal have different values from the expected one"
    return "Test case passed successfully"

def quantize(noOfLevels, samples):
    minValue =min(samples)
    maxValue =max(samples)
    #width of each quantization interval
    delta = (maxValue - minValue) / noOfLevels
    interval_index = []
    quantizedValues = []
    quantizationErrors = []
    encodedLevels = []

    for sample in samples:
        quantized_level = int((sample - minValue) / delta) #shifting sample down by the minimum value
        quantized_level = min(quantized_level, noOfLevels - 1)  # Avoid overflow
        quantized_value = minValue + quantized_level * delta + delta / 2 #reduce quantization error by mapping it to midpoint