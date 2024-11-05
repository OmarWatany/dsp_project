from enum import Enum
import streamlit as st
import math


class Signal_type(Enum):
    Time = 0
    Freq = 1


class Signal:
    periodic: bool
    signal_type: Signal_type
    amplitudes: []
    indices: []
    angles: []

    def __init__(
        self,
        periodic: bool = False,
        signal_type: Signal_type = Signal_type.Time,
        amps=[],
        indices=[],
        angles=[],
    ):
        self.periodic = periodic
        self.signal_type = signal_type
        self.amplitudes = amps
        self.indices = indices
        self.angles = angles


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


def read_file(uploaded_file, bin_flag: bool = 0) -> Signal:
    file_content = uploaded_file.read().decode("utf-8").splitlines()

    periodic = int(file_content[0])  # Second line
    freqDomain = int(file_content[1])  # First line
    nOfSamples = int(file_content[2])  # Third line

    indices = []
    amplitudes = []
    for line in file_content[3 : 3 + nOfSamples]:
        values = line.strip().split(" ")
        if not freqDomain:
            indices.append(int(values[0]) if not bin_flag else float(values[0]))
            amplitudes.append(float(values[1]))
        else:
            amplitudes.append(float(values[0]))  # instead of freqs
            indices.append(float(values[1]))  # angles

    return Signal(
        periodic,
        Signal_type.Freq if freqDomain else Signal_type.Freq,
        amps=amplitudes,
        indices=[i for i in range(nOfSamples)] if freqDomain else indices,
        angles=indices if freqDomain else [],
    )


def sig_sub(signal_1, signal_2) -> Signal:
    # Determine the maximum length based on the longest signal
    # Create a list to hold the subtracted amplitudes

    subtracted_amplitudes = [
        y - x for x, y in zip(signal_1.amplitudes, signal_2.amplitudes)
    ]

    return Signal(
        False,
        Signal_type.Time,
        subtracted_amplitudes,
        [i for i in range(len(subtracted_amplitudes))],
    )


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

    return Signal(
        False,
        Signal_type.Time,
        added_amplitudes,
        [i for i in range(len(added_amplitudes))],
    )


def sig_mul(signal, value) -> Signal:
    if signal is None:
        return None

    # Initialize the multiplied amplitudes list
    multiplied_amplitudes = [0 for i in range(len(signal.amplitudes))]

    # Multiply each amplitude by the given value
    for i in range(len(signal.amplitudes)):
        multiplied_amplitudes[i] = signal.amplitudes[i] * value

    return Signal(
        False,
        Signal_type.Time,
        multiplied_amplitudes,
        [i for i in range(len(multiplied_amplitudes))],
    )


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


def quantize(signal: Signal = None, noOfLevels=0):
    minValue = min(signal.amplitudes)
    maxValue = max(signal.amplitudes)
    # width of each quantization interval
    delta = (maxValue - minValue) / noOfLevels
    interval_index = []
    quantizedValues = []
    quantizationErrors = []
    encodedLevels = []

    for sample in signal.amplitudes:
        quantized_level = min(
            int((sample - minValue) / delta), noOfLevels - 1
        )  # Avoid overflow
        quantized_value = (
            minValue + quantized_level * delta + delta / 2
        )  # reduce quantization error by mapping it to midpoint

        interval_index.append(quantized_level + 1)
        quantizedValues.append(round(quantized_value, 3))
        quantizationErrors.append(round(quantized_value - sample, 3))
        encodedLevels.append(
            f"{quantized_level:0{int(math.ceil(math.log2(noOfLevels)))}b}"
        )

    return interval_index, encodedLevels, quantizedValues, quantizationErrors


def fourier_transform(check=0, sig=None, fs=0):
    N = len(sig.amplitudes)

    # DFT
    if check == 0:
        freq = []
        angle = []

        for k in range(N):
            real_part = 0
            imag_part = 0
            for n in range(N):
                exponent = (2 * math.pi * k * n) / N
                real_part += sig.amplitudes[n] * math.cos(exponent)
                imag_part -= sig.amplitudes[n] * math.sin(exponent)

            freq.append(math.sqrt(real_part**2 + imag_part**2))
            angle.append(math.atan2(imag_part, real_part))
        omega = (2 * math.pi) / (N / fs)
        newIndices = [omega * i for i in range(1, N + 1)]
        sig.indices = newIndices
        sig.amplitudes = freq
        sig.angles = angle
        return sig

    elif check == 1:
        amplitudes = []
        for n in range(N):
            real_part = 0
            imag_part = 0
            for k in range(N):
                real_amplitude = sig.amplitudes[k] * math.cos(sig.angles[k])
                imag_amplitude = sig.amplitudes[k] * math.sin(sig.angles[k])

                exponent = (2 * math.pi * k * n) / N
                real_part += real_amplitude * math.cos(
                    exponent
                ) - imag_amplitude * math.sin(exponent)
                imag_part += real_amplitude * math.sin(
                    exponent
                ) + imag_amplitude * math.cos(exponent)

            amplitudes.append(round((real_part + imag_part) / N))
        # st.write(sig)
        # st.write(amplitudes)
        sig.signal_type = Signal_type.Time
        sig.amplitudes = amplitudes
        sig.indices = [i for i in range(N)]
        return sig
