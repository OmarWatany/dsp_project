from typing import NewType
from enum import Enum
import streamlit as st
import math

from numpy.ma.core import indices

Signal: NewType = NewType("Signal", dict)
# layout
# {
#   "periodic": periodic,
#   "signal_type": sig_type,
#   idx: [sample,phase_shift]
#   ...
# }


class Signal_type(Enum):
    TIME = 0
    FREQ = 1


def signal(
    periodic=0, sig_type=Signal_type.TIME, indices=None, samples=None, phase_shifts=None
) -> Signal:
    d = {
        "periodic": periodic,
        "signal_type": sig_type,
    }
    for i in range(len(indices)):
        d[indices[i]] = [samples[i], phase_shifts[i] if phase_shifts else 0]
    return d


def signal_idx(sig: Signal):
    return [i for i in sig.keys() if isinstance(i, int | float)]


def signal_samples(sig: Signal):
    return [sig[i][0] for i in sig.keys() if isinstance(i, int | float)]


def signal_phase_shifts(sig: Signal):
    return [sig[i][1] for i in sig.keys() if isinstance(i, int | float)]


def generate_signal(
    sin_flag: bool,
    periodic: bool,
    amp,
    sampling_freq,
    analog_freq,
    phase_shift,
) -> Signal:
    # TODO: Handle Division by 0
    w = 2 * math.pi * (analog_freq / sampling_freq)

    def ang(i):
        return w * i + phase_shift

    s = [
        amp * (math.sin(ang(i)) if sin_flag else math.cos(ang(i)))
        for i in range(0, sampling_freq)
    ]

    return signal(periodic, Signal_type.TIME, [i for i in range(sampling_freq)], s)


def read_file(uploaded_file, bin_flag: bool = 0) -> Signal:
    file_content = uploaded_file.read().decode("utf-8").splitlines()

    freqDomain = int(file_content[0])  # First line
    periodic = int(file_content[1])  # Second line
    nOfSamples = int(file_content[2])  # Third line

    x = []
    y = []
    for line in file_content[3 : 3 + nOfSamples]:
        values = line.strip().split(" ")
        if freqDomain:
            x.append(float(values[0]))  # instead of freqs
            y.append(float(values[1]))  # angles
        else:
            x.append(int(values[0]) if not bin_flag else float(values[0]))
            y.append(float(values[1]))

    return signal(
        periodic,
        sig_type=Signal_type.FREQ if freqDomain else Signal_type.TIME,
        indices=[i for i in range(len(x))] if freqDomain else x,
        samples=x if freqDomain else y,
        phase_shifts=y if freqDomain else None,
    )


def sig_add_sub(add_f: bool, sig1: Signal, sig2: Signal):
    indices_1, indices_2 = signal_idx(sig1), signal_idx(sig2)

    newIndices = sorted(set(indices_1 + indices_2))
    new_sig: dict = {}

    # for each index as key do operation on value if exist or with 0
    if add_f:
        for i in newIndices:
            # sig 1 get indix i [0](samlple) of [0] if i not found so sample = 0
            new_sig[i] = sig1.get(i, [0])[0] + sig2.get(i, [0])[0]
    else:
        for i in newIndices:
            new_sig[i] = sig2.get(i, [0])[0] - sig1.get(i, [0])[0]

    return signal(
        False,
        Signal_type.TIME,
        indices=list(new_sig.keys()),
        samples=list(new_sig.values()),
    )


def sig_sub(signal_1: Signal, signal_2: Signal):
    return sig_add_sub(
        0,
        signal_1,
        signal_2,
    )


def sig_add(signal_1: Signal, signal_2: Signal):
    # Determine the maximum length based on the longest signal
    return sig_add_sub(
        1,
        signal_1,
        signal_2,
    )


def sig_mul(signal: Signal, value):
    if signal is None:
        return None

    # Multiply each amplitude by the given value
    for i in signal_idx(signal):
        signal[i][0] *= value

    return signal


def norm(x, mx, mn):
    r = mx - mn
    return (x - mn) / r


def sig_norm(signal: Signal, _range: bool):
    if not signal:
        return None
    # _range 0 -> [0,1] , 1 -> [-1,1]
    samples = signal_samples(signal)
    mx = max(samples)
    mn = min(samples)
    for i in signal_idx(signal):
        signal[i][0] = norm(signal[i][0], mx, mn) * (1 if _range else 2) - (
            0 if _range else 1
        )

    return signal


def sig_square(signal: Signal):
    if not signal:
        return None
    indices = signal_idx(signal)
    for i in indices:
        signal[i][0] *= signal[i][0]
    return signal


def sig_acc(signal: Signal):
    if not signal:
        return None
    indices = signal_idx(signal)
    for i in range(1, len(indices)):
        signal[indices[i]][0] += signal[indices[i - 1]][0]
    return signal


def sig_shift(sig: Signal, steps: int):
    return signal(
        periodic=sig["periodic"],
        sig_type=sig["signal_type"],
        indices=[i + steps for i in signal_idx(sig)],
        samples=signal_samples(sig),
        phase_shifts=signal_phase_shifts(sig),
    )


def sig_fold(sig: Signal) -> Signal:
    return signal(
        periodic=sig["periodic"],
        sig_type=sig["signal_type"],
        indices=signal_idx(sig),
        samples=[sig.get(-i, [0, 0])[0] for i in signal_idx(sig)],
        phase_shifts=signal_phase_shifts(sig),
    )


def quantize(signal: Signal = None, noOfLevels=0):
    samples = signal_samples(signal)
    # indices = signal_idx(signal)

    minValue = min(samples)
    maxValue = max(samples)
    # width of each quantization interval
    delta = (maxValue - minValue) / noOfLevels
    interval_index = []
    quantizedValues = []
    quantizationErrors = []
    encodedLevels = []

    for sample in samples:
        quantized_level = min(
            int(norm(sample, maxValue, minValue) * noOfLevels), noOfLevels - 1
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


def fourier_transform(check=0, sig: Signal = None, fs=0):
    idx = signal_idx(sig)

    N = len(idx)

    # DFT
    if check == 0:
        samples = signal_samples(sig)
        freq = []
        angle = []

        for k in range(N):
            real_part = 0
            imag_part = 0
            for n in range(N):
                exponent = (2 * math.pi * k * n) / N
                real_part += samples[n] * math.cos(exponent)
                imag_part -= samples[n] * math.sin(exponent)

            freq.append(math.sqrt(real_part**2 + imag_part**2))
            angle.append(math.atan2(imag_part, real_part))
        omega = (2 * math.pi) / (N / fs)
        newIndices = [omega * i for i in range(1, N + 1)]

        return signal(
            sig["periodic"],
            sig_type=Signal_type.FREQ,
            indices=newIndices,
            samples=freq,
            phase_shifts=angle,
        )

    elif check == 1:
        freq = signal_samples(sig)
        pha = signal_phase_shifts(sig)
        y = []
        for n in range(N):
            real_part = 0
            imag_part = 0
            for k in range(N):
                real_amplitude = freq[k] * math.cos(pha[k])
                imag_amplitude = freq[k] * math.sin(pha[k])

                exp = (2 * math.pi * k * n) / N
                real_part += real_amplitude * math.cos(exp) - imag_amplitude * math.sin(
                    exp
                )
                imag_part += real_amplitude * math.sin(exp) + imag_amplitude * math.cos(
                    exp
                )

            y.append(round((real_part + imag_part) / N))

        return signal(
            sig["periodic"],
            sig_type=Signal_type.TIME,
            indices=[i for i in range(N)],
            samples=y,
        )


def sig_dct(sig: Signal):
    amps = signal_samples(sig)
    N = len(amps)
    y = []

    for k in range(N):
        sum = 0
        for n in range(1, N + 1):
            sum += amps[n - 1] * math.cos(
                math.pi * (2 * (n - 1) - 1) * (2 * k - 1) / (4 * N)
            )

        y.append(math.sqrt(2 / N) * sum)

    return signal(
        sig["periodic"],
        sig_type=Signal_type.TIME,
        indices=[i for i in range(N)],
        samples=y,
    )


def compute_first_derivative(sig):
    amplitudes = signal_samples(sig)
    return [amplitudes[i] - amplitudes[i - 1] for i in range(1, len(amplitudes))]


def compute_second_derivative(sig):
    amplitudes = signal_samples(sig)
    return [
        amplitudes[i + 1] - 2 * amplitudes[i] + amplitudes[i - 1]
        for i in range(1, len(amplitudes) - 1)
    ]


def sig_avg(samples: [int or float]):
    sum = 0
    size = len(samples)
    for i in range(size):
        sum += samples[i]
    return sum / size


def sig_smoothe(sig, windowSize=0):
    samples = signal_samples(sig)
    averages = []

    for i in range(len(samples) - windowSize + 1):
        averages.append(
            sig_avg(samples[i : i + windowSize])
        )  # take slice(array) from samples from i to i+windowSize

    return signal(
        periodic=sig["periodic"],
        sig_type=sig["signal_type"],
        indices=[i for i in range(len(averages))],
        samples=averages,
    )


def sig_rm_dc(sig):
    samples = signal_samples(sig)
    size = len(samples)
    if sig["signal_type"] == Signal_type.TIME:
        avg = sig_avg(samples)
        return signal(
            periodic=sig["periodic"],
            sig_type=sig["signal_type"],
            indices=[i for i in range(size)],
            samples=[samples[i] - avg for i in range(size)],
        )

    elif sig["sig_type"] == Signal_type.FREQ:
        indices = signal_idx(sig)
        sig.pop(indices[0])
        return sig
    return None
