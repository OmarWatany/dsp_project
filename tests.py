from numpy.ma.core import indices

import signalProcessing as sp
import streamlit as st
import math

from signalProcessing import signal


def SignalSamplesAreEqual(file, indices, samples):
    sig = sp.read_file(file, 0)
    expected_samples = sp.signal_samples(sig)

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


# Use to test the Amplitude of DFT and IDFT
def SignalComapreAmplitude(SignalInput=[], SignalOutput=[]):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            if abs(SignalInput[i] - SignalOutput[i]) > 0.001:
                return False
            elif SignalInput[i] != SignalOutput[i]:
                return False
        return True


def RoundPhaseShift(P):
    while P < 0:
        P += 2 * math.pi
    return float(P % (2 * math.pi))


# Use to test the PhaseShift of DFT
def SignalComaprePhaseShift(SignalInput=[], SignalOutput=[]):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            A = round(SignalInput[i])
            B = round(SignalOutput[i])
            if abs(A - B) > 0.0001:
                return False
            elif A != B:
                return False
        return True


def QuantizationTest1(compareFile, Your_EncodedValues, Your_QuantizedValues):
    sig = sp.read_file(compareFile, 1)
    expectedEncodedValues, expectedQuantizedValues = (
        sp.signal_idx(sig),
        sp.signal_samples(sig),
    )

    if (len(Your_EncodedValues) != len(expectedEncodedValues)) or (
        len(Your_QuantizedValues) != len(expectedQuantizedValues)
    ):
        st.write(
            "QuantizationTest1 Test case failed, your signal have different length from the expected one"
        )
        return
    for i in range(len(Your_EncodedValues)):
        if Your_EncodedValues[i] != expectedEncodedValues[i]:
            st.write(
                "QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one"
            )
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            st.write(
                "QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one"
            )
            return
    st.write("QuantizationTest1 Test case passed successfully")


def QuantizationTest2(
    file_name,
    Your_IntervalIndices,
    Your_EncodedValues,
    Your_QuantizedValues,
    Your_SampledError,
):
    expectedIntervalIndices = []
    expectedEncodedValues = []
    expectedQuantizedValues = []
    expectedSampledError = []
    with open(file_name, "r") as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(" ")) == 4:
                L = line.split(" ")
                V1 = int(L[0])
                V2 = str(L[1])
                V3 = float(L[2])
                V4 = float(L[3])
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break
    if (
        len(Your_IntervalIndices) != len(expectedIntervalIndices)
        or len(Your_EncodedValues) != len(expectedEncodedValues)
        or len(Your_QuantizedValues) != len(expectedQuantizedValues)
        or len(Your_SampledError) != len(expectedSampledError)
    ):
        st.write(
            "QuantizationTest2 Test case failed, your signal have different length from the expected one"
        )
        return
    for i in range(len(Your_IntervalIndices)):
        if Your_IntervalIndices[i] != expectedIntervalIndices[i]:
            st.write(
                "QuantizationTest2 Test case failed, your signal have different indicies from the expected one"
            )
            return
    for i in range(len(Your_EncodedValues)):
        if Your_EncodedValues[i] != expectedEncodedValues[i]:
            st.write(
                "QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one"
            )
            return

    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            st.write(
                "QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one"
            )
            return
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            st.write(
                "QuantizationTest2 Test case failed, your SampledError have different values from the expected one"
            )
            return
    st.write("QuantizationTest2 Test case passed successfully")


def DerivativeSignal():
    InputSignal = [i for i in range(1, 101)]
    expectedOutput_first = [1 for i in range(99)]
    expectedOutput_second = [0 for i in range(98)]

    """
    Write your Code here:
    Start
    """

    sig = signal(indices=[i for i in range(len(InputSignal))], samples=InputSignal)
    FirstDrev = sp.compute_first_derivative(sig)
    SecondDrev = sp.compute_second_derivative(sig)

    """
    End
    """

    """
    Testing your Code
    """
    if (len(FirstDrev) != len(expectedOutput_first)) or (
        len(SecondDrev) != len(expectedOutput_second)
    ):
        return "mismatch in length"

    first = second = True
    for i in range(len(expectedOutput_first)):
        if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
            continue
        else:
            first = False
            return "1st derivative wrong"
            return
    for i in range(len(expectedOutput_second)):
        if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
            continue
        else:
            second = False
            return "2nd derivative wrong"
            return
    if first and second:
        return "Derivative Test case passed successfully"
    else:
        return "Derivative Test case failed"
    return


def Shift_Fold_Signal(file_name, Your_indices, Your_samples):
    # return(indices)
    expected_indices = []
    expected_samples = []
    with open(file_name, "r") as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(" ")) == 2:
                L = line.split(" ")
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    # return(f"Current Output Test file is: {file_name} \n")
    if (len(expected_samples) != len(Your_samples)) and (
        len(expected_indices) != len(Your_indices)
    ):
        return "Shift_Fold_Signal Test case failed, your signal have different length from the expected one"

    for i in range(len(Your_indices)):
        if Your_indices[i] != expected_indices[i]:
            return "Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one"

    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            return "Shift_Fold_Signal Test case failed, your signal have different values from the expected one"

    return "Shift_Fold_Signal Test case passed successfully"


def ConvTest(Your_indices, Your_samples):
    """
    Test inputs
    InputIndicesSignal1 =[-2, -1, 0, 1]
    InputSamplesSignal1 = [1, 2, 1, 1 ]

    InputIndicesSignal2=[0, 1, 2, 3, 4, 5 ]
    InputSamplesSignal2 = [ 1, -1, 0, 0, 1, 1 ]
    """

    expected_indices = [-2, -1, 0, 1, 2, 3, 4, 5, 6]
    expected_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1]

    if (len(expected_samples) != len(Your_samples)) and (
        len(expected_indices) != len(Your_indices)
    ):
        return "Conv Test case failed, your signal have different length from the expected one"
    for i in range(len(Your_indices)):
        if Your_indices[i] != expected_indices[i]:
            return "Conv Test case failed, your signal have different indicies from the expected one"
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            return "Conv Test case failed, your signal have different values from the expected one"
    return "Conv Test case passed successfully"


def Compare_Signals(file_name, Your_indices, Your_samples):
    # Correlation
    expected_indices = []
    expected_samples = []
    with open(file_name, "r") as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(" ")) == 2:
                L = line.split(" ")
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    st.write(f"Current Output Test file is: {file_name}")
    if (len(expected_samples) != len(Your_samples)) and (
        len(expected_indices) != len(Your_indices)
    ):
        return "Shift_Fold_Signal Test case failed, your signal have different length from the expected one"
    for i in range(len(Your_indices)):
        if Your_indices[i] != expected_indices[i]:
            return "Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one"
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            return "Correlation Test case failed, your signal have different values from the expected one"
    return "Correlation Test case passed successfully"
