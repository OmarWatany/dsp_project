import signalProcessing as sp
import streamlit as st


def SignalSamplesAreEqual(file, indices, samples):
    sig = sp.read_file(file, 1)
    # exiected_indices = sig.indices
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


def QuantizationTest1(compareFile, Your_EncodedValues, Your_QuantizedValues):
    sig = sp.read_file(compareFile, 1)
    expectedEncodedValues, expectedQuantizedValues = sig.indices, sig.amplitudes
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
