import streamlit as st
import tests as tst
from plot import *
from signalProcessing import *


def file_signal(index=-1, f_flag: bool = 0):
    if index > 0:
        title = f"Upload signal {index} txt file"
    else:
        title = "Upload signal txt file"
    uploaded_file = st.file_uploader(title, type="txt", key=index)
    if uploaded_file is not None:
        return read_file(uploaded_file, f_flag), uploaded_file
    return None, None


def generated_signal(index=0):
    titles = ["Wave Type ", "Amp ", "Phase shift ", "Freq ", "Sampling Freq "]
    for i in range(len(titles)):
        titles[i] = "Insert " + titles[i] + (str(index) if index > 0 else " ")

    sin_flag = (
        st.selectbox(
            titles[0] + (str(index) if index > 0 else ""),
            ("Sin", "Cos"),
        )
        == "Sin"
    )
    amp = st.number_input(titles[1], value=1)
    phase_shift = st.number_input(titles[2], value=0.0, format="%f")
    freq = st.number_input(titles[3], value=1)
    samplingFreq = st.number_input(titles[4], value=10)

    # TODO: Handle Error
    # if samplingFreq < 2 * freq:

    return generate_signal(
        sin_flag, False, Signal_type.TIME, amp, samplingFreq, freq, phase_shift
    )


def Signal_Source(f_flag: bool = 0):
    sig, uploaded_file = None, None
    operation = st.selectbox(
        "Choose Source",
        ("Read from file", "Generate"),
    )
    if operation == "Generate":
        sig = generated_signal()
    elif operation == "Read from file":
        sig, uploaded_file = file_signal(f_flag=f_flag)
    return sig, uploaded_file


def Arithmatic_Operations():
    operations = {
        "Add": sig_add,
        "Sub": sig_sub,
        "Mul": sig_mul,
        "Normalize": sig_norm,
        "Square": sig_square,
        "Accumulate": sig_acc,
        "Shift": None,
    }

    op = st.selectbox("Choose Arithmatic Operation", operations.keys())
    sig = None
    if op in ["Add", "Sub"]:
        signals = []
        cols = st.columns(2)
        # for col in cols:
        for i in range(len(cols)):
            with cols[i]:
                operation = st.selectbox(
                    f"Choose Operation {i+1}",
                    ("Read from file", "Generate"),
                    # ("Read from file", "Generate"),
                )
                signal = None
                if operation == "Generate":
                    signal = generated_signal(i)
                elif operation == "Read from file":
                    signal, uploaded_file = file_signal(i)
                signals.append(signal)

        if signals[0] and signals[1]:
            sig = operations[op](signals[0], signals[1])

    if op not in ["Add", "Sub"]:
        sig, uploaded_file = Signal_Source()

    if op in ["Mul"]:
        value = st.number_input("Insert Scalar", format="%f")
        if sig:
            sig = operations[op](sig, value)

    if op in ["Normalize"]:
        _range = st.radio("Range", ["0 , 1", "-1 , 1"], horizontal=True)
        if sig:
            sig = operations[op](sig, _range == "0 , 1")

    if op in ["Accumulate", "Square"]:
        if sig:
            sig = operations[op](sig)
    return sig, None


def fourier_transform():
    sig, uploaded_file = Signal_Source(f_flag=1)
    func = st.radio("Transformation function", ["DFT", "IDFT"], horizontal=True)
    if func == "DFT" and sig:
        fs = st.number_input("Sampling Freq (HZ) :", value=1)
        sig = fourier_transform_(0, sig, fs)
    elif func == "IDFT" and sig:
        sig = fourier_transform_(1, sig)

    return sig, uploaded_file


# similar to FOS Commands list
operations = {
    "Plot": Signal_Source,
    "Fourier Transform": fourier_transform,
    "Quantize": Signal_Source,
    "Arithmatic": Arithmatic_Operations,
}

if __name__ == "__main__":
    st.set_page_config(page_title="DSP Framework", layout="wide")
    st.title("DSP Framework")

    main_cols = st.columns(2)
    with main_cols[0]:
        op = st.selectbox("**Choose operation type**", operations.keys())
        sig, uploaded_file = operations[op]()

    with main_cols[1]:
        if sig and op not in ["Quantize"]:
            plot_signal(sig)

    interval_index, encoded, quantized, error = None, None, None, None
    if op == "Quantize":
        with main_cols[0]:
            quant_type = st.radio(
                "Quantization Type", ["Levels", "Bits"], horizontal=True
            )
            no_of_levels = (
                st.number_input("Number of Levels:", min_value=1, value=2)
                if quant_type == "Levels"
                else (1 << st.number_input("Bits:", min_value=1, value=1))
            )

            show_data = st.checkbox("Show Data", value=False)
            show_error = st.checkbox("Show Error", value=False)

        with main_cols[1]:
            if sig:
                interval_index, encoded, quantized, error = quantize(sig, no_of_levels)
                # Display chosen outputs
                if show_data:
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("Interval Index:", interval_index)
                        st.write("Encoded Signal:", encoded)
                    with cols[1]:
                        st.write("Quantized Signal:", quantized)
                        st.write("Quantization Error:", error)
                draw_quantization(quantized, sig, error, show_error)

    with main_cols[0]:
        test_file = st.file_uploader("Test file", type="txt")
        if test_file and sig and op not in ["Quantize", "Fourier Transform"]:
            st.write(
                tst.SignalSamplesAreEqual(
                    test_file, signal_idx(sig), signal_samples(sig)
                )
            )

        if test_file and encoded and quantized:
            if uploaded_file.name == "Quan1_input.txt":
                tst.QuantizationTest1(test_file, encoded, quantized)
            else:
                tst.QuantizationTest2(
                    f"Tasks/Task 3/{test_file.name}",
                    interval_index,
                    encoded,
                    quantized,
                    error,
                )
