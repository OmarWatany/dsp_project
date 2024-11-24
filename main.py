from audioop import reverse

import streamlit as st
from numpy.ma.core import indices

import tests as tst
import signalProcessing as sp
import plot as plt
from signalProcessing import convolution


def file_signal(index=-1, f_flag: bool = 0):
    if index > 0:
        title = f"Upload signal {index} txt file"
    else:
        title = "Upload signal txt file"
    uploaded_file = st.file_uploader(title, type="txt", key=index)
    if uploaded_file:
        return sp.read_file(uploaded_file, f_flag), uploaded_file
    return None, None


def generated_signal(index=0):
    titles = ["Wave Type ", "Amp ", "Phase shift ", "Freq ", "Sampling Freq "]
    for i in range(len(titles)):
        titles[i] = "Insert " + titles[i] + (str(index) if index > 0 else " ")

    sin_flag = (
        st.selectbox(
            titles[0],
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

    return sp.generate_signal(sin_flag, False, amp, samplingFreq, freq, phase_shift)


def Signal_Source(f_flag: bool = 0, idx: int = -1):
    sig, uploaded_file = None, None
    operation = st.selectbox(
        f"Choose Source {idx+1 if idx >= 0 else ''}",
        ("Read from file", "Generate"),
    )
    if operation == "Generate":
        sig = generated_signal(idx + 1)
    elif operation == "Read from file":
        sig, uploaded_file = file_signal(idx + 1, f_flag=f_flag)
    return sig, uploaded_file


def Arithmatic_Operations():
    operations = {
        "Add": sp.sig_add,
        "Sub": sp.sig_sub,
        "Mul": sp.sig_mul,
        "Normalize": sp.sig_norm,
        "Square": sp.sig_square,
        "Accumulate": sp.sig_acc,
    }

    op = st.selectbox("Choose Arithmatic Operation", operations.keys())
    sig = None
    if op in ["Add", "Sub"]:
        signals = []
        cols = st.columns(2)
        for i in range(len(cols)):
            with cols[i]:
                signal, up_file = Signal_Source(idx=i)
                signals.append(signal)

        if signals[0] and signals[1]:
            sig = operations[op](signals[0], signals[1])

    else:
        sig, uploaded_file = Signal_Source()

    if op == "Mul":
        value = st.number_input("Insert Scalar", format="%f")
        if sig:
            sig = operations[op](sig, value)

    elif op == "Normalize":
        _range = st.radio("Range", ["0 , 1", "-1 , 1"], horizontal=True)
        if sig:
            sig = operations[op](sig, _range == "0 , 1")

    elif op in ["Accumulate", "Square"]:
        if sig:
            sig = operations[op](sig)

    return sig, None


def Fourier_Transform():
    sig, uploaded_file = Signal_Source(f_flag=1)
    func = st.radio("Transformation function", ["DFT", "IDFT"], horizontal=True)
    if func == "DFT" and sig:
        fs = st.number_input("Sampling Freq (HZ) :", value=1)
        sig = sp.fourier_transform(0, sig, fs)
    elif func == "IDFT" and sig:
        sig = sp.fourier_transform(1, sig)

    return sig, uploaded_file


def Conv():
    sig = None
    signals = []
    cols = st.columns(2)
    for i in range(len(cols)):
        with cols[i]:
            signal, up_file = Signal_Source(idx=i)
            signals.append(signal)

    if signals[0] and signals[1]:
        sig = convolution(signals[0], signals[1])
    return sig, None


# similar to FOS Commands list
operations = {
    "Plot": Signal_Source,
    "Fourier Transform": Fourier_Transform,
    "DCT": None,
    "Quantize": Signal_Source,
    "Arithmatic": Arithmatic_Operations,
    "Shift": None,
    "Fold": None,
    "Sharpning": None,
    "Smoothe": None,
    "Remove DC component": None,
    "Convolution": Conv,
}

if __name__ == "__main__":
    st.set_page_config(page_title="DSP Framework", layout="wide")
    st.title("DSP Framework")

    main_cols = st.columns(2)
    with main_cols[0]:
        op = st.selectbox("**Choose operation type**", operations.keys())
        if op == "DCT":
            sig, uploaded_file = Signal_Source()
            if sig:
                sig = sp.sig_dct(sig)

        elif op == "Shift":
            sig, uploaded_file = Signal_Source()
            steps = st.number_input("Steps (+ for Delay , - for Advance)", value=0)
            fold = st.checkbox("Fold", value=False)
            if fold and sig:
                sig = sp.sig_fold(sig)
            if sig:
                sig = sp.sig_shift(sig, steps)

        elif op == "Fold":
            sig, uploaded_file = Signal_Source()
            if sig:
                sig = sp.sig_fold(sig)

        elif op == "Smoothe":
            sig, uploaded_file = Signal_Source()
            winSize = st.number_input("Window Size:", min_value=1, value=2)
            if sig:
                sig = sp.sig_smoothe(sig, winSize)

        elif op == "Sharpning":
            st.write(tst.DerivativeSignal())
            sig = None

        elif op == "Remove DC component":
            sig, uploaded_file = Signal_Source()
            if sig:
                sig = sp.sig_rm_dc(sig)

        else:
            sig, uploaded_file = operations[op]()

    with main_cols[1]:
        if sig and op not in ["Quantize"]:
            plt.plot_signal(sig)
            content = sp.sig_to_text(sig)

            st.download_button(
                label="Export Signal",
                data=content,
                file_name="signal_exported.txt",
            )

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
                interval_index, encoded, quantized, error = sp.quantize(
                    sig, no_of_levels
                )
                # Display chosen outputs
                if show_data:
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("Interval Index:", interval_index)
                        st.write("Encoded Signal:", encoded)
                    with cols[1]:
                        st.write("Quantized Signal:", quantized)
                        st.write("Quantization Error:", error)
                plt.draw_quantization(quantized, sig, error, show_error)

    with main_cols[0]:
        test_file = st.file_uploader("Test file", type="txt")
        if test_file and sig and op in ["Fold", "Shift"]:
            indices = sp.signal_idx(sig)
            amps = sp.signal_samples(sig)
            st.write(
                tst.Shift_Fold_Signal(
                    test_file,
                    indices,
                    amps,
                )
            )

        elif sig and op == "Convolution":
            st.write(tst.ConvTest(sp.signal_idx(sig), sp.signal_samples(sig)))

        elif test_file and op == "Quantize" and encoded and quantized:
            if uploaded_file.name == "Quan1_input.txt":
                tst.QuantizationTest1(test_file, encoded, quantized)
            else:
                tst.QuantizationTest2(
                    f"Tasks/task3/{test_file.name}",
                    interval_index,
                    encoded,
                    quantized,
                    error,
                )

        elif test_file and sig and op not in ["Fourier Transform"]:
            st.write(
                tst.SignalSamplesAreEqual(
                    test_file, sp.signal_idx(sig), sp.signal_samples(sig)
                )
            )
