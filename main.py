import streamlit as st
import tests as tst
import signalProcessing as sp
import plot as plt


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
    out_sig = None
    sig, uploaded_file = Signal_Source(f_flag=1)
    func = st.radio("Transformation function", ["DFT", "IDFT"], horizontal=True)
    if func == "DFT" and sig:
        fs = st.number_input("Sampling Freq (HZ) :", value=1)
        out_sig = sp.fourier_transform(0, sig, fs)

    elif func == "IDFT" and sig:
        out_sig = sp.fourier_transform(1, sig)

    # testing
    test_file = st.file_uploader("Test file", type="txt")
    if test_file and out_sig:
        if func == "DFT" and out_sig:
            test_sig = sp.read_file(test_file)
            r = False
            r = tst.SignalComapreAmplitude(
                sp.signal_samples(test_sig), sp.signal_samples(out_sig)
            ) and tst.SignalComaprePhaseShift(
                sp.signal_phase_shifts(test_sig), sp.signal_phase_shifts(out_sig)
            )
            if r:
                st.write("Test case passed successfully")
            else:
                st.write("Test case failed")

        elif func == "IDFT" and sig:
            st.write(
                tst.SignalSamplesAreEqual(
                    test_file, sp.signal_idx(out_sig), sp.signal_samples(out_sig)
                )
            )

    return out_sig, uploaded_file


def Quantization():
    sig, uploaded_file = Signal_Source(f_flag=1)
    interval_index, encoded, quantized, error = None, None, None, None
    with main_cols[0]:
        quant_type = st.radio("Quantization Type", ["Levels", "Bits"], horizontal=True)
        no_of_levels = (
            st.number_input("Number of Levels:", min_value=1, value=2)
            if quant_type == "Levels"
            else (1 << st.number_input("Bits:", min_value=1, value=1))
        )

        show_data = st.checkbox("Show Data", value=False)
        show_error = st.checkbox("Show Error", value=False)

    with main_cols[1]:
        if sig:
            interval_index, encoded, quantized, error = sp.quantize(sig, no_of_levels)
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

    # testing
    turn_off_test_file_input()
    test_file = st.file_uploader("Test file", type="txt")
    if test_file and sig and op == "Quantize" and encoded and quantized:
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
    return None, None


def Conv():
    turn_off_test_file_input()
    sig = None
    signals = []
    cols = st.columns(2)
    for i in range(len(cols)):
        with cols[i]:
            signal, up_file = Signal_Source(idx=i)
            signals.append(signal)

    if signals[0] and signals[1]:
        sig = sp.convolution(signals[0], signals[1])
    return sig, None


def Corr():
    sig = None
    signals = []
    cols = st.columns(2)
    for i in range(len(cols)):
        with cols[i]:
            signal, up_file = Signal_Source(idx=i)
            signals.append(signal)

    if signals[0] and signals[1]:
        sig = sp.correlation(signals[0], signals[1])
    return sig, None


def Shift_Fold(op):
    sig, uploaded_file = Signal_Source()
    if op == "Shift":
        fold = st.checkbox("Fold", value=False)
        steps = st.number_input("Steps (+ for Delay , - for Advance)", value=0)
        if fold and sig:
            sig = sp.sig_fold(sig)
        if sig:
            sig = sp.sig_shift(sig, steps)

    elif op == "Fold":
        if sig:
            sig = sp.sig_fold(sig)

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

    return sig, uploaded_file
    pass


def Filter(filter, sig):
    fs = st.number_input("Sampling Freq (Hz)", value=1)
    transitionBand = st.number_input("Transition Band (Hz)", value=1) / fs
    stopA = st.number_input("Stopband attenuation (dB)", value=1)

    if filter in ["Low pass", "High pass"]:
        # input
        fc = st.number_input("Cutoff Freq (Hz)", value=1) / fs
        # output
        filter_coefficients = sp.sig_filter(filter, fs, transitionBand, stopA, fc)

    elif filter in ["Band pass", "Band stop"]:
        # input
        f1 = st.number_input("F1 (Hz)", value=1) / fs
        f2 = st.number_input("F2 (Hz)", value=1) / fs
        # output
        filter_coefficients = sp.sig_filter(filter, fs, transitionBand, stopA, f1, f2)

    if sig:
        sig = sp.convolution(sig, filter_coefficients)
        return sig
    else:
        return filter_coefficients


def Resample(sig):
    if not sig:
        return None
    M = st.number_input("Decimation Factor", value=0)
    L = st.number_input("Interpolation Factor", value=0)

    if M == 0 and L != 0:
        sig = sp.sig_upsample_resample(sig, L)
        sig = Filter("Low pass", sig)
        samples = sp.signal_samples(sig)
        indices = sp.signal_idx(sig)
        sz = len(samples)
        zeroCount = 0
        for s in samples[::-1]:
            if s == 0:
                zeroCount += 1
            else:
                break
        sig = sp.signal(
            sig["periodic"],
            sig["signal_type"],
            indices[: sz - zeroCount],
            samples[: sz - zeroCount],
        )
    elif M != 0 and L == 0:
        sig = Filter("Low pass", sig)
        sig = sp.sig_downsample_resample(sig, M)
    else:
        sig = sp.sig_upsample_resample(sig, L)
        sig = Filter("Low pass", sig)
        sig = sp.sig_downsample_resample(sig, M)
        samples = sp.signal_samples(sig)
        indices = sp.signal_idx(sig)
        sz = len(samples)
        zeroCount = 0
        for s in samples[::-1]:
            if s == 0:
                zeroCount += 1
            else:
                break
        sig = sp.signal(
            sig["periodic"],
            sig["signal_type"],
            indices[: sz - zeroCount],
            samples[: sz - zeroCount],
        )

    return sig


def Time_Domain():
    time_domain_ops = [
        "Remove DC component",
        "Filter",
        "Resample",
        "Shift",
        "Fold",
        "Sharpning",
        "Smoothe",
    ]
    op = st.selectbox("Choose operation", time_domain_ops)
    sig, uploaded_file = None, None

    if op == "Smoothe":
        sig, uploaded_file = Signal_Source()
        winSize = st.number_input("Window Size:", min_value=1, value=2)
        if sig:
            sig = sp.sig_smoothe(sig, winSize)
    elif op == "Filter":
        sig, uploaded_file = Signal_Source()
        filters = ["Low pass", "High pass", "Band pass", "Band stop"]
        filter = st.selectbox("**Choose Filter**", filters)
        sig = Filter(filter, sig)

    elif op == "Resample":
        sig, uploaded_file = Signal_Source()
        sig = Resample(sig)

    elif op in ["Shift", "Fold"]:
        turn_off_test_file_input()
        sig, uploaded_file = Shift_Fold(op)

    elif op == "Sharpning":
        turn_off_test_file_input()
        st.write(tst.DerivativeSignal())

    elif op == "Remove DC component":
        sig, uploaded_file = Signal_Source()
        if sig:
            sig = sp.sig_rm_dc(sig)

    return sig, uploaded_file


def Freq_Domain():
    freq_domain_ops = ["Remove DC component", "Fourier Transform", "DCT"]
    op = st.selectbox("Choose operation", freq_domain_ops)

    sig, uploaded_file = None, None
    if op == "DCT":
        sig, uploaded_file = Signal_Source()
        if sig:
            sig = sp.sig_dct(sig)

    elif op == "Remove DC component":
        sig, uploaded_file = Signal_Source()
        if sig:
            sig = sp.sig_rm_dc_freq(sig)

    elif op == "Fourier Transform":
        turn_off_test_file_input()
        sig, uploaded_file = Fourier_Transform()

    else:
        sig, uploaded_file = freq_domain_ops[op]()

    return sig, uploaded_file


def turn_off_test_file_input():
    global draw_test_file_input
    draw_test_file_input = False


if __name__ == "__main__":
    main_menu = {
        "Plot": Signal_Source,
        "Arithmatic": Arithmatic_Operations,
        "Time Domain": Time_Domain,
        "Freq Domain": Freq_Domain,
        "Quantize": Quantization,
        "Convolution": Conv,
        "Correlation": Corr,
    }

    st.set_page_config(page_title="DSP Framework", layout="wide")
    st.title("DSP Framework")
    draw_test_file_input = True

    main_cols = st.columns(2)
    with main_cols[0]:
        op = st.selectbox("**Choose operation type**", main_menu.keys())
        sig, uploaded_file = main_menu[op]()

    with main_cols[1]:
        if sig and op not in ["Quantize"]:
            plt.plot_signal(sig)
            content = sp.sig_to_text(sig)

            st.download_button(
                label="Export Signal",
                data=content,
                file_name="signal_exported.txt",
            )

    with main_cols[0]:
        test_file = None
        if draw_test_file_input:
            test_file = st.file_uploader("Test file", type="txt")

        if sig and op == "Convolution":
            st.write(tst.ConvTest(sp.signal_idx(sig), sp.signal_samples(sig)))

        elif sig and test_file:
            st.write(
                tst.SignalSamplesAreEqual(
                    test_file, sp.signal_idx(sig), sp.signal_samples(sig)
                )
            )


# resampling
def lowpass_filter_manual(signal, cutoff, fs):
    """
    Manually applies a lowpass filter using a basic moving average method.

    Parameters:
        signal (list): Input signal
        cutoff (float): Cutoff frequency
        fs (int): Sampling frequency

    Returns:
        list: Filtered signal
    """
    RC = 1.0 / (2 * np.pi * cutoff)  # Time constant
    dt = 1.0 / fs  # Time step
    alpha = dt / (RC + dt)
    filtered_signal = [signal[0]]  # Initialize with the first value
    for i in range(1, len(signal)):
        filtered_signal.append(
            filtered_signal[-1] + alpha * (signal[i] - filtered_signal[-1])
        )
    return filtered_signal


def upsample_manual(signal, L):
    """
    Upsamples a signal by inserting zeros.

    Parameters:
        signal (list): Input signal
        L (int): Interpolation factor

    Returns:
        list: Upsampled signal
    """
    upsampled = [0] * (len(signal) * L)
    for i, value in enumerate(signal):
        upsampled[i * L] = value
    return upsampled


def downsample_manual(signal, M):
    """
    Downsamples a signal by keeping every Mth sample.

    Parameters:
        signal (list): Input signal
        M (int): Decimation factor

    Returns:
        list: Downsampled signal
    """
    return signal[::M]


def resample_without_libs(signal, M, L, lowpass_cutoff, fs):
    """
    Resamples a signal manually without using libraries.

    Parameters:
        signal (list): Input signal
        M (int): Decimation factor
        L (int): Interpolation factor
        lowpass_cutoff (float): Lowpass filter cutoff frequency
        fs (int): Sampling frequency

    Returns:
        list: Resampled signal
    """
    if M == 0 and L == 0:
        raise ValueError("At least one of M or L must be non-zero.")

    # Case 1: Upsampling
    if M == 0 and L != 0:
        upsampled_signal = upsample_manual(signal, L)
        filtered_signal = lowpass_filter_manual(
            upsampled_signal, lowpass_cutoff, fs * L
        )
        return filtered_signal

    # Case 2: Downsampling
    elif M != 0 and L == 0:
        filtered_signal = lowpass_filter_manual(signal, lowpass_cutoff, fs)
        downsampled_signal = downsample_manual(filtered_signal, M)
        return downsampled_signal

    # Case 3: Change sample rate by fraction
    elif M != 0 and L != 0:
        upsampled_signal = upsample_manual(signal, L)
        filtered_signal = lowpass_filter_manual(
            upsampled_signal, lowpass_cutoff, fs * L
        )
        resampled_signal = downsample_manual(filtered_signal, M)
        return resampled_signal


# Input from the user
# signal = [float(x) for x in input("Enter the signal values separated by spaces: ").split()]
# M = int(input("Enter the decimation factor M: "))
# L = int(input("Enter the interpolation factor L: "))
# lowpass_cutoff = float(input("Enter the lowpass filter cutoff frequency: "))
# fs = int(input("Enter the sampling frequency of the input signal: "))
#
# # Resample the signal
# resampled_signal = resample_without_libs(signal, M, L, lowpass_cutoff, fs)
#
# print("Resampled signal:", resampled_signal)
