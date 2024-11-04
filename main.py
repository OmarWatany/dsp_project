import signalProcessing as sp
import streamlit as st
import plotly.graph_objects as go
import tests as tst


def scatter_cont_sig(fig_cont, Title: str, Color: str, Signal: sp.Signal):
    # Add a continuous line for the signal
    ln = 50 if len(Signal.indices) > 50 else len(Signal.indices)
    fig_cont.add_trace(
        go.Scatter(
            x=Signal.indices[:ln],
            y=Signal.amplitudes[:ln],
            mode="lines",
            name=Title,
            line=dict(color=Color),
        )
    )

    fig_cont.update_layout(
        title="Continuous Wave",
        xaxis_title="Index",
        yaxis_title="Amplitude",
        width=1000,
        height=500,
    )


def scatter_disc_sig(fig_disc, Title: str, Color: str, Signal: sp.Signal):
    # Add a continuous line for the signal
    ln = 50 if len(Signal.indices) > 50 else len(Signal.indices)
    for i in range(ln):
        fig_disc.add_trace(
            go.Scatter(
                x=[Signal.indices[i], Signal.indices[i]],
                y=[0, Signal.amplitudes[i]],
                line=dict(color=Color),
                mode="lines",
                showlegend=False,
            )
        )
    fig_disc.add_trace(
        go.Scatter(
            x=Signal.indices[:ln],
            y=Signal.amplitudes[:ln],
            mode="markers",
            marker=dict(color=Color, size=8),
            name="Markers",
            showlegend=False,
        )
    )

    fig_disc.update_layout(
        title="Discreate Wave",
        xaxis_title="Index",
        yaxis_title="Amplitude",
        width=1000,
        height=500,
    )


def draw_quantization(quantized_signal, org_signal: sp.Signal, error, show_error):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=org_signal.indices,
            y=org_signal.amplitudes,
            mode="lines",
            name="Original Signal",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=org_signal.indices,
            y=quantized_signal,
            mode="lines",
            name="Quantized Signal",
        )
    )

    if show_error:
        fig.add_trace(
            go.Scatter(
                x=quantized_signal.indices,
                y=error,
                mode="lines",
                name="Quantization Error",
            )
        )

    fig.update_layout(
        title="Quantized Signal and Quantization Error",
        xaxis_title="Index",
        yaxis_title="Amplitude",
        width=1000,
        height=500,
    )

    st.plotly_chart(fig)


def file_signal(index=-1):
    if index > 0:
        title = f"Upload signal {index} txt file"
    else:
        title = "Upload signal txt file"
    uploaded_file = st.file_uploader(title, type="txt", key=index)
    if uploaded_file is not None:
        return sp.read_file(uploaded_file), uploaded_file
    return None, None


def generated_signal(index=0) -> sp.Signal:
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

    return sp.generate_signal(
        sin_flag, False, sp.Signal_type.Time, amp, samplingFreq, freq, phase_shift
    )


def Plot_signal(sig: sp.Signal):
    # clicked = st.button("Show", type="primary")
    if sig:
        cont_fig = go.Figure()
        disc_fig = go.Figure()
        scatter_cont_sig(cont_fig, "Signal", "blue", sig)
        scatter_disc_sig(disc_fig, "Signal", "blue", sig)
        st.plotly_chart(disc_fig)
        st.plotly_chart(cont_fig)


def todo(signal):
    return signal


def Signal_Source():
    sig, uploaded_file = None, None
    operation = st.selectbox(
        "Choose Source",
        ("Read from file", "Generate"),
    )
    if operation == "Generate":
        sig = generated_signal()
    elif operation == "Read from file":
        sig, uploaded_file = file_signal()
    return sig, uploaded_file


def Arithmatic_Operations():
    operations = {
        "Add": sp.sig_add,
        "Sub": sp.sig_sub,
        "Mul": sp.sig_mul,
        "Normalize": sp.sig_norm,
        "Square": sp.sig_square,
        "Accumulate": sp.sig_acc,
        "Shift": todo,
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


def quantize_signal():
    pass


# similar to FOS Commands list
operations = {
    "Plot": Signal_Source,
    "Quantize": Signal_Source,
    "Arithmatic": Arithmatic_Operations,
}

if __name__ == "__main__":
    st.set_page_config(page_title="DSP Framework", layout="wide")
    st.title("DSP Framework")

    main_cols = st.columns(2)
    with main_cols[0]:
        # st.markdown("#### Choose operation")
        op = st.selectbox("**Choose operation type**", operations.keys())
        sig, uploaded_file = operations[op]()
    with main_cols[1]:
        # Plotting
        if sig and op not in "Quantize":
            Plot_signal(sig)

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
                draw_quantization(quantized, sig, error, 0)

    with main_cols[0]:
        test_file = st.file_uploader("Test file", type="txt")
        if test_file and sig and op != "Quantize":
            st.write(tst.SignalSamplesAreEqual(test_file, sig.indices, sig.amplitudes))

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
