import signalProcessing as sp
import streamlit as st
import plotly.graph_objects as go


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


def file_signal(index=-1) -> sp.Signal:
    if index > 0:
        title = f"Upload signal {index} txt file"
    else:
        title = "Upload signal txt file"
    uploaded_file = st.file_uploader(title, type="txt", key=index)
    if uploaded_file is not None:
        return sp.read_file(uploaded_file)


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
    clicked = st.button("Show", type="primary")
    if clicked:
        cont_fig = go.Figure()
        disc_fig = go.Figure()
        scatter_cont_sig(cont_fig, "Signal", "blue", sig)
        scatter_disc_sig(disc_fig, "Signal", "blue", sig)
        st.plotly_chart(disc_fig)
        st.plotly_chart(cont_fig)


def todo(signal):
    return signal


# similar to FOS Commands list
arth_operation = {
    # two signals
    "Add": sp.sig_add,
    "Sub": sp.sig_sub,
    # signal , value
    "Mul": sp.sig_mul,
    # signal , range
    "Normalize": sp.sig_norm,
    # Todo
    "Square": sp.sig_square,
    "Accumulate": sp.sig_acc,
    "Shift": todo,
}


if __name__ == "__main__":
    st.set_page_config(
        page_title="DSP Framework",
    )
    st.title("DSP Framework")

    # st.write("### Select an option:")
    # arth_op = st.selectbox("Choose Arithmatic arth_op", arth_operation.keys(),label_visibility="hidden")
    arth_op = st.selectbox("Choose Arithmatic Operation", arth_operation.keys())

    sig = None
    # Todo Accept number of signals
    if arth_op in ["Add", "Sub"]:
        signals = []
        cols = st.columns(2)
        # for col in cols:
        for i in range(len(cols)):
            with cols[i]:
                operation = st.selectbox(
                    f"Choose Operation {i+1}",
                    ("Generate", "Read from file"),
                    # ("Read from file", "Generate"),
                )
                signals.append(
                    generated_signal(i + 1)
                    if operation == "Generate"
                    else file_signal(i + 1)
                    if operation == "Read from file"
                    else None
                )

        # arth_op[arth_op](sig1,sig2)
        st.markdown("Two arguments")

    if arth_op not in ["Add", "Sub"]:
        operation = st.selectbox(
            "Choose Operation",
            ("Read from file", "Generate"),
        )
        sig = (
            generated_signal()
            if operation == "Generate"
            else file_signal()
            if operation == "Read from file"
            else None
        )

    if arth_op in ["Mul"]:
        value = st.number_input("Insert Scalar", format="%f")
        if sig:
            sig = arth_operation[arth_op](sig, value)

    if arth_op in ["Normalize"]:
        _range = st.radio("Range", ["0 , 1", "-1 , 1"], horizontal=True)
        if sig:
            sig = arth_operation[arth_op](sig, _range == "0 , 1")
    if arth_op in ["Accumulate", "Square"]:
        if sig:
            sig = arth_operation[arth_op](sig)

    if sig:
        Plot_signal(sig)

    test_file = st.file_uploader("Test file", type="txt")
    if test_file and sig:
        st.write(sp.SignalSamplesAreEqual(test_file, sig.indices, sig.amplitudes))
