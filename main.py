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


def sig_add(signal_1, signal_2) -> sp.Signal:
    return signal_1


def sig_sub(signal_1, signal_2) -> sp.Signal:
    return signal_1


def sig_mul(signal, value) -> sp.Signal:
    return signal


def todo(signal):
    return signal


# similar to FOS Commands list
arth_operation = {
    # two signals
    "Add": sig_add,
    "Sub": sig_sub,
    # signal , value
    "Mul": sig_mul,
    # signal , range
    "Normalize": sp.sig_norm,
    # Todo
    "square": sp.sig_square,
    "Accumulate": todo,
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
        if sig:
            Plot_signal(sig)

    if arth_op in ["Normalize"]:
        _range = st.radio("Range", ["0 , 1", "-1 , 1"], horizontal=True)
        if sig:
            sig = arth_operation[arth_op](sig, _range == "0 , 1")
        if sig:
            Plot_signal(sig)

    # Temporary
    temp_clicked = st.button("Show Temp", type="primary", disabled=0)
    if temp_clicked:
        s = sp.generate_signal(
            True, False, sp.Signal_type.Time, 3, 720, 360, 1.96349540849362
        )

        c = sp.generate_signal(
            False, False, sp.Signal_type.Time, 3, 500, 200, 2.35619449019235
        )

        # showing sin and cosine signals at the same time
        cont_fig = go.Figure()
        disc_fig = go.Figure()

        scatter_cont_sig(cont_fig, "Sin Wave", "blue", s)
        scatter_disc_sig(disc_fig, "Sin Wave", "blue", s)
        scatter_cont_sig(cont_fig, "Cos Wave", "red", c)
        scatter_disc_sig(disc_fig, "Cos Wave", "red", c)

        st.plotly_chart(disc_fig)
        st.plotly_chart(cont_fig)
