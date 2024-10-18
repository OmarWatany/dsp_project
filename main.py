import signalProcessing as sp
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def scatter_cont_sig(fig_cont, Title: str, Color: str, Signal, Indices):
    # Add a continuous line for the signal
    ln = 50 if len(Indices) > 50 else len(Indices)
    fig_cont.add_trace(
        go.Scatter(
            x=Indices[:ln],
            y=Signal[:ln],
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


def scatter_disc_sig(fig_disc, Title: str, Color: str, Signal, Indices):
    # Add a continuous line for the signal
    ln = 50 if len(Indices) > 50 else len(Indices)
    for i in range(ln):
        fig_disc.add_trace(
            go.Scatter(
                x=[Indices[i], Indices[i]],
                y=[0, Signal[i]],
                line=dict(color=Color),
                mode="lines",
                showlegend=False,
            )
        )
    fig_disc.add_trace(
        go.Scatter(
            x=Indices[:ln],
            y=Signal[:ln],
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


def Plot_signal_pd(Continuous: bool, Signal, Indices):
    dt = pd.DataFrame({"Amplitude": Signal})
    if Continuous:
        st.line_chart(dt)


def generated_signal_args() -> sp.Signal:
    sin_flag = (
        st.selectbox(
            "Chose wave",
            ("Sin", "Cos"),
        )
        == "Sin"
    )
    amp = st.number_input("Insert Amp", value=1)
    phase_shift = st.number_input("Insert Phase shift", value=0.0, format="%f")
    freq = st.number_input("Insert Freq", value=1)
    samplingFreq = st.number_input("Insert Sampling Freq", value=10)

    # TODO: Handle Error
    # if samplingFreq < 2 * freq:

    clicked = st.button("Show", type="primary")
    if clicked:
        sig = sp.generate_signal(
            sin_flag,
            False,
            sp.Signal_type.Time,
            amp,
            samplingFreq,
            freq,
            phase_shift,
        )
        return sig
    return sp.Signal()


def Plot_signal(s, indices):
    cont_fig = go.Figure()
    disc_fig = go.Figure()
    scatter_cont_sig(cont_fig, "Signal", "blue", s, indices)
    scatter_disc_sig(disc_fig, "Signal", "blue", s, indices)
    st.plotly_chart(disc_fig)
    st.plotly_chart(cont_fig)


def add_sig(signal_1, signal_2):
    pass


def sub_sig(signal_1, signal_2):
    pass


if __name__ == "__main__":
    st.write("""
    # DSP Framework
    """)

    operation = st.selectbox(
        "Choose Operation",
        ("Read from file", "Generate"),
    )
    if operation == "Generate":
        sig = generated_signal_args()
        Plot_signal(sig.amplitudes, sig.indices)
    else:
        uploaded_file = st.file_uploader("Upload a signal txt file", type="txt")
        read_button = st.button("Read Signal")
        if read_button and uploaded_file is not None:
            sig = sp.read_file(uploaded_file)
            Plot_signal(sig.amplitudes, sig.indices)

    # Temporary
    temp_clicked = st.button("Show Temp", type="primary", disabled=True)
    if temp_clicked:
        s, s_indices = sp.generate_signal(
            True, False, sp.Signal_type.Time, 3, 720, 360, 1.96349540849362
        )

        c, c_indices = sp.generate_signal(
            False, False, sp.Signal_type.Time, 3, 500, 200, 2.35619449019235
        )

        # showing sin and cosine signals at the same time
        cont_fig = go.Figure()
        disc_fig = go.Figure()

        scatter_cont_sig(cont_fig, "Sin Wave", "blue", s, s_indices)
        scatter_disc_sig(disc_fig, "Sin Wave", "blue", s, s_indices)

        scatter_cont_sig(cont_fig, "Cos Wave", "red", c, s_indices)
        scatter_disc_sig(disc_fig, "Cos Wave", "red", c, c_indices)

        st.plotly_chart(disc_fig)
        st.plotly_chart(cont_fig)
