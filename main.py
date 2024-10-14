import signalProcessing as sp
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def Plot_signal_plotly(Continuous: bool, Signal, Indices):
    fig_cont = go.Figure()
    # Add a continuous line for the signal
    fig_cont.add_trace(
        go.Scatter(
            x=Indices[:100], y=Signal[:100], mode="lines", name="Continuous Signal"
        )
    )

    fig_cont.update_layout(
        title="Continuous Signal",
        xaxis_title="Index",
        yaxis_title="Amplitude",
        width=1000,
        height=500,
    )
    st.plotly_chart(fig_cont)


def Plot_signal_pd(Continuous: bool, Signal, Indices):
    dt = pd.DataFrame({"Amplitude": Signal})
    if Continuous:
        st.line_chart(dt)


st.write("""
# DSP Framwork
""")

operation = st.selectbox(
    "Choose Operation",
    ("Read from file", "Generate"),
)

if operation == "Generate":
    sin_flag = st.selectbox(
        "Chose wave",
        ("Sin", "Cos", "both"),
    )
    amp = st.number_input("Insert Amp", value=0)
    phase_shift = st.number_input("Insert Phase shift", value=0)
    freq = st.number_input("Insert Freq", value=0)
    samplingFreq = st.number_input("Insert Sampling Freq", value=0)
    # TODO: Handle Error
    # if samplingFreq < 2 * freq:

    clicked = st.button("Show", type="primary")

    fig_width = 10
    fig_height = fig_width / 2

    if clicked:
        # s, indices = sp.generate_signal(
        #     True, False, sp.Signal_type.Time, amp, samplingFreq, freq, phase_shift
        # )
        s, indices = sp.generate_signal(
            True, False, sp.Signal_type.Time, 3, 720, 360, 1.96349540849362
        )
        Plot_signal_plotly(True, s, indices)
else:
    file_name = st.text_input("File name")
