import signalProcessing as sp
import streamlit as st
import matplotlib.pyplot as plt


def Plot_signal(Continuous: bool, Signal, Indices):
    if Continuous:
        continuous_fig, continuous_ax = plt.subplots(figsize=(fig_width, fig_height))
        continuous_ax.plot(Indices, Signal)
        st.pyplot(continuous_fig)  # continuous
    else:
        dis_fig, dis_ax = plt.subplots(figsize=(fig_width, fig_height))
        dis_ax.stem(Indices, Signal)
        st.pyplot(dis_fig)  # discrete


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
    amp = st.number_input("Insert Amp", value=None, placeholder=0.0)
    phase_shift = st.number_input("Insert Phase shift", value=None, placeholder=0.0)
    freq = st.number_input("Insert Freq", value=None, placeholder=0)
    samplingFreq = st.number_input("Insert Sampling Freq", value=None, placeholder=0)
    # TODO: Handle Error
    # if samplingFreq < 2 * freq:

    clicked = st.button("Show", type="primary")

    fig_width = 10
    fig_height = fig_width / 2

    if clicked:
        s, indices = sp.generate_signal(
            True, False, sp.Signal_type.Time, amp, samplingFreq, freq, phase_shift
        )
        Plot_signal(True, s, indices)
else:
    file_name = st.text_input("File name")
