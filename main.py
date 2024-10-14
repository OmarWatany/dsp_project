from enum import Enum
import streamlit as st
import matplotlib.pyplot as plt
import math


class Signal_type(Enum):
    Time = 0
    Freq = 1


def generate_signal(
    sin_flag: bool,
    periodic: bool,
    signal_type: Signal_type,
    amp,
    sampling_freq,
    analog_freq,
    phase_shift,
):
    w = 2 * math.pi * (analog_freq / sampling_freq)
    if sin_flag:
        s = [amp * math.sin(w * i + phase_shift) for i in range(0, sampling_freq)]
    else:
        s = [amp * math.cos(w * i + phase_shift) for i in range(0, sampling_freq)]
    return s, [i for i in range(0, sampling_freq)]


s, indices = generate_signal(
    True, False, Signal_type.Time, 3, 720, 360, 1.96349540849362
)
# c = generate_signal(False, False, Signal_type.Time, 3, 500, 200, 2.35619449019235)

st.write("""
# DSP Framwork
""")


fig_width = 10
fig_height = fig_width / 2

continuous_fig, continuous_ax = plt.subplots(figsize=(fig_width, fig_height))
continuous_ax.plot(indices, s)
# Plotting the signal
st.pyplot(continuous_fig)  # continuous

dis_fig, dis_ax = plt.subplots(figsize=(fig_width, fig_height))
dis_ax.stem(indices, s)
st.pyplot(dis_fig)  # discrete
