import signalProcessing as sp
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def Plot_signal_cont(fig_cont, Title: str, Color: str, Signal, Indices):
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


def Plot_signal_stem(fig_disc, Title: str, Color: str, Signal, Indices):
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
        s, indices = sp.generate_signal(
            True, False, sp.Signal_type.Time, amp, samplingFreq, freq, phase_shift
        )
        Plot_signal_cont(True, s, indices)
        Plot_signal_stem(True, s, indices)
else:
    file_name = st.text_input("File name")


# Temporary
s, s_indices = sp.generate_signal(
    True, False, sp.Signal_type.Time, 3, 720, 360, 1.96349540849362
)

c, c_indices = sp.generate_signal(
    False, False, sp.Signal_type.Time, 3, 500, 200, 2.35619449019235
)

# showing sin and cosine signals at the same time
cont_fig = go.Figure()
disc_fig = go.Figure()

Plot_signal_cont(cont_fig, "Sin Wave", "blue", s, s_indices)
Plot_signal_stem(disc_fig, "Sin Wave", "blue", s, s_indices)

Plot_signal_cont(cont_fig, "Cos Wave", "red", c, s_indices)
Plot_signal_stem(disc_fig, "Cos Wave", "red", c, c_indices)

st.plotly_chart(cont_fig)
st.plotly_chart(disc_fig)
