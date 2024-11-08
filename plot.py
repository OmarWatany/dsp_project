import plotly.graph_objects as go
import streamlit as st
import signalProcessing as sp


class Plot_Fig:
    def __init__(self, title: str = "", nPts: int = 50):
        self.fig = go.Figure()
        self.title = title
        self.nPts = nPts

    def scatter(self, indices=[], amplitudes=[]):
        pass

    def update_layout(self, xtitle="Index", ytitle="Amplitude"):
        self.fig.update_layout(
            title=self.title,
            xaxis_title=xtitle,
            yaxis_title=ytitle,
            width=1000,
            height=500,
        )

    def plot(self):
        st.plotly_chart(self.fig)


class Disc_Fig(Plot_Fig):
    def __init__(self, title: str = "", nPts: int = 50):
        Plot_Fig.__init__(self, title, nPts)

    def scatter(self, legend: str = "", color=None, indices=[], amplitudes=[]):
        ln = self.nPts if len(indices) > self.nPts else len(indices)
        for i in range(ln):
            # scatter stims
            self.fig.add_trace(
                go.Scatter(
                    x=[indices[i], indices[i]],
                    y=[0, amplitudes[i]],
                    line=dict(color=color) if color else color,
                    mode="lines",
                    showlegend=False,
                )
            )

        self.fig.add_trace(
            go.Scatter(
                x=indices[:ln],
                y=amplitudes[:ln],
                mode="markers",
                marker=dict(color=color, size=8),
                name=legend,
            )
        )


class Cont_Fig(Plot_Fig):
    def __init__(self, title: str = "", nPts: int = 50):
        Plot_Fig.__init__(self, title, nPts)

    def scatter(self, legend: str = "", color=None, indices=[], amplitudes=[]):
        # Add a continuous line for the signal
        ln = self.nPts if len(indices) > self.nPts else len(indices)
        self.fig.add_trace(
            go.Scatter(
                x=indices[:ln],
                y=amplitudes[:ln],
                mode="lines",
                line=dict(color=color) if color else None,
                name=legend,
            )
        )


def plot_freq_signal(sig, nPts: int = 50):
    indices = sp.signal_idx(sig)
    disc_fig = Disc_Fig("Frequencies", nPts)
    disc_fig.scatter("signal", "blue", indices, sp.signal_samples(sig))
    disc_fig.update_layout()
    disc_fig.plot()

    disc_fig = Disc_Fig("Phase Shifts", nPts)
    disc_fig.scatter("signal", "red", indices, sp.signal_phase_shifts(sig))
    disc_fig.update_layout()
    disc_fig.plot()


def plot_time_signal(sig, nPts: int = 50):
    idx = sp.signal_idx(sig)
    s = sp.signal_samples(sig)
    cont_fig = Cont_Fig("Continuous", nPts)
    cont_fig.scatter("signal", "blue", idx, s)
    cont_fig.update_layout()
    cont_fig.plot()
    disc_fig = Disc_Fig("Discrete", nPts)
    disc_fig.scatter("signal", "red", idx, s)
    disc_fig.update_layout()
    disc_fig.plot()


def plot_signal(sig):
    max = False
    nPts = 50
    nPts = st.number_input("Number of pointes to plot:", min_value=1, value=50)
    max = st.checkbox("MAX", value=False)
    plot_func = {
        sp.Signal_type.TIME: plot_time_signal,
        sp.Signal_type.FREQ: plot_freq_signal,
    }
    if sig:
        plot_func[sig["signal_type"]](sig, len(sig.keys()) if max else nPts)


def draw_quantization(quantized_signal, org_signal, error, show_error):
    cont_fig = Cont_Fig("Quantized Signal and Error")
    org_indices = sp.signal_idx(org_signal)
    cont_fig.scatter(
        "Original Signal", "blue", org_indices, sp.signal_samples(org_signal)
    )
    cont_fig.scatter("Quantized Signal", None, org_indices, quantized_signal)
    if show_error:
        cont_fig.scatter("Quantized Error", "red", org_indices, error)
    cont_fig.update_layout()
    cont_fig.plot()
