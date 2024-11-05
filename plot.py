import plotly.graph_objects as go
import streamlit as st
import signalProcessing as sp


class Plot_Fig:
    def __init__(self, title: str = ""):
        self.fig = go.Figure()
        self.title = title

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
    def __init__(self, title: str = ""):
        Plot_Fig.__init__(self, title)

    def scatter(self, legend: str = "", color: str = "", indices=[], amplitudes=[]):
        ln = 50 if len(indices) > 50 else len(indices)
        for i in range(ln):
            # scatter stims
            self.fig.add_trace(
                go.Scatter(
                    x=[indices[i], indices[i]],
                    y=[0, amplitudes[i]],
                    line=dict(color=color) if color != "" else None,
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
    def __init__(self, title: str = ""):
        Plot_Fig.__init__(self, title)

    def scatter(self, legend: str = "", color: str = "", indices=[], amplitudes=[]):
        # Add a continuous line for the signal
        ln = 50 if len(indices) > 50 else len(indices)
        self.fig.add_trace(
            go.Scatter(
                x=indices[:ln],
                y=amplitudes[:ln],
                mode="lines",
                line=dict(color=color) if color != "" else None,
                name=legend,
            )
        )


def plot_signal(sig: sp.Signal):
    if sig:
        disc_fig = Disc_Fig("Discreate Signal")
        disc_fig.scatter("signal", "red", sig.indices, sig.amplitudes)
        disc_fig.update_layout()
        disc_fig.plot()

        cont_fig = Cont_Fig("Continuous Signal")
        cont_fig.scatter("signal", "blue", sig.indices, sig.amplitudes)
        cont_fig.update_layout()
        cont_fig.plot()
