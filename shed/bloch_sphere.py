from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import qutip as qu


def bloch_spherical_coords(state: qu.Qobj) -> Tuple[float, float, float]:
    """
    Compute Bloch spherical coordinates (r, ) on any quantum state.

    Uses only the first two amplivector (x, y, ztudes, reflecting leakage as a shortening of the vector.
    """
    a = state[0, 0]
    b = state[1, 0]
    a_abs, a_angle = np.abs(a), np.angle(a)
    b_abs, b_angle = np.abs(b), np.angle(b)
    r = np.sqrt(a_abs ** 2 + b_abs ** 2)
    θ = 2 * np.arctan2(b_abs, a_abs)
    ϕ = b_angle - a_angle
    return r, θ, ϕ


def bloch_vector(state: qu.Qobj) -> Tuple[float, float, float]:
    """
    Compute Bloch vector (x, y, z) on any quantum state.

    Uses only the first two amplitudes, reflecting leakage as a shortening of the vector.
    """
    r, θ, ϕ = bloch_spherical_coords(state)
    return (
        r * np.sin(θ) * np.cos(ϕ),
        r * np.sin(θ) * np.sin(ϕ),
        r * np.cos(θ)
    )


def add_vector(fig: go.Figure, vec: Tuple[float, float, float]) -> None:
    """Draw the given vector on the given Figure."""
    fig.add_trace(
        go.Scatter3d(
            x=[vec[0]],
            y=[vec[1]],
            z=[vec[2]],
            mode="markers",
            marker=dict(
                color=[vec[2]],
                coloraxis="coloraxis",
                size=2
            ),
            hoverinfo="none",
            showlegend=False,
            meta=dict(showscale=False)
        )
    )


def bloch_sphere_figure():
    """Produce blank Bloch sphere with basic labels."""
    θ = np.linspace(0, 2 * np.pi, 100)
    ϕ = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(θ), np.sin(ϕ))
    y = np.outer(np.sin(θ), np.sin(ϕ))
    z = np.outer(np.ones(100), np.cos(ϕ))  # note this is 2d now

    layout = go.Layout(
        title='Bloch sphere',
        autosize=False,
        width=500,
        height=500,
        margin=go.layout.Margin(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    fig = go.Figure(layout=layout)
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            showscale=False,
            colorscale=[[0, "grey"], [1, "grey"]],
            opacity=0.25,
            surfacecolor=0 * np.zeros(len(z)),
            hoverinfo="none"
        )
    )

    ϕs = np.linspace(0, 2 * np.pi, 401)
    fig.add_trace(
        go.Scatter3d(
            x=np.cos(ϕs),
            y=np.sin(ϕs),
            z=np.zeros(len(ϕs)),
            mode="lines",
            line={"color": "grey"},
            hoverinfo="none",
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[1],
            mode="markers+text",
            marker={"color": "black", "size": 2},
            text=r"0",
            hoverinfo="none",
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[-1],
            mode="markers+text",
            marker={"color": "black", "size": 2},
            text=r"1",
            textposition="bottom center",
            hoverinfo="none",
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[1],
            y=[0],
            z=[0],
            mode="markers+text",
            marker={"color": "black", "size": 2},
            text=r"x",
            hoverinfo="none",
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[1],
            z=[0],
            mode="markers+text",
            marker={"color": "black", "size": 2},
            text=r"y",
            hoverinfo="none",
            showlegend=False
        )
    )
    fig.update_layout(
        {
            "coloraxis": {
                "cmin": -1,
                "cmax": 1,
                "colorscale": "bluered_r",
            },
        }
    )
    return fig
