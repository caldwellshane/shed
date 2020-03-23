from typing import Tuple, List

import numpy as np
import plotly.graph_objects as go
import qutip as qu


def bloch_spherical_coords(state: qu.Qobj) -> np.ndarray:
    """
    Compute Bloch spherical coordinates (r, θ, ϕ) on any quantum state.

    Uses only the first two amplitudes, reflecting leakage as a shortening of the radius.
    """
    a = state[0, 0]
    b = state[1, 0]
    a_abs, a_angle = np.abs(a), np.angle(a)
    b_abs, b_angle = np.abs(b), np.angle(b)
    r = np.sqrt(a_abs ** 2 + b_abs ** 2)
    θ = 2 * np.arctan2(b_abs, a_abs)
    ϕ = b_angle - a_angle
    return np.array([r, θ, ϕ])


def bloch_vector(state: qu.Qobj) -> np.ndarray:
    """
    Compute Bloch vector (x, y, z) on any quantum state.

    Uses only the first two amplitudes, reflecting leakage as a shortening of the vector.
    """
    arr = bloch_spherical_coords(state)
    r, θ, ϕ = arr[0], arr[1], arr[2]
    return np.array([
        r * np.sin(θ) * np.cos(ϕ),
        r * np.sin(θ) * np.sin(ϕ),
        r * np.cos(θ)
    ])


def add_states(fig: go.Figure, states: List[qu.Qobj], *, mode="markers") -> None:
    """Draw many states to the given Bloch sphere Figure."""
    vectors = np.array([bloch_vector(state) for state in states])
    fig.add_trace(
        go.Scatter3d(
            x=vectors[:, 0],
            y=vectors[:, 1],
            z=vectors[:, 2],
            mode=mode,
            marker=dict(
                color=vectors[:, 2],
                coloraxis="coloraxis",
                size=2
            ),
            hoverinfo="none",
            showlegend=False,
            meta=dict(showscale=False)
        )
    )


def bloch_sphere_figure(*, show_colorbar=True):
    """Produce blank Bloch sphere with basic labels."""
    θ = np.linspace(0, np.pi, 51)
    ϕ = np.linspace(0, 2 * np.pi, 91)
    x = np.outer(np.cos(ϕ), np.sin(θ))
    y = np.outer(np.sin(ϕ), np.sin(θ))
    z = np.outer(np.ones(91), np.cos(θ))

    layout = go.Layout(
        title=None,
        autosize=False,
        width=500,
        height=500,
        margin=go.layout.Margin(l=0, r=0, b=0, t=0),
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
    fig.add_trace(  # z axis
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[-1, 1],
            mode="lines",
            line=dict(color="grey"),
            hoverinfo="none",
            showlegend=False
        )
    )
    fig.add_trace(  # equator
        go.Scatter3d(
            x=np.cos(ϕ),
            y=np.sin(ϕ),
            z=np.zeros(len(ϕ)),
            mode="lines",
            line=dict(color="grey"),
            hoverinfo="none",
            showlegend=False
        )
    )
    fig.add_trace(  # label z = +1 --> "0"
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[1],
            mode="markers+text",
            marker=dict(color="black", size=2),
            text="0",
            hoverinfo="none",
            showlegend=False
        )
    )
    fig.add_trace(  # label z = -1 --> "1"
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[-1],
            mode="markers+text",
            marker=dict(color="black", size=2),
            text="1",
            textposition="bottom center",
            hoverinfo="none",
            showlegend=False
        )
    )
    axis_options = dict(
        showaxeslabels=False,
        showbackground=False,
        showspikes=False,
        showticklabels=False,
    )
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.1, y=1.1, z=0.33)
    )
    layout = dict(
        coloraxis=dict(
            cmin=-1,
            cmax=1,
            colorscale="bluered_r",
            showscale=show_colorbar,
        ),
        scene=dict(
            xaxis=axis_options,
            xaxis_title=None,
            yaxis=axis_options,
            yaxis_title=None,
            zaxis=axis_options,
            zaxis_title=None,
        ),
        scene_camera=camera
    )
    fig.update_layout(layout)
    return fig
