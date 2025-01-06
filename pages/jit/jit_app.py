from dash import Input, Output, State, dcc, html, ALL, dash_table, callback, register_page
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import math

from pages.jit.jit import fk_2d, fk_3d, JIT
from utils import slider, updatemenu

with open('pages/jit/text.md') as f:
    text = f.read()

register_page(__name__)

layout = html.Div([
    dcc.Markdown(text, mathjax=True),
    html.Div([
        html.Label('Choose dimensions of the system'),
        dcc.RadioItems(id='d', options=(2, 3), value=2)
    ]),
    html.Br(),
    html.Div([
        html.Label('Choose desired end effector position'),
        html.Div(id='ee_true_container'),
    ]),
    html.Br(),
    html.Div([
        html.Label('Choose the number of joints'),
        html.Br(),
        dcc.Input(id='n', type='number', value=4, min=1, step=1)
    ]),
    html.Br(),
    html.Div(id='lengths_container'),
    html.Br(),
    html.Button('Submit', id='submit'),
    html.Div(id='graph_jit'),
    html.Div(id='angles_data_container')
])

@callback(
    Output('ee_true_container', 'children'),
    Input('d', 'value')
)
def set_dimensions(d):
    labels = {0: 'x', 1: 'y', 2: 'z'}
    inputs = [
        html.Div([
            html.Label(labels[i] + ': '),
            dcc.Input(id={'type': 'ee_true', 'index': i}, type='number', value=1)
        ])
        for i in range(d)
    ]
    return inputs

@callback(
    Output('lengths_container', 'children'),
    Input('n', 'value')
)
def set_length(n):
    inputs = [html.Label('Choose the lengths of the links')] + [
        html.Div([
            html.Label(f'Link {i}: '),
            dcc.Input(id={'type': 'length', 'index': i}, type='number', value=1, min=0)
        ])
        for i in range(n)
    ]
    return inputs

@callback(
    Output('graph_jit', 'children'),
    Output('angles_data_container', 'children'),
    Input('submit', 'n_clicks'),
    State({'type': 'length', 'index': ALL}, 'value'),
    State({'type': 'ee_true', 'index': ALL}, 'value')
)
def generate(n_clicks, lengths, ee_true):
    if n_clicks is None:
        return None, None
    
    n = len(lengths)
    d = len(ee_true)
    fk = fk_2d if d == 2 else fk_3d
    lengths = np.array(lengths)[np.newaxis]
    predicted_angles = JIT(lengths, np.array(ee_true)[np.newaxis], save_all=True)
    trajectories = fk(lengths, angles=predicted_angles, save_all=True)

    fig = (
        _plot_2d(x=trajectories[0], y=trajectories[1], ee_true=ee_true)
        if d == 2 else
        _plot_3d(x=trajectories[0], y=trajectories[1], z=trajectories[2], ee_true=ee_true)
    )

    updatemenu['buttons'][0]['args'][1] = {
        'frame': {'duration': 1000},
        'mode': 'next', 
        'transition': {'duration': 500, 'easing': 'linear'}
    }
    fig.update_layout(
        showlegend=False,
        updatemenus=[updatemenu],
        sliders=[slider(len(trajectories[0]))]
    )

    columns = [{'name': col, 'id': col} for col in ['Angle'] + [f'Joint {i}' for i in range(n)]]
    if d == 2:
        angles_2d = predicted_angles[-1]
        data = [{'Angle': 'θ'} | {f'Joint {i}': round(angles_2d[i], 4) for i in range(n)}]
    else:
        angles_3d = predicted_angles[-1].reshape(-1, 2)
        data = [
            {'Angle': 'θ'} | {f'Joint {i}': round(angles_3d[i, 0], 4) for i in range(n)},
            {'Angle': 'φ'} | {f'Joint {i}': round(angles_3d[i, 1], 4) for i in range(n)}
        ]

    return (
        dcc.Graph(figure=fig, id=f'graph_{d}d'), 
        html.Div([
            html.Label('Found optimal angles (in radians)'),
            dash_table.DataTable(data=data, columns=columns)
        ])
    )

_hover_suffix = '<br>x: %{x}<br>y: %{y}<extra></extra>'
_color_palette = px.colors.qualitative.Plotly

def _plot_2d(x: np.ndarray, y: np.ndarray, ee_true = np.ndarray) -> go.Figure:
    
    n_joints = x.shape[1] - 1
    n_iterations = len(x)
    errors = np.linalg.norm(
        np.stack((x[:, -1], y[:, -1]), axis=1) - np.array(ee_true), 
        axis=1
    )

    fig = go.Figure()

    marker_target = {'color': 'orange', 'symbol': 'star', 'size': 15}
    marker_start = {'color': 'black', 'symbol': 'x', 'size': 10}
    marker_end = {'color': 'red', 'size': 10}

    fig.add_traces([

        # LINKS
        # (Intentionally split from lines to do the hovering properly)
        *[
            go.Scatter(
                x=[x[-1, joint], x[-1, joint+1]], y=[y[-1, joint], y[-1, joint+1]], 
                mode='lines', 
                hoverinfo='skip',
                line_color=_color_palette[joint % len(_color_palette)]
            )
            for joint in range(n_joints)
        ],

        # JOINTS TODO MAYBE MAKE THEM THE SAME COLOR AS THE LINES
        *[
            go.Scatter(
                x=[x[-1, joint]], y=[y[-1, joint]],
                mode='markers',
                hovertemplate=f'Joint {joint}' + _hover_suffix,
                marker_color=_color_palette[joint % len(_color_palette)]
            )
            for joint in range(1, n_joints)
        ],

        # END POINT
        go.Scatter(
            x=[x[-1, -1]], y=[y[-1, -1]], name='End', mode='markers', marker=marker_end,
            hovertemplate='End' + _hover_suffix
        ),

        # TARGET
        go.Scatter(
            x=[ee_true[0]], y=[ee_true[1]], name='Target', mode='markers', marker=marker_target,
            hovertemplate='Target' + _hover_suffix
        ),

        # START
        go.Scatter(
            x=[0], y=[0], name='Start', mode='markers', marker=marker_start,
            hovertemplate='Joint 0' + _hover_suffix
        )
    ])

    fig.frames = [
        go.Frame(
            # ANIMATE THE LINES, JOINTS, AND END
            data=[
                go.Scatter(
                    x=[x[frame, joint], x[frame, joint+1]], 
                    y=[y[frame, joint], y[frame, joint+1]]
                ) 
                for joint in range(n_joints)
            ] + [
                go.Scatter(x=[x[frame, joint]], y=[y[frame, joint]])
                for joint in range(1, n_joints)
            ] + [
                go.Scatter(x=[x[frame, -1]], y=[y[frame, -1]])
            ],
            traces=list(range(2 * n_joints)), name=str(frame), 
            layout={'title': f'Iteration: {frame} | Error: {errors[frame]:.4f}'}
        )
        for frame in range(len(x))
    ]

    axis_range = min(x.min(), y.min()) - .2, max(x.max(), y.max()) + .2
    fig.update_xaxes(title='x', range=axis_range, constrain='domain')
    fig.update_yaxes(title='y', range=axis_range, scaleanchor='x')
    fig.update_layout(
        autosize=True,
        height=800,
        width=800,
        title={'text': f'Iterations: {n_iterations} | Error: {errors[-1]:.4f}'}
    )

    return fig

def _plot_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, ee_true = np.ndarray) -> go.Figure:
    
    n_joints = x.shape[1] - 1
    n_iterations = len(x)
    errors = np.linalg.norm(
        np.stack((x[:, -1], y[:, -1], z[:, -1]), axis=1) - np.array(ee_true), 
        axis=1
    )
    
    fig = go.Figure()
    
    marker_target = {'color': 'orange', 'symbol': 'diamond', 'size': 5}
    marker_start = {'color': 'black', 'symbol': 'x', 'size': 5}
    marker_end = {'color': 'red', 'size': 5}

    fig.add_traces([
        
        # LINKS
        # (Intentionally split from lines to do the hovering properly)
        *[
            go.Scatter3d(
                x=[x[-1, joint], x[-1, joint+1]], 
                y=[y[-1, joint], y[-1, joint+1]],
                z=[z[-1, joint], z[-1, joint+1]], 
                mode='lines', line_width=3, hoverinfo='skip',
                line_color=_color_palette[joint % len(_color_palette)]
            )
            for joint in range(n_joints)
        ],
        
        # JOINTS TODO MAYBE MAKE THEM THE SAME COLOR AS THE LINES
        *[
            go.Scatter3d(
                x=[x[-1, joint]], y=[y[-1, joint]], z=[z[-1, joint]], 
                mode='markers', marker_size=3,
                hovertemplate=f'Joint {joint}' + _hover_suffix,
                marker_color=_color_palette[joint % len(_color_palette)]
            )
            for joint in range(1, n_joints)
        ],

        # END
        go.Scatter3d(
            x=[x[-1, -1]], y=[y[-1, -1]], z=[z[-1, -1]], 
            name='End', mode='markers', marker=marker_end,
            hovertemplate='End' + _hover_suffix
        ),

        # TARGET
        go.Scatter3d(
            x=[ee_true[0]], y=[ee_true[1]], z=[ee_true[2]], 
            name='Target', mode='markers', marker=marker_target,
            hovertemplate='Target' + _hover_suffix
        ),

        # START
        go.Scatter3d(
            x=[0], y=[0], z=[0],
            name='Start', mode='markers', marker=marker_start,
            hovertemplate='Joint 0' + _hover_suffix
        )
    ])

    fig.frames = [
        go.Frame(

            # ANIMATE THE LINES, JOINTS, AND END
            data=[
                go.Scatter3d(
                    x=[x[frame, joint], x[frame, joint+1]], 
                    y=[y[frame, joint], y[frame, joint+1]],
                    z=[z[frame, joint], z[frame, joint+1]]
                ) 
                for joint in range(n_joints)
            ] + [
                go.Scatter3d(x=[x[frame, joint]], y=[y[frame, joint]], z=[z[frame, joint]])
                for joint in range(1, n_joints)
            ] + [
                go.Scatter3d(
                    x=[x[frame, -1]], 
                    y=[y[frame, -1]],
                    z=[z[frame, -1]]
                )
            ],
            traces=list(range(2 * n_joints)), name=str(frame), 
            layout={'title': f'Iteration: {frame} | Error: {errors[frame]:.4f}'}
        )
        for frame in range(n_iterations)
    ]

    axis_range = min(x.min(), y.min(), z.min()) - .2, max(x.max(), y.max(), z.max()) + .2
    tick_vals = np.linspace(math.floor(axis_range[0]), math.ceil(axis_range[1]), 5)
    tick_text = list(map(str, tick_vals))
    axes = {'range': axis_range, 'tickvals': tick_vals, 'ticktext': tick_text}

    fig.update_layout(
        scene={
            'xaxis': axes, 'yaxis': axes, 'zaxis': axes, 
            'aspectratio': {'x':1, 'y':1, 'z':1}
        },
        autosize=True,
        height=800,
        width=800,
        title={'text': f'Iterations: {n_iterations} | Error: {errors[-1]:.4f}'}
    )

    return fig
