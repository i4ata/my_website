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
    
    ee_true = np.array(ee_true)
    n = len(lengths)
    d = len(ee_true)
    fk = fk_2d if d == 2 else fk_3d
    lengths = np.array(lengths)[np.newaxis]
    predicted_angles = JIT(lengths, ee_true[np.newaxis], save_all=True)
    trajectories = fk(lengths, angles=predicted_angles, save_all=True)

    fig = _plot(trajectories[:d], ee_true)

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

def _plot(x: np.ndarray, ee_true: np.ndarray) -> go.Figure:
    
    d = len(ee_true)
    n_joints = x.shape[2] - 1
    n_iterations = x.shape[1]
    errors = np.linalg.norm(x[:, :, -1].T - ee_true, axis=1)
    
    fig = go.Figure()

    marker_target = {'color': 'orange', 'symbol': 'star' if d==2 else 'diamond', 'size': 15 if d ==2 else 5}
    marker_start = {'color': 'black', 'symbol': 'x', 'size': 10 if d ==2 else 5}
    marker_end = {'color': 'red', 'size': 10 if d ==2 else 5}
    marker_size = None if d==2 else 3
    line_width = None if d==2 else 3

    scatter = go.Scatter if d == 2 else go.Scatter3d
    iterator = list(zip('xyz', range(d)))

    fig.add_traces([

        # LINKS
        # (Intentionally split from lines to do the hovering properly)
        *[
            scatter( 
                mode='lines', line_width=line_width, hoverinfo='skip',
                line_color=_color_palette[joint % len(_color_palette)],
                **{coord: [x[i, -1, joint], x[i, -1, joint+1]] for coord, i in iterator}
            )
            for joint in range(n_joints)
        ],

        # JOINTS
        *[
            scatter(
                mode='markers', marker_size=marker_size,
                hovertemplate=f'Joint {joint}' + _hover_suffix,
                marker_color=_color_palette[joint % len(_color_palette)],
                **{coords: [x[i, -1, joint]] for coords, i in iterator}
            )
            for joint in range(1, n_joints)
        ],

        # END POINT
        scatter(
            mode='markers', marker=marker_end,
            hovertemplate='End' + _hover_suffix,
            **{coord: [x[i, -1, -1]] for coord, i in iterator}
        ),

        # TARGET
        scatter(
            mode='markers', marker=marker_target,
            hovertemplate='Target' + _hover_suffix,
            **{coord: [ee_true[i]] for coord, i in iterator}
        ),

        # START
        scatter(
            mode='markers', marker=marker_start,
            hovertemplate='Joint 0' + _hover_suffix,
            **{coord: [0] for coord, _ in iterator}
        )
    ])

    fig.frames = [
        go.Frame(

            # ANIMATE THE LINES, JOINTS, AND END
            # SADLY WE NEED TO PASS AN ENTIRE SCATTER3D OBJECT INSTEAD OF JUST XYZ AS IN 2D
            data=[
                # Links
                scatter(**{coord: [x[i, frame, joint], x[i, frame, joint+1]] for coord, i in iterator})
                for joint in range(n_joints)
            ] + [
                # Joints
                scatter(**{coord: [x[i, frame, joint]] for coord, i in iterator})
                for joint in range(1, n_joints)
            ] + [
                # End
                scatter(**{coord: [x[i, frame, -1]] for coord, i in iterator})
            ],
            traces=list(range(2 * n_joints)), name=str(frame), 
            layout={'title': f'Iteration: {frame} | Error: {errors[frame]:.4f}'}
        )
        for frame in range(n_iterations)
    ]

    fig.update_layout(
        autosize=True,
        height=800,
        width=800,
        title={'text': f'Iterations: {n_iterations} | Error: {errors[-1]:.4f}'},
    )

    axis_range = x.min() - .2, x.max() + .2
    if d == 2:
        fig.update_xaxes(title='x', range=axis_range, constrain='domain')
        fig.update_yaxes(title='y', range=axis_range, scaleanchor='x')
    else:
        tick_vals = np.linspace(math.floor(axis_range[0]), math.ceil(axis_range[1]), 5)
        tick_text = list(map(str, tick_vals))
        axes = {'range': axis_range, 'tickvals': tick_vals, 'ticktext': tick_text}
        scene = {'xaxis': axes, 'yaxis': axes, 'zaxis': axes, 'aspectratio': {'x':1, 'y':1, 'z': 1}}
        fig.update_layout(scene=scene)

    return fig
