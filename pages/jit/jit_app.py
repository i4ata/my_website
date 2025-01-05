from dash import Input, Output, State, dcc, html, ALL, dash_table, callback, register_page
import numpy as np
import plotly.graph_objects as go
import math

from pages.jit.jit import fk_2d, fk_3d, JIT

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
    
    fk = fk_2d if len(ee_true) == 2 else fk_3d
    lengths = np.array(lengths)[np.newaxis]
    predicted_angles = JIT(lengths, np.array(ee_true)[np.newaxis], save_all=True)
    trajectories = fk(lengths, angles=predicted_angles, save_all=True)

    fig = (
        _plot_2d(x=trajectories[0], y=trajectories[1], ee_true=ee_true)
        if len(ee_true) == 2 else
        _plot_3d(x=trajectories[0], y=trajectories[1], z=trajectories[2], ee_true=ee_true)
    )

    fig.update_layout(
        showlegend=False,
        updatemenus=[_updatemenu],
        sliders=[_slider(len(trajectories[0]))]
    )

    columns = [{'name': col, 'id': col} for col in ['Angle'] + [f'Joint {i}' for i in range(len(lengths[0]))]]
    if len(ee_true) == 2:
        angles_2d = predicted_angles[-1]
        data = [{'Angle': 'θ'} | {f'Joint {i}': round(angles_2d[i], 4) for i in range(len(lengths[0]))}]
    else:
        angles_3d = predicted_angles[-1].reshape(-1, 2)
        data = [
            {'Angle': 'θ'} | {f'Joint {i}': round(angles_3d[i, 0], 4) for i in range(len(lengths[0]))},
            {'Angle': 'φ'} | {f'Joint {i}': round(angles_3d[i, 1], 4) for i in range(len(lengths[0]))}
        ]
    table = dash_table.DataTable(data=data, columns=columns)
    
    return (
        dcc.Graph(figure=fig), 
        html.Div([
            html.Label('Found optimal angles (in radians)'),
            table
        ])
    )

_hover_suffix = '<br>x: %{x}<br>y: %{y}<extra></extra>'

def _plot_2d(x: np.ndarray, y: np.ndarray, ee_true = np.ndarray) -> go.Figure:
    
    n_joints = x.shape[1] - 1
    fig = go.Figure()
    errors = np.linalg.norm(np.stack((x[:, -1], y[:, -1]), axis=1) - np.array(ee_true), axis=1)

    marker_target = dict(
        color='orange',
        symbol='star',
        size=15
    )

    marker_start=dict(
        color='black',
        symbol='x',
        size=10
    )

    marker_end = dict(
        color='red',
        size=10
    )

    fig.add_traces([
        *[
            go.Scatter(
                x=[x[-1, joint], x[-1, joint+1]], y=[y[-1, joint], y[-1, joint+1]], 
                mode='lines+markers', name='Arm', 
                hovertemplate=f'Joint {joint}' + _hover_suffix
            )
            for joint in range(n_joints)
        ],
        go.Scatter(
            x=[x[-1, -1]], y=[y[-1, -1]], name='End', mode='markers', marker=marker_end,
            hovertemplate='End' + _hover_suffix
        ),
        go.Scatter(
            x=[ee_true[0]], y=[ee_true[1]], name='Target', mode='markers', marker=marker_target,
            hovertemplate='Target' + _hover_suffix
        ),
        go.Scatter(
            x=[0], y=[0], name='Start', mode='markers', marker=marker_start,
            hovertemplate='Joint 0' + _hover_suffix
        )
    ])

    fig.frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=[x[frame, joint], x[frame, joint+1]], 
                    y=[y[frame, joint], y[frame, joint+1]]
                ) 
                for joint in range(n_joints)
            ] + [
                go.Scatter(x=[x[frame, -1]], y=[y[frame, -1]])
            ],
            traces=list(range(n_joints + 1)), name=str(frame), 
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
        title={'text': f'Iterations: {len(x)} | Error: {errors[-1]:.4f}'}
    )

    return fig

def _plot_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, ee_true = np.ndarray) -> go.Figure:
    
    n_joints = x.shape[1] - 1
    fig = go.Figure()
    errors = np.linalg.norm(
        np.stack((x[:, -1], y[:, -1], z[:, -1]), axis=1) - np.array(ee_true), 
        axis=1
    )

    marker_target = dict(
        color='orange',
        symbol='diamond',
        size=5
    )

    marker_start=dict(
        color='black',
        symbol='x',
        size=5
    )

    marker_end = dict(
        color='red',
        size=5
    )

    fig.add_traces([
        *[
            go.Scatter3d(
                x=[x[-1, joint], x[-1, joint+1]], 
                y=[y[-1, joint], y[-1, joint+1]],
                z=[z[-1, joint], z[-1, joint+1]], 
                mode='lines+markers', name='Arm',
                marker_size=3, line_width=3,
                hovertemplate=f'Joint {joint}' + _hover_suffix
            )
            for joint in range(n_joints)
        ],
        go.Scatter3d(
            x=[x[-1, -1]], y=[y[-1, -1]], z=[z[-1, -1]], 
            name='End', mode='markers', marker=marker_end,
            hovertemplate='End' + _hover_suffix
        ),
        go.Scatter3d(
            x=[ee_true[0]], y=[ee_true[1]], z=[ee_true[2]], 
            name='Target', mode='markers', marker=marker_target,
            hovertemplate='Target' + _hover_suffix
        ),
        go.Scatter3d(
            x=[0], y=[0], z=[0],
            name='Start', mode='markers', marker=marker_start,
            hovertemplate='Joint 0' + _hover_suffix
        )
    ])

    fig.frames = [
        go.Frame(
            data=[
                go.Scatter3d(
                    x=[x[frame, joint], x[frame, joint+1]], 
                    y=[y[frame, joint], y[frame, joint+1]],
                    z=[z[frame, joint], z[frame, joint+1]]
                ) 
                for joint in range(n_joints)
            ] + [
                go.Scatter3d(
                    x=[x[frame, -1]], 
                    y=[y[frame, -1]],
                    z=[z[frame, -1]]
                )
            ],
            traces=list(range(n_joints + 1)), name=str(frame), 
            layout={'title': f'Iteration: {frame} | Error: {errors[frame]:.4f}'}
        )
        for frame in range(len(x))
    ]

    axis_range = min(x.min(), y.min(), z.min()) - .2, max(x.max(), y.max(), z.max()) + .2
    tick_vals = np.linspace(math.floor(axis_range[0]), math.ceil(axis_range[1]), 5)
    tick_text = list(map(str, tick_vals))
    axes = dict(
        range=axis_range,
        tickvals=tick_vals,
        ticktext=tick_text
    )
    fig.update_layout(
        scene=dict(
            xaxis=axes, yaxis=axes, zaxis=axes, 
            aspectratio={'x':1, 'y':1, 'z':1},
            camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        autosize=True,
        height=800,
        width=800,
        title={'text': f'Iterations: {len(x)} | Error: {errors[-1]:.4f}'}
    )

    return fig

def _slider(n: int) -> dict:
    return {
        'active': 0,
        'currentvalue': {'prefix': 'Iteration: '},
        'len': 0.9,
        'pad': {'b': 10, 't': 60},
        'steps': [
            {
                'args': [
                    [str(i)],  
                    {
                        'frame': {'duration': 0, 'redraw': True}, 
                        'mode': 'immediate',
                        'fromcurrent': True,
                        'transition': {'duration': 0, 'easing': 'linear'}
                    }
                ],
                'label': str(i),
                'method': 'animate',
            }
            for i in range(n)
        ],
        'active': 0,
        'x': 0.1,
        'xanchor': 'left',
        'y': 0,
        'yanchor': 'top'
    }

_play_button = {
    'label': 'Play', 
    'method': 'animate',
    'args': [
        None, 
        {
            # CHANGED TO 750 TO AVOID BUG WHERE THE ANIMATION GOES FROM THE FIRST FRAME TO THE LAST
            # PLAY AROUND WITH THE TIMING MAYBE TO ENSURE THE SMOOTHNESS TODO
            'frame': {'duration': 1000},#, 'redraw': True},
            'mode': 'next', 
            # 'fromcurrent': True,
            'transition': {'duration': 500, 'easing': 'linear'}
        }
    ]
}

_pause_button = {
    'label': 'Pause',
    'method': 'animate',
    'args': [
        [None], 
        {
            'frame': {'duration': 0, 'redraw': True},
            'mode': 'immediate', 
            'fromcurrent': True, 
            'transition': {'duration': 0, 'easing': 'linear'}
        }
    ]
}

_updatemenu = {
    'buttons': [_play_button, _pause_button],
    'direction': 'left',
    'pad': {'r': 10, 't': 70},
    'showactive': False,
    'type': 'buttons',
    'x': 0.1,
    'xanchor': 'right',
    'y': 0,
    'yanchor': 'top'            
}
