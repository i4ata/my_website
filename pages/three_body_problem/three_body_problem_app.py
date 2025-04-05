from dash import Input, Output, State, dcc, html, dash_table, ctx, no_update, callback, register_page
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random as rd
import numpy as np

from pages.three_body_problem.three_body_problem import run_euler
from utils import updatemenu

with open('pages/three_body_problem/text.md') as f:
    text = f.readlines()

register_page(__name__, path='/3bp')

layout = html.Div([
    dcc.Markdown(text, mathjax=True),
    html.Div([
        html.Label('Choose dimensions'),
        dcc.RadioItems(id='dims', value=2, options=(2,3))
    ]),
    html.Br(),
    html.Div([
        html.Label('Choose the number of bodies'),
        html.Br(),
        dcc.Input(id='n', type='number', value=3, min=1, step=1)
    ]),
    html.Br(),
    html.Div([
        html.Label("Choose the bodies' parameters"),
        dash_table.DataTable(id='table', editable=True),
        html.Br(),
        html.Button('Reset', id='reset')
    ]),
    html.Br(),
    html.Div([
        html.Label('Step size'),
        html.Br(),
        dcc.Input(id='step', type='number', value=0.005, min=0, step=0.001)
    ]),
    html.Br(),
    html.Div([
        html.Label('Total timesteps'),
        html.Br(),
        dcc.Input(id='max_t', type='number', value=1000, min=0, step=1)
    ]),
    html.Br(),
    html.Div([
        html.Label('G'),
        html.Br(),
        dcc.Input(id='G', type='number', value=1, min=0, step=0.1)
    ]),
    html.Br(),
    html.Div([
        html.Label('Maximum acceleration magnitude'),
        html.Br(),
        dcc.Input(id='max_magnitude', type='number', value=15, min=0, step=1)
    ]),
    html.Br(),
    html.Div([
        html.Label('Animate'),
        dcc.RadioItems(
            id='animate', value=False, 
            options=[
                {'label': 'Static', 'value': False},
                {'label': 'Animated', 'value': True}
            ]
        )
    ]),
    html.Br(),
    html.Button('Submit', id='submit'),
    html.Div(id='trajectories_graph'),
    html.Div(id='values_graph')
])

colors = px.colors.qualitative.Plotly
n_colors = len(colors)

@callback(
    Output('table', 'columns'),
    Output('table', 'data'),
    Input('dims', 'value'),
    Input('n', 'value'),
    Input('reset', 'n_clicks'),
    State('table', 'data')
)
def set_bodies(d, n, n_clicks, data):
    
    if d is None or n is None: return [], []

    cols = ('x', 'y', 'vx', 'vy', 'M') if d == 2 else ('x', 'y', 'z', 'vx', 'vy', 'vz', 'M')
    get_cols = lambda cols: [{'name': col, 'id': col} for col in cols]

    if ctx.triggered_id in (None, 'reset'):
        data = [{k: rd.random() * 2 - 1 if k != 'M' else 1. for k in cols} for i in range(n)]
        return get_cols(cols), data
    
    if ctx.triggered_id == 'n':
        if n <= len(data):
            return no_update, data[:n]
        else:
            data.extend([{k: rd.random() * 2 - 1 if k != 'M' else 1. for k in cols} for i in range(n-len(data))])
            return no_update, data
        
    else: # d
        if d == 2:
            data = [{col: data[i][col] for col in cols} for i in range(len(data))]
            return get_cols(cols), data
        else:
            for i in range(len(data)):
                data[i]['z'] = rd.random() * 2 - 1
                data[i]['vz'] = rd.random() * 2 - 1
            return get_cols(cols), data
    
@callback(
    Output('trajectories_graph', 'children'),
    Output('values_graph', 'children'),
    Input('submit', 'n_clicks'),
    Input('dims', 'value'),
    Input('n', 'value'),
    Input('table', 'data'),
    Input('step', 'value'),
    Input('max_t', 'value'),
    Input('G', 'value'),
    Input('max_magnitude', 'value'),
    Input('animate', 'value')
)
def generate(n_clicks, dims, n, data, step, max_t, G, max_magnitude, animate):

    if ctx.triggered_id != 'submit':
        return None, None
    
    position_cols, velocity_cols = (('x', 'y', 'z')[:dims], ('vx', 'vy', 'vz')[:dims])
    init_position = np.array([[data[i][col] for col in position_cols] for i in range(n)])
    init_velocity = np.array([[data[i][col] for col in velocity_cols] for i in range(n)])
    M = np.array([data[i]['M'] for i in range(n)])
    
    x, v, a = run_euler(
        n=n, 
        d=dims, 
        init_position=init_position, 
        init_velocity=init_velocity,
        M=M,
        step=step,
        max_t=max_t,
        G=G,
        max_magnitude=max_magnitude
    )
    
    fig = _plot(x, animate)
    fig_values = _plot_values(x, v, a)
    
    return dcc.Graph(figure=fig), dcc.Graph(figure=fig_values)

def _plot_values(x: np.ndarray, v: np.ndarray, a: np.ndarray) -> go.Figure:
    
    t, n, d = v.shape
    fig = make_subplots(rows=3*d, shared_xaxes=True, x_title='t')
    xva = np.concatenate((x, v, a), axis=2).transpose(2, 1, 0)
    time = np.arange(t)
    for i, data in enumerate(xva, start=1):
        fig.add_traces(
            [
                go.Scatter(
                    x=time, y=data[body_id], mode='lines', line_color=colors[body_id%n_colors], 
                    hovertemplate=f'Body: {body_id+1}<br>t: %{{x}}<br>value: %{{y}}<br><extra></extra>'
                )
                for body_id in range(n)
            ],
            cols=1,
            rows=i
        )
    fig.update_layout(
        showlegend=False, 
        autosize=True, height=500*d, width=1000,
        title='Simulation parameters'
    )
    for i, name in enumerate(('x', 'y', 'z')[:d] + ('vx', 'vy', 'vz')[:d] + ('ax', 'ay', 'az')[:d], start=1):
        fig.update_yaxes(title=name, row=i)
    return fig

def _plot(x: np.ndarray, animate: bool) -> go.Figure:
    t, n, d = x.shape
    fig = go.Figure()
    scatter = go.Scatter if d == 2 else go.Scatter3d
    iterator = list(zip('xyz', range(d)))
    marker_size = 10 if d == 2 else 2
    fig.add_traces(
        [
            scatter(
                mode='lines', hoverinfo='skip', line_color=colors[body_id%n_colors],
                **{coord: x[:, body_id, i] for coord, i in iterator}
            )
            for body_id in range(n)
        ]
        +
        [
            scatter(
                mode='markers', hoverinfo='skip', marker_color=colors[body_id%n_colors], marker_size=marker_size,
                **{coord: [x[0, body_id, i]] for coord, i in iterator}
            )
            for body_id in range(n)
        ]
    )

    if animate:
        fig.frames = [
            go.Frame(
                data=[
                    scatter(**{coord: x[max(frame-50, 0):frame, body_id, i] for coord, i in iterator}) for body_id in range(n)
                ] + [
                    scatter(**{coord: [x[frame, body_id, i]] for coord, i in iterator}) for body_id in range(n)
                ],
                traces=list(range(n*2))
            )
            for frame in range(t)
        ]

    nbp_menu = updatemenu
    nbp_menu['buttons'][0]['args'][1]['frame']['duration'] = 2 # Set to 2ms per frame
    fig.update_layout(
        title=f'Simulation of {n} bodies in {d}D', 
        showlegend=False, 
        autosize=True, height=1000, width=1000,
        updatemenus=[nbp_menu] if animate else []
    )

    if d == 2:
        fig.update_xaxes(title='x', range=[x[..., 0].min() - 1, x[..., 0].max() + 1], constrain='domain')
        fig.update_yaxes(title='y', range=[x[..., 1].min() - 1, x[..., 1].max() + 1], scaleanchor='x')
    else:
        xaxis = {'title': 'x', 'range': [x[..., 0].min() - 1, x[..., 0].max() + 1]}
        yaxis = {'title': 'y', 'range': [x[..., 1].min() - 1, x[..., 1].max() + 1]}
        zaxis = {'title': 'z', 'range': [x[..., 2].min() - 1, x[..., 2].max() + 1]}
        fig.update_layout(
            scene={
                'xaxis': xaxis, 'yaxis': yaxis, 'zaxis': zaxis, 
                'aspectratio': {'x':1, 'y':1, 'z':1}
            }
        )
    return fig
