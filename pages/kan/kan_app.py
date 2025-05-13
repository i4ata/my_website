from dash import Input, Output, State, html, dcc, ALL, register_page, callback, ctx
import plotly.graph_objects as go
import dash_cytoscape as cyto
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Literal, Dict, Tuple
import pickle

from pages.kan.kan import BSpline, MSE

with open('pages/kan/text.md') as f:
    text = f.read().split('<!-- INTERACTION -->')

with open('assets/kan/nn.pkl', 'rb') as f:
    kan_layers: List[Tuple[np.ndarray, int, int]] = pickle.load(f)
    n_layers = len(kan_layers)

register_page(__name__, path='/kan', name='Kolmogorov-Arnold Networks', order=3)

def get_training() -> go.Figure:
    preds = np.squeeze(np.load('assets/kan/y_pred.npy'))
    train = np.squeeze(np.load('assets/kan/y_train.npy'))
    x = np.linspace(-1, 1, len(preds))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=preds, mode='lines', name='Predictions'))
    fig.add_trace(go.Scatter(x=x, y=train, mode='markers', name='Ground Truth'))
    loss = MSE().forward(train, preds)
    fig.update_layout(
        title=f'KAN Learning Evaluation<br>MSE: {loss:.4f}',
        xaxis_title='x',
        yaxis_title='f(x)'
    )
    return fig

def get_kan():
    data = []
    for i, (w, _, _) in enumerate(kan_layers):
        if i == 0:
            data.extend([
                {'data': {'id': f'-1_{j}', 'label': f'Input {j+1}'}}
                for j in range(len(w))
            ])
        data.extend([
            {'data': {'id': f'{i}_{j}', 'label': None if i != n_layers - 1 else f'Output {j}'}}
            for j in range(w.shape[1])
        ])
        data.extend([
            {'data': {'source': source, 'target': target}}
            for source in [x['data']['id'] for x in data if 'id' in x['data'] and x['data']['id'].startswith(str(i-1))]
            for target in [x['data']['id'] for x in data if 'id' in x['data'] and x['data']['id'].startswith(str(i))]
        ])
    return data

layout = html.Div([
    dcc.Markdown(text[0], mathjax=True, link_target='_blank'),
    html.Div([
        html.Label('Number of basis elements'),
        html.Br(),
        dcc.Input(id='n', type='number', value=7, min=1, step=1)
    ]),
    html.Br(),
    html.Div([
        html.Label('B-Spline degree'),
        html.Br(),
        dcc.Input(id='p', type='number', value=3, min=1, step=1)
    ]),
    html.Br(),
    html.Div(id='weights_container'),
    html.Button('Random', id='random_weights'),
    dcc.Graph(id='basis_elements'),
    html.Label(id='error'),
    dcc.Markdown(text[1], mathjax=True, link_target='_blank'),
    dcc.Graph(figure=get_training()),
    dcc.Markdown(text[2], mathjax=True, link_target='_blank'),
    cyto.Cytoscape(
        id='nn',
        elements=get_kan(),
        userZoomingEnabled=False,
        userPanningEnabled=False,
        layout={
            'name': 'breadthfirst',
            'directed': True,
        }
    ),
    dcc.Graph(id='spline_graph', style={'display': 'none'})
])

@callback(
    Output('spline_graph', 'figure'),
    Output('spline_graph', 'style'),
    Input('nn', 'tapEdgeData')
)
def plot_spline(data: Dict[Literal['source', 'target'], str]):
    if data is None: return {}, {'display': 'none'}
    
    source_layer, source_node = map(int, data['source'].split('_'))
    target_node = int(data['target'].split('_')[1])

    spline = BSpline(p=kan_layers[source_layer+1][2], n=kan_layers[source_layer+1][1])
    weights = kan_layers[source_layer+1][0][source_node, target_node][:, np.newaxis]

    xs = np.linspace(-1, 1, 500)
    ys = np.sum(spline(xs) * weights, axis=0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=xs, y=ys, mode='lines', name='spline')
    )
    line = {'line_dash': 'dash', 'opacity': .5}
    fig.add_vline(-1, **line)
    fig.add_vline(1, **line)
    fig.update_xaxes(range=[-1.1, 1.1])
    fig.update_layout(
        title=f'B-Spline between unit {source_node+1} in layer {source_layer+2} and unit {target_node+1} in layer {source_layer+3}',
        xaxis_title='x',
        yaxis_title='spline(x)'
    )

    return fig, {'display': 'block'}

@callback(
    Output('weights_container', 'children'),
    Input('n', 'value'),
    Input('random_weights', 'n_clicks'),
    State({'type': 'w', 'index': ALL}, 'value')
)
def set_weights(n: int, n_clicks: int, ws: List[float]):
    
    if (diff := n - len(ws)) > 0:
        ws.extend([1] * diff)

    if ctx.triggered_id == 'random_weights': ws = np.round(np.random.rand(len(ws)) * 2 - 1, 1)
        
    inputs = [html.Label('Choose the weights of the basis elements')] + [
        html.Div([
            html.Label(f'Basis {i+1}: '),
            dcc.Input(id={'type': 'w', 'index': i}, type='number', value=w, step=.1, inputMode='numeric'),
        ])
        for i, w in zip(range(n), ws)
    ]
    return inputs

@callback(
    Output('basis_elements', 'figure'),
    Output('basis_elements', 'style'),
    Output('error', 'children'),
    Input('n', 'value'),
    Input('p', 'value'),
    Input({'type': 'w', 'index': ALL}, 'value')
)
def plot_basis(n: int, p: int, ws: List[float]):
    if ws == []: return {}, {'display': 'none'}, None
    if p >= n: return {}, {'display': 'none'}, 'The degree cannot be smaller than the number of basis elements'
    spline = BSpline(p, n)
    xs = np.linspace(spline.t[0]-.2, spline.t[-1]+.2, 1000)
    ys = spline(xs)

    fig = make_subplots(rows=2, shared_xaxes=True)
    
    ws = np.nan_to_num(np.array(ws, dtype=float))[:, np.newaxis]
    fig.add_trace(
        go.Scatter(x=xs, y=np.sum(ys*ws, axis=0), mode='lines', line_color='black', name='spline'), 
        row=1, col=1
    )
    fig.add_traces(
        data=[
            go.Scatter(
                x=xs, 
                y=ys[i], 
                mode='lines', 
                name=f'Basis {i+1}', 
            ) 
            for i in range(n)
        ],
        rows=2, cols=1
    )

    line = {'line_dash': 'dash', 'opacity': .5}
    fig.add_vline(-1, annotation_text='-1', annotation_position='top', **line)
    fig.add_vline(1, annotation_text='1', annotation_position='top', **line)
    fig.update_xaxes(tickmode='array', tickvals=spline.t, ticktext=[f't{i}' for i in range(len(spline.t))], title='x', row=2, col=1)
    fig.update_yaxes(title='y')

    fig.update_layout(
        showlegend=False,
        title=f'Grid of {n} B-Spline basis elements of degree {p}',
        autosize=True,
        height=800,
        width=800,
    )

    return fig, {'display': 'block'}, None
