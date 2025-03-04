from dash import Input, Output, State, dcc, html, register_page, callback
import plotly.graph_objects as go
import numpy as np
import dash_cytoscape as cyto
np.random.seed(1)

from pages.backprop.backprop import NN, Linear
from utils import slider, updatemenu

# TODO: THE EDGES' LABELS APPEAR BEHIND THE EDGES

nn_styles = [
    {'selector': 'node', 'style': {'label': '', 'font-size': '30'}},
    {'selector': 'edge', 'style': {'label': '', 'font-size': '30'}}
]

with open('pages/backprop/text.md') as f:
    text = f.read()

register_page(__name__, path='/backprop')

layout = html.Div([
    dcc.Markdown(text, mathjax=True),
    html.Div([
        html.Label('Activation Function'),
        html.Br(),
        dcc.RadioItems(id='activation', options=('tanh', 'relu'), value='tanh')
    ]),
    html.Br(),
    html.Div([
        html.Label('Number of hidden layers'),
        html.Br(),
        dcc.Input(id='n_layers', value=2, step=1, min=0, type='number')
    ]),
    html.Br(),
    html.Div([
        html.Label('Number of units per hidden layer'),
        html.Br(),
        dcc.Input(id='size', value=8, step=1, min=0, type='number')
    ]),
    html.Br(),
    html.Div([
        html.Label('Learning rate'),
        html.Br(),
        dcc.Input(id='lr', value=0.01, step=.001, min=0, type='number')
    ]),
    html.Br(),
    html.Div([
        html.Label('Target function'),
        html.Br(),
        dcc.RadioItems(id='target', options=('sin(4πx)', 'e^x', 'log(1+x)'), value='sin(4πx)')
    ]),
    html.Br(),
    html.Div([
        html.Label('Number of samples'),
        html.Br(),
        dcc.Input(id='n_samples', value=50, step=10, min=0, type='number')
    ]),
    html.Br(),
    html.Div([
        html.Label('Number of epochs'),
        html.Br(),
        dcc.Input(id='epochs', value=5, step=1, min=0, type='number')
    ]),
    html.Br(),
    html.Button('Submit', id='submit'),
    html.Div(id='graph'),
    # cyto.Cytoscape(
    #     id='nn',
    #     userZoomingEnabled=False,
    #     userPanningEnabled=False,
    #     layout={
    #         'name': 'breadthfirst',
    #         'directed': True,
    #     },
    #     stylesheet=nn_styles,
    #     style={'width': '100vw', 'height': '90vh'}
    # )
])

def get_nn(nn: NN):
    data = []
    for i, layer in enumerate(filter(lambda l: type(l) == Linear, nn.layers)):
        if i == 0:
            data.extend([
                {'data': {'id': f'-1_{j}', 'label': f'Input {j+1}'}} 
                for j in range(len(layer.W))
            ])
        data.extend([
            {'data': {'id': f'{i}_{j}', 'label': f'{layer.b[0, j]:.4f}'}} 
            for j in range(layer.W.shape[1])
        ])
        data.extend([
            {'data': {'source': source, 'target': target, 'label': f'{layer.W[source_idx, target_idx]:.4f}'}}
            for source_idx, source in enumerate([x['data']['id'] for x in data if 'id' in x['data'] and x['data']['id'].startswith(str(i-1))])
            for target_idx, target in enumerate([x['data']['id'] for x in data if 'id' in x['data'] and x['data']['id'].startswith(str(i))])
        ])
    return data

@callback(
    Output('graph', 'children'),
    # Output('nn', 'elements'),
    Input('submit', 'n_clicks'),
    State('activation', 'value'),
    State('n_layers', 'value'),
    State('size', 'value'),
    State('lr', 'value'),
    State('target', 'value'),
    State('n_samples', 'value'),
    State('epochs', 'value')
)
def train(n_clicks, activation, n_layers, size, lr, target, n_samples, epochs):
    if n_clicks is None: return None#, []
    nn = NN(n_layers=n_layers, activation=activation, size=size, lr=lr)
    
    x_train = np.linspace(0, 1, n_samples).reshape(-1, 1, 1)
    y_train: np.ndarray = {
        'sin(4πx)': lambda x: np.sin(x * 4 * np.pi),
        'e^x': np.exp,
        'log(1+x)': np.log1p
    }[target](x_train)
    losses, preds = nn.fit(x_train, y_train, epochs=epochs, store_all=True)
    x, y = np.squeeze(x_train), np.squeeze(y_train)
    fig = go.Figure()
    fig.add_traces([
        go.Scatter(x=x, y=y, mode='markers', name='Ground Truth'),
        go.Scatter(x=x, y=preds[-1], mode='lines', name='Prediction')
    ])

    fig.frames = [
        go.Frame(
            data=[{'y': preds[epoch]}], 
            traces=[1], 
            layout={'title': f'Mean Loss: {losses[epoch]}'},
            name=str(epoch)
        ) 
        for epoch in range(len(losses))
    ]

    y_margin = max(abs(y.min()-preds.min()), abs(y.max()-preds.max()))
    fig.update_layout(updatemenus=[updatemenu], sliders=[slider(len(losses))])
    fig.update_xaxes(range=[-.1, 1.1])
    fig.update_yaxes(range=[y.min() - y_margin + .1, y.max() + y_margin + .1])  
    return dcc.Graph(figure=fig)#, get_nn(nn)

# @app.callback(
#     Output('nn', 'stylesheet'),
#     Input('nn', 'tapNodeData')
# )
# def update_stylesheet(node_data):
    
#     if node_data:
#         node_id = node_data['id']
#         edge_label_styles = [
#             {'selector': f'edge[source="{node_id}"], edge[target="{node_id}"]',
#              'style': {
#                  'label': 'data(label)',
#              }},
#             {'selector': f'node[id="{node_id}"]',
#             'style': {
#                 'label': 'data(label)',
#             }}
#         ]
#         return nn_styles + edge_label_styles
    
#     return nn_styles
