from dash import Input, Output, State, dcc, html, register_page, callback
import plotly.graph_objects as go
import numpy as np
np.random.seed(1)

from pages.backprop.backprop import NN
from utils import slider, updatemenu

nn_styles = [
    {'selector': 'node', 'style': {'label': '', 'font-size': '30'}},
    {'selector': 'edge', 'style': {'label': '', 'font-size': '30'}}
]

with open('pages/backprop/text.md') as f:
    text = f.read()

with open('pages/backprop/text_batches.md') as f:
    text_batches = f.read()

register_page(__name__, path='/backprop', name='Backpropagation + Interaction', order=8)

layout = html.Div([
    dcc.Markdown(text, mathjax=True, link_target='_blank'),
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
    html.Br(),
    dcc.Markdown(text_batches, mathjax=True, link_target='_blank')
])

@callback(
    Output('graph', 'children'),
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
    if n_clicks is None: return None
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
            layout={'title': f'Neural Network Training | Epoch: {epoch} | Mean Loss: {losses[epoch]:.3f}'},
            name=str(epoch)
        ) 
        for epoch in range(len(losses))
    ]

    y_margin = max(abs(y.min()-preds.min()), abs(y.max()-preds.max()))
    fig.update_xaxes(range=[-.1, 1.1])
    fig.update_yaxes(range=[y.min() - y_margin - .1, y.max() + y_margin + .1])  

    nn_menu = updatemenu
    nn_menu['buttons'][0]['args'][1]['frame']['duration'] = 1000 # Set to 1 frame per second
    fig.update_layout(
        title=f'Neural Network Training | Epoch: {len(losses)} | Mean Loss: {losses[-1]:.3f}', 
        updatemenus=[nn_menu], sliders=[slider(len(losses))]
    )
    return dcc.Graph(figure=fig)