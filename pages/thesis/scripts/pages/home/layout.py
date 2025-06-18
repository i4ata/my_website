from dash import html, dcc

PREFIX = 'home-'

layout = html.Div([
    html.H2('Model Selection'),
    html.P('Load a trained causal forest from memory (looking in resources/models)'),
    dcc.Dropdown(id=PREFIX+'models', placeholder='Choose a model', persistence=True, persistence_type='session'),
    dcc.Loading(html.Div(id=PREFIX+'info'))
])
