from dash import html, dcc

PREFIX = 'home-'

with open('pages/thesis/scripts/pages/home/text.md') as f: text = f.read()

layout = html.Div([
    html.H2('Model Selection'),
    html.P('Load a trained causal forest from memory'),
    dcc.Dropdown(id=PREFIX+'models', placeholder='Choose a model', persistence=True, persistence_type='session'),
    dcc.Loading(html.Div(id=PREFIX+'info')),
    html.Br(),
    dcc.Markdown(text, link_target='_blank', mathjax=True, dangerously_allow_html=True)
])
