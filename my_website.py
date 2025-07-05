import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from typing import List, Tuple

dmc.add_figure_templates(default="mantine_light")
app = Dash(__name__, use_pages=True, external_stylesheets=[dmc.styles.ALL])
server = app.server

pages: List[Tuple[str, str]] = [
    (page['name'], page['relative_path']) 
    for page in dash.page_registry.values() 
    if page['relative_path'].count('/') == 1
]

app.layout = dbc.Container([
    dcc.Location(id='url'),
    dbc.Row([
        dbc.Col([
            html.H2('My Website'),
            html.Hr(),
            dbc.Nav(
                children=[dbc.NavLink(name, href=href, active='exact') for name, href in pages],
                vertical=True, pills=True
            ),
        ], width={'size': 2, 'offset': 0, 'order': 2}),
        dbc.Col(
            [dash.page_container], 
            style={'overflowY': 'auto', 'height': '100vh'}, 
            className='fixed-top',
            width={'size': 10, 'offset': 2, 'order': 2}
        )
    ])
], fluid=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
