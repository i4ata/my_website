import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

load_figure_template('bootstrap')
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    html.H1('My Projects'),
    html.Ul([
        html.Li(
            dcc.Link(f'{page["name"]}', href=page['relative_path'])
        ) for page in dash.page_registry.values()
    ]),
    dash.page_container
])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
