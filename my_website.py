import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from typing import List, Tuple

load_figure_template('bootstrap')
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

pages: List[Tuple[str, str]] = [(page['name'], page['relative_path']) for page in dash.page_registry.values()]
components = []
current_component = []
for name, href in pages:
    if href.count('/') == 1:
        if current_component:
            components.append(html.Ul(current_component))
            current_component = []
        components.append(html.Li(dcc.Link(name, href=href)))
    else:
        current_component.append(html.Li(dcc.Link(name, href)))
if current_component: components.append(html.Ul(current_component))
del current_component

app.layout = html.Div([
    html.H1('My Projects'),
    html.Ul(components),
    dcc.Location(id='url'),
    dash.page_container
])
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
