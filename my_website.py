import dash
from dash import Dash, html, dcc

app = Dash(__name__, use_pages=True)
server = app.server

app.layout = html.Div([
    html.Div([
        html.Div(
            dcc.Link(f'{page["name"]}', href=page['relative_path'])
        ) for page in dash.page_registry.values()
    ]),
    dash.page_container
])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
