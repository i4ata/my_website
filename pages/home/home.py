from dash import dcc, html, register_page

with open('pages/home/text.md') as f:
    text = f.read()

register_page(__name__, path='/', name='Home', order=0)

layout = html.Div([dcc.Markdown(text, link_target='_blank', dangerously_allow_html=True)])

# import dash_bootstrap_components as dbc
# layout = dbc.Container([
#     dbc.Card([
#         dbc.CardBody([
#             dcc.Markdown(text, link_target='_blank', dangerously_allow_html=True)
#         ])
#     ], className="mt-4 shadow-sm")
# ])