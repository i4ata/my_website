from dash import dcc, html, register_page

with open('pages/udl/text.md') as f:
    text = f.read()

register_page(__name__, path='/udl', name='Unupervised Deep Learning Tutorials', order=9)

layout = html.Div([dcc.Markdown(text, mathjax=True)])
