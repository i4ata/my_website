from dash import dcc, html, register_page

with open('pages/unet/text.md') as f:
    text = f.read()

register_page(__name__, path='/unet', name='The U-Net', order=6)

layout = html.Div([dcc.Markdown(text, mathjax=True)])
