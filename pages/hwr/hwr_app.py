from dash import dcc, html, register_page

with open('pages/hwr/text.md') as f:
    text = f.read()

register_page(__name__, path='/hwr', name='Handwriting Recognition', order=3, icon='fluent-emoji:writing-hand')

layout = html.Div([dcc.Markdown(text, mathjax=True, link_target='_blank', dangerously_allow_html=True)])
