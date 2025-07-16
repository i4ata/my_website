from dash import dcc, html, register_page
import dash_mantine_components as dmc

with open('pages/home/text.md') as f: text = f.read().split('<!-- A -->')

register_page(__name__, path='/', name='Home', order=0, icon='fluent-color:home-16')

layout = html.Div([
    dcc.Markdown(text[0], link_target='_blank', dangerously_allow_html=True),
    dmc.Accordion(
        children=[
            dmc.AccordionItem(
                children=[
                    dmc.AccordionControl('About the website'),
                    dmc.AccordionPanel(dcc.Markdown(text[1], link_target='_blank', dangerously_allow_html=True))
                ],
                value='about_website'
            ),
            dmc.AccordionItem(
                children=[
                    dmc.AccordionControl('About me'),
                    dmc.AccordionPanel(dcc.Markdown(text[2], link_target='_blank', dangerously_allow_html=True))
                ],
                value='about_me'
            ),
            dmc.AccordionItem(
                children=[
                    dmc.AccordionControl('How to run it yourself'),
                    dmc.AccordionPanel(dcc.Markdown(text[3], link_target='_blank', dangerously_allow_html=True))
                ],
                value='run_locally'
            )
        ],
        multiple=True,
        chevronPosition='left',
        variant='separated'
    )
])
