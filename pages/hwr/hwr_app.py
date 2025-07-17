from dash import dcc, html, register_page
import dash_mantine_components as dmc

with open('pages/hwr/text.md') as f:
    text = f.read().split('<!-- code -->')

register_page(__name__, path='/hwr', name='Handwriting Recognition', order=4, icon='fluent-emoji:writing-hand')

def get_md(component: str) -> dcc.Markdown:
    return dcc.Markdown(component, mathjax=True, link_target='_blank', dangerously_allow_html=True)

def get_accordion(i: int, component: str) -> dmc.Accordion:
    return dmc.Accordion(
        children=[
            dmc.AccordionItem(
                children=[
                    dmc.AccordionControl('Python'),
                    dmc.AccordionPanel(get_md(component))
                ], 
                value=str(i))
        ], 
        chevronPosition='left'
    )

components = []
for i, component in enumerate(text):
    if i % 2 == 0: components.append(get_md(component))
    else: components.append(get_accordion(i, component))


layout = html.Div(components)
