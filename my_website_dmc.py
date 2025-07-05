import dash
from dash import dcc
import dash_mantine_components as dmc
from dash_iconify import DashIconify

dmc.add_figure_templates(default='mantine_light')

app = dash.Dash(__name__, use_pages=True, external_stylesheets=dmc.styles.ALL)

def create_nav_link(icon: str, label: str, href: str):
    return dmc.NavLink(
        label=label,
        href=href,
        active='exact',
        leftSection=DashIconify(icon=icon)
    )

navbar_links = dmc.Box([
    dmc.Stack([
        create_nav_link(icon='radix-icons:rocket', label=page['name'], href=page['path'])
        for page in dash.page_registry.values()
        if page['path'].count('/') == 1
    ])
])

layout = dmc.AppShell(
    [
        dcc.Location(id='url'),
        dmc.AppShellHeader(dmc.Title('My Projects')),
        dmc.AppShellNavbar(children=navbar_links),
        dmc.AppShellMain(dash.page_container),
    ],
    header={'height': 60},
    navbar={'width': 250}
)


app.layout = dmc.MantineProvider(layout)

if __name__ == '__main__':
    app.run(debug=True)