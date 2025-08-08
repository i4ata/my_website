import dash
from dash import dcc
import dash_mantine_components as dmc
from dash_iconify import DashIconify

dmc.add_figure_templates(default='mantine_light')
app = dash.Dash(__name__, use_pages=True, external_stylesheets=dmc.styles.ALL)
server = app.server

def create_nav_link(icon: str, label: str, href: str):
    return dmc.NavLink(
        label=dmc.Text(label, size='md'),
        href=href, active='exact',
        leftSection=DashIconify(icon=icon, width=28, height=28)
    )

navbar_links = dmc.Stack(
    [
        create_nav_link(icon=page['icon'], label=page['name'], href=page['path'])
        for page in dash.page_registry.values() if page['path'].count('/') == 1
    ],
    gap='md'
)

def create_icon_link(icon: str, href: str):
    return dmc.Anchor(
        DashIconify(icon=icon, width=50, height=50),
        href=href, target='_blank',
    )

contact_links = dmc.Group(
    justify='center', gap='md',
    children=[
        create_icon_link('logos:google-gmail', 'mailto:ivaylo.russinov@gmail.com'),
        create_icon_link('logos:github-icon', 'https://github.com/i4ata'),
        create_icon_link('logos:linkedin-icon', 'https://www.linkedin.com/in/ivaylo-rusinov-7002b2230/'),
    ]  
)

navbar = dmc.Box([
    navbar_links,
    dmc.Divider(label='Contact', size='md', style={'marginBottom': 30, 'marginTop': 20}),
    contact_links
])

layout = dmc.AppShell(
    [
        dcc.Location(id='url'),
        dmc.AppShellHeader(dmc.Title('My Projects')),
        dmc.AppShellNavbar(navbar),
        dmc.AppShellMain(dmc.Container(dash.page_container, px=20, size='100%', pb=250)),
    ],
    header={'height': 60}, navbar={'width': 250}
)

app.layout = dmc.MantineProvider(layout)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
