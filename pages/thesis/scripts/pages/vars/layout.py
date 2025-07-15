from dash import html, dcc

PREFIX = 'vars-'

layout = html.Div([    
    html.H2('Variables'),
    html.P(['Here you can interact with individual variables. Click on one of the available ones to visualize how many trees selected it and in the context of which other variables or go back to ', dcc.Link('Model Selection', href='/thesis')]),
    dcc.RadioItems(id=PREFIX+'radio-variable'),
    dcc.Graph(id=PREFIX+'graph', style={'display': 'none'})
])
