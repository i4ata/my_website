from dash import html, dcc, dash_table

PREFIX = 'vars-'
var_table_styles = [{'if': {'filter_query': '{is split} = T'}, 'backgroundColor': 'paleturquoise'}]

layout = html.Div([    
    html.H2('Variables', id=PREFIX+'txt-warning'),
    dcc.RadioItems(id=PREFIX+'radio-variable'),

    html.Div(
        children=[
            dcc.Graph(id=PREFIX+'graph'),

            # Button to trigger the table with stats for that variable across all trees
            html.Button('Show Table', id=PREFIX+'button-show-table'),
        ],
        id=PREFIX+'variable-visualization-div',
        hidden=True
    ),

    html.Div(id=PREFIX+'table-header-div'),
    
    # The table
    dash_table.DataTable(
        id=PREFIX+'table', 
        row_selectable='single', 
        column_selectable='single',
        style_data_conditional=var_table_styles,
        style_cell={'whiteSpace': 'pre-line', 'textAlign': 'left'}
    ),

    dcc.Loading(html.Div(
        children=[
            html.P('This graph shows the two resulting populations after splitting the data at the node'),
            dcc.Graph(id=PREFIX+'graph-at-node'),
            html.P('This graph shows the split after stratification'),
            dcc.Graph(id=PREFIX+'graph-at-stratum')
        ],
        id=PREFIX+'graphs-div',
        hidden=True
    ))
])
