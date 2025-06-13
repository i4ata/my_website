from dash import Input, Output, State, dcc, html, dash_table, no_update, Patch, ctx, register_page, callback, clientside_callback
from dash.exceptions import PreventUpdate
from dash.dash_table.Format import Format
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Literal, List, Union
from pages.sebra.utils import (
    con, df_clients, df_orgs, df_payments, df_primary_orgs, df_sebra, pies, pay_codes, 
    compare_codes, make_pies, make_timeline, plot_primary_orgs, compare_weekdays, plot_treemap, make_sankey
)

register_page(__name__, path='/sebra', name='SEBRA Payments', order=1)

example_query = """SELECT ORGANIZATION, PRIMARY_ORGANIZATION, ROUND(AVG(AMOUNT)) AS MEAN_AMOUNT, COUNT(AMOUNT) AS TOTAL_PAYMENTS 
FROM payments 
JOIN organizations USING (ORGANIZATION_ID) 
JOIN primary_organizations USING (PRIMARY_ORG_CODE) 
WHERE SEBRA_PAY_CODE = 10 
GROUP BY ORGANIZATION 
ORDER BY MEAN_AMOUNT DESC 
LIMIT 5;"""

with open('pages/sebra/text.md') as f:
    text = f.read()

layout = html.Div([
    dcc.Markdown(text, link_target='_blank', dangerously_allow_html=True),
    # dcc.Graph(figure=make_sankey()),
    html.Div([
        html.P('Overall stats:'),
        html.Ul([
            html.Li(f'Total amount spent: {round(df_payments["AMOUNT"].sum()):,}'),
            html.Li(f'Total number of transactions: {len(df_payments):,}'),
            html.Li(f'Number of unique clients: {df_payments["CLIENT_ID"].nunique()}'),
            html.Li(f'Number of unique organizations: {df_payments["ORGANIZATION_ID"].nunique()}')
        ])
    ]),
    dcc.Tabs([
        dcc.Tab(
            children=[
                dcc.Store(id='code_selection'),
                html.Div([
                    html.H2('SEBRA Pay Codes'),
                    html.P("Click on the pies' sectors or the bars to get more information about the payments for different categories."),
                    html.Div(
                        children=[
                            dcc.Graph(id='pie', figure=make_pies(), style={'width': '50%'}),
                            dcc.Graph(id='bar', figure=compare_codes(), style={'width': '50%'})
                        ],
                        style={'display': 'flex'}
                    )
                ]),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.P('Choose one from the group with rarely occuring codes:'),
                                dcc.RadioItems(id='sebra_other_radio')
                            ],
                            id='sebra_other_container',
                            hidden=True
                        ),
                        html.Div(id='sebra_summary'),
                        html.Div(
                            children=[
                                html.P('Most paying organizations'),
                                dash_table.DataTable(id='sebra_orgs_table'),
                                html.Br(), html.Button('Select a specific organization', id='org_selection_button'),
                                html.Div(
                                    children=[
                                        html.P(id='orgs_filter_summary'),
                                        html.Label('Minimum Total Amount: '), dcc.Input(id='orgs_min_amount', type='number', min=0, step=1_000),
                                        html.Label(' Minimum Number of Payments: '), dcc.Input(id='orgs_min_payments', type='number', min=0, step=1),
                                        html.Br(),
                                        html.Label('Maximum Total Amount: '), dcc.Input(id='orgs_max_amount', type='number', min=0, step=1_000),
                                        html.Label(' Maximum Number of Payments: '), dcc.Input(id='orgs_max_payments', type='number', min=0, step=1),
                                        html.Br(),
                                        dcc.Dropdown(id='primary_orgs_selection', placeholder='Select a Primary Organization ...'),
                                        html.Br(),
                                        html.Button('Submit', id='submit_org_filter'),
                                        html.P(id='orgs_filter_output_summary')
                                    ],
                                    id='sebra_filter_orgs',
                                    hidden=True
                                )
                            ],
                            id='sebra_orgs_container',
                            hidden=True
                        ),
                        html.Div(
                            children=[
                                html.Br(),
                                dcc.Dropdown(id='sebra_orgs_dropdown'),
                                dash_table.DataTable(id='sebra_specific_org_table')
                            ],
                            id='sebra_individual_org_container',
                            hidden=True,
                        ),
                        html.Div(
                            children=[
                                html.Br(),
                                html.P('Most receiving clients'),
                                dash_table.DataTable(id='sebra_clients_table'),
                                html.Br(),
                                html.Button('Select a specific client', id='client_selection_button'),
                                html.Div(
                                    children=[
                                        html.P(id='clients_filter_summary'),
                                        html.Label('Minimum Total Amount: '), dcc.Input(id='clients_min_amount', type='number', min=0, step=1_000),
                                        html.Label(' Minimum Number of Payments: '), dcc.Input(id='clients_min_payments', type='number', min=0, step=1),
                                        html.Br(),
                                        html.Label('Maximum Total Amount: '), dcc.Input(id='clients_max_amount', type='number', min=0, step=1_000),
                                        html.Label(' Maximum Number of Payments: '), dcc.Input(id='clients_max_payments', type='number', min=0, step=1),
                                        html.Br(),
                                        html.Button('Submit', id='submit_client_filter'),
                                        html.P(id='clients_filter_output_summary')
                                    ],
                                    id='sebra_filter_clients',
                                    hidden=True
                                )
                            ],
                            id='sebra_clients_container',
                            hidden=True
                        ),
                        html.Div(
                            children=[
                                html.Br(),
                                dcc.Dropdown(id='sebra_clients_dropdown'),
                                dash_table.DataTable(id='sebra_specific_client_table')
                            ],
                            id='sebra_individual_client_container',
                            hidden=True,
                        ),
                    ],
                    id='pie_info',
                    hidden=True
                ),
            ],
            label='Pay Codes'
        ),
        dcc.Tab(
            children=[
                dcc.Store(id='date_selection'),
                html.Div([
                    html.H2('Payments Over Time'),
                    dcc.Graph(id='timeline', figure=make_timeline()),
                    dcc.Checklist(id='timeline_options', options=['Log scale', 'Hide weekends & holidays'])
                ]),
                html.Div(
                    children=[
                        html.Div(id='timeline_output'),
                        html.P('Largest payment on this date'),
                        dash_table.DataTable(id='timeline_table')
                    ],
                    id='timeline_info',
                    hidden=True
                )
            ],
            label='Timeline'
        ),
        dcc.Tab(
            children=[
                dcc.Store(id='primary_org_selection'),
                html.Div([
                    html.H2('Primary Organizations'),
                    html.P('Click on the points to see more details about that organization.'),
                    dcc.Graph(id='primary_orgs', figure=plot_primary_orgs(), mathjax=True),
                ]),
                html.Div(
                    children=[
                        html.Div(id='primary_orgs_output'),
                        html.P('Most receiving clients'),
                        dash_table.DataTable(id='primary_orgs_table')
                    ],
                    id='primary_orgs_info',
                    hidden=True
                ),
            ],
            label='Primary Organizations'
        ),
        dcc.Tab(
            children=[
                html.Div([
                    html.H2('Payments Per Day of the Week'),
                    dcc.Graph(figure=compare_weekdays())
                ]),
                html.P('Run an arbitrary SQL query on the data'),
                dcc.Textarea(
                    id='query', 
                    value=example_query,
                    style={'width': '60%', 'height': 200}
                ), html.Br(),
                html.Button('Submit', id='submit'),
                html.P(id='query_feedback'),
                dash_table.DataTable(id='query_result'),
                html.Button('Download result as CSV', id='download_button'),
                dcc.Download(id='download_query'),
            ],
            label='Misc'
        )
    ])
])

def get_columns(df: pd.DataFrame): 
    return (
        [{'name': col, 'id': col} for col in df.select_dtypes([object, 'datetime']).columns]
        +
        [{'name': col, 'id': col, 'type': 'numeric', 'format': Format().group(True)} for col in df.select_dtypes('number').columns]
    )

# TAB 1

@callback(
    Output('code_selection', 'data'),
    Output('pie', 'figure'),
    Output('bar', 'figure'),
    Output('sebra_other_container', 'hidden'),
    Output('sebra_other_radio', 'options'),

    Input('pie', 'clickData'),
    Input('bar', 'clickData'),
    Input('sebra_other_radio', 'value'),
    State('code_selection', 'data')
)
def click_on_pie(click_data_pie, click_data_bar, other_code, code_selection):

    # Initial call. No one has clicked on any of the graphs or the radio: don't do anything
    if click_data_pie is None and click_data_bar is None and other_code is None: return [no_update] * 5
    
    # If the callback is triggered by clicking one of the 'other' codes: Don't change the plots
    if ctx.triggered_id == 'sebra_other_radio':
        fig_pie, fig_bar = no_update, no_update
        code = other_code

    # If the callback is triggered by clicking on either graph
    else:
        point = (click_data_pie if ctx.triggered_id == 'pie' else click_data_bar)['points'][0]
        code = point['label']

        # If the selected code is the same as the old one
        if code_selection is not None and code == str(code_selection): return [no_update] * 5

        # Update the pie chart by pulling the selected section
        fig_pie = Patch()
        pull = [0] * len(pies['labels'])
        if ctx.triggered_id == 'bar' and int(code) in pies['to_remove']:
            pull[pies['labels'].index(pies['other_label'])] = .2
        else:
            pull[pies['labels'].index(int(code) if code != pies['other_label'] else code)] = .2
        fig_pie['data'][0]['pull'] = pull
        fig_pie['data'][1]['pull'] = pull

        # Update the bar graph by highlighting the selected bar
        fig_bar = Patch()
        colors = ['#636efa'] * len(pay_codes)
        if code == pies['other_label']: 
            for i in pies['to_remove']: colors[pay_codes.index(i)] = 'red'
        else:
            colors[pay_codes.index(int(code))] = 'red'
        fig_bar['data'][0]['marker']['color'] = colors

    # If the callback is triggered by clicking the "other" section of the pie, render the radio
    if ctx.triggered_id == 'pie' and code == pies['other_label']:
        options = [
            {'label': f'{code}: {df_sebra.loc[code, "DESCRIPTION"]}', 'value': code}
            for code in pies['to_remove']
        ]
        return no_update, fig_pie, fig_bar, False, options
    
    to_return = (
        int(code), fig_pie, fig_bar, 
        True if ctx.triggered_id == 'pie' else no_update, 
        [] if ctx.triggered_id == 'pie' else no_update,
    )
    return to_return 

@callback(
    Output('sebra_summary', 'children'),
    Output('pie_info', 'hidden'),
    
    Output('sebra_orgs_container', 'hidden'),
    Output('sebra_orgs_table', 'data'),
    Output('sebra_orgs_table', 'columns'),

    Output('sebra_clients_container', 'hidden'),
    Output('sebra_clients_table', 'data'),
    Output('sebra_clients_table', 'columns'),

    Input('code_selection', 'data')
)
def pie_summary(code: int):
    if code is None: return [no_update] * 8
    
    payments = (
        df_payments
        .query('SEBRA_PAY_CODE == @code')
        .merge(df_orgs, on='ORGANIZATION_ID')
        .merge(df_primary_orgs, on='PRIMARY_ORG_CODE')
        .merge(df_clients, on='CLIENT_ID')
    )
    size, total = len(payments), payments['AMOUNT'].sum()

    orgs_summary = (
        payments
        .groupby('ORGANIZATION', as_index=False)
        .agg({'PRIMARY_ORGANIZATION': 'first', 'AMOUNT': ['sum', 'median', 'size'], 'CLIENT_ID': 'nunique'})
        .nlargest(5, [('AMOUNT', 'sum')])
        .round()
    )
    orgs_summary.columns = ['Organization', 'Primary Organization', 'Total Amount', 'Median Amount', 'Total Payments', 'Number of Unique Clients']

    clients_summary = (
        payments
        .groupby('CLIENT_RECEIVER_NAME', as_index=False)
        .agg({'AMOUNT': ['sum', 'median', 'size'], 'ORGANIZATION': 'nunique', 'PRIMARY_ORGANIZATION': 'nunique'})
        .nlargest(5, [('AMOUNT', 'sum')])
        .round()
    )
    clients_summary.columns = ['Receiver Name', 'Total Amount', 'Median Amount', 'Total Payments', 'Unique Organizations', 'Unique Primary Organizations']
    
    sebra_summary = [
        html.P(f'You selected SEBRA code {code}'),
        html.Ul([
            html.Li(f'Definition: {df_sebra.loc[code, "DESCRIPTION"]}'),
            html.Li(f'Monetary share: {total:,.2f} ({100 * total / df_payments["AMOUNT"].sum() :.2f}%)'),
            html.Li(f'Frequency share: {size:,} ({100 * size / len(df_payments) :.2f}%)'),
            html.Li(f'{payments["ORGANIZATION_ID"].nunique()} organizations from {payments["PRIMARY_ORGANIZATION"].nunique()} primary organizations have made payments to {payments["CLIENT_ID"].nunique()} clients')
        ])
    ]

    return (
        sebra_summary, False, 
        False, orgs_summary.to_dict('records'), get_columns(orgs_summary), 
        False, clients_summary.to_dict('records'), get_columns(clients_summary)
    )

@callback(
    Output('sebra_individual_org_container', 'hidden'),
    Output('sebra_orgs_dropdown', 'options'),

    Output('sebra_filter_orgs', 'hidden'),
    Output('orgs_filter_summary', 'children'),
    Output('primary_orgs_selection', 'options'),

    Output('orgs_filter_output_summary', 'children'),

    Input('org_selection_button', 'n_clicks'),
    Input('submit_org_filter', 'n_clicks'),
    Input('code_selection', 'data'),

    State('orgs_min_amount', 'value'),
    State('orgs_min_payments', 'value'),
    State('orgs_max_amount', 'value'),
    State('orgs_max_payments', 'value'),
    State('primary_orgs_selection', 'value')
)
def select_specifig_org(
    org_selection: Optional[int],
    submit_org_filter: Optional[int],
    code: int,

    min_amount: Optional[int], min_payments: Optional[int], max_amount: Optional[int], max_payments: Optional[int], primary_org: Optional[str] 
):
    # Initial call: All is None
    if org_selection is None and submit_org_filter is None and code is None: return [no_update] * 6

    # If we chose a different sebra code, reset everything
    if ctx.triggered_id == 'code_selection': return True, [], True, None, [], None

    df_payments_orgs = df_payments.query('SEBRA_PAY_CODE == @code').merge(df_orgs, on='ORGANIZATION_ID')
    unique_orgs = np.sort(df_payments_orgs['ORGANIZATION'].unique())

    # If we click on the button for selecting an individual organization
    if ctx.triggered_id == 'org_selection_button':
        if len(unique_orgs) < 50: return False, unique_orgs, True, None, [], None
        primary_orgs = np.sort(
            df_orgs[df_orgs['ORGANIZATION'].isin(unique_orgs)]
            .merge(df_primary_orgs, on='PRIMARY_ORG_CODE')
            ['PRIMARY_ORGANIZATION']
            .unique()
        )
        summary = f'There are {len(unique_orgs)} unique organizations, you can filter them first',
        return True, [], False, summary, primary_orgs, None

    # If we filter the orgs
    if primary_org:
        df_payments_orgs = df_payments_orgs.merge(df_primary_orgs.query('PRIMARY_ORGANIZATION == @primary_org'), on='PRIMARY_ORG_CODE')
    aggregation: pd.DataFrame = (
        df_payments_orgs
        .groupby('ORGANIZATION', as_index=False)
        ['AMOUNT']
        .agg(['sum', 'size'])
        .rename({'sum': 'amount', 'size': 'payments'}, axis='columns')
    )
    query = ' & '.join(
        f'{col} {sign} {value}' 
        for col, sign, value in 
        zip(('amount', 'payments', 'amount', 'payments'), ('>=', '>=', '<=', '<='), (min_amount, min_payments, max_amount, max_payments))
        if value is not None
    )
    orgs = np.sort((aggregation.query(query) if query else aggregation)['ORGANIZATION'].unique())
    return False, orgs, no_update, no_update, no_update, f'Filtered {len(orgs)} organizations'

@callback(
    Output('sebra_individual_client_container', 'hidden'),
    Output('sebra_clients_dropdown', 'options'),

    Output('sebra_filter_clients', 'hidden'),
    Output('clients_filter_summary', 'children'),
    
    Output('clients_filter_output_summary', 'children'),

    Input('client_selection_button', 'n_clicks'),
    Input('submit_client_filter', 'n_clicks'),
    Input('code_selection', 'data'),

    State('clients_min_amount', 'value'),
    State('clients_min_payments', 'value'),
    State('clients_max_amount', 'value'),
    State('clients_max_payments', 'value'),
)
def select_specifig_client(
    client_selection: Optional[int],
    submit_client_filter: Optional[int],
    code: int,

    min_amount: Optional[int], min_payments: Optional[int], max_amount: Optional[int], max_payments: Optional[int] 
):
    # Initial call: All is None
    if client_selection is None and submit_client_filter is None and code is None: return [no_update] * 5
    
    df_payments_clients = df_payments.query('SEBRA_PAY_CODE == @code').merge(df_clients, on='CLIENT_ID')
    unique_clients = np.sort(df_payments_clients['CLIENT_RECEIVER_NAME'].unique())

    # If we chose a different sebra code, reset everything
    if ctx.triggered_id == 'code_selection': return True, [], True, None, None

    # If we click on the button for selecting an individual client
    if ctx.triggered_id == 'client_selection_button':
        if len(unique_clients) < 50: return False, unique_clients, True, None, None
        summary = f'There are {len(unique_clients)} unique clients, you can filter them first',
        return True, [], False, summary, None

    # If we filter the clients
    aggregation: pd.DataFrame = (
        df_payments_clients
        .groupby('CLIENT_RECEIVER_NAME', as_index=False)
        ['AMOUNT']
        .agg(['sum', 'size'])
        .rename({'sum': 'amount', 'size': 'payments'}, axis='columns')
    )
    query = ' & '.join(
        f'{col} {sign} {value}' 
        for col, sign, value in 
        zip(('amount', 'payments', 'amount', 'payments'), ('>=', '>=', '<=', '<='), (min_amount, min_payments, max_amount, max_payments))
        if value is not None
    )
    clients = np.sort((aggregation.query(query) if query else aggregation)['CLIENT_RECEIVER_NAME'].unique())
    return False, clients, no_update, no_update, f'Filtered {len(clients)} clients'

@callback(
    Output('sebra_specific_org_table', 'data'),
    Output('sebra_specific_org_table', 'columns'),
    Input('sebra_orgs_dropdown', 'value'),
    State('code_selection', 'data')
)
def display_org_info(org: Optional[str], code: int):

    if org is None: return [], []
    org_id = df_orgs.query('ORGANIZATION == @org').iloc[0].name
    data = (
        df_payments
        .query('SEBRA_PAY_CODE == @code & ORGANIZATION_ID == @org_id')
        .merge(df_clients, on='CLIENT_ID')
        [['SETTLEMENT_DATE', 'AMOUNT', 'CLIENT_RECEIVER_NAME', 'REASON1', 'REASON2']]
    )
    # Hacky way to avoid the datatable to show the hours, minutes, and seconds
    data['SETTLEMENT_DATE'] = data['SETTLEMENT_DATE'].dt.strftime('%d-%m-%Y')
    data.columns = ['Settlement Date', 'Payment Amount', 'Client Name', 'Reason 1', 'Reason 2']
    return data.to_dict('records'), get_columns(data)

@callback(
    Output('sebra_specific_client_table', 'data'),
    Output('sebra_specific_client_table', 'columns'),
    Input('sebra_clients_dropdown', 'value'),
    State('code_selection', 'data')
)
def display_client_info(client: Optional[str], code: int):
    if client is None: return [], []
    client_id = df_clients.query('CLIENT_RECEIVER_NAME == @client').iloc[0].name
    data = (
        df_payments
        .query('SEBRA_PAY_CODE == @code & CLIENT_ID == @client_id')
        .merge(df_orgs, on='ORGANIZATION_ID')
        .merge(df_primary_orgs, on='PRIMARY_ORG_CODE')
        [['SETTLEMENT_DATE', 'AMOUNT', 'ORGANIZATION', 'PRIMARY_ORGANIZATION', 'REASON1', 'REASON2']]
    )
    # Hacky way to avoid the datatable to show the hours, minutes, and seconds
    data['SETTLEMENT_DATE'] = data['SETTLEMENT_DATE'].dt.strftime('%d-%m-%Y')
    data.columns = ['Settlement Date', 'Payment Amount', 'Organization', 'Primary Organization', 'Reason 1', 'Reason 2']
    return data.to_dict('records'), get_columns(data)

# TAB 2

def draw_rectangle(x: str) -> dict:
    date = datetime(*map(int, x.split('-')))
    time_delta = timedelta(hours=12)
    return {
        'fillcolor': 'green',
        'layer': 'below',
        'line': {'width': 0},
        'opacity': 0.3,
        'type': 'rect',
        'x0': date - time_delta,
        'x1': date + time_delta,
        'xref': 'x',
        'y0': 0,
        'y1': 1,
        'yref': 'y domain'
    }
    
@callback(
    Output('timeline', 'figure'),
    Output('date_selection', 'data'),
    Input('timeline_options', 'value'),
    Input('timeline', 'clickData'),
    State('date_selection', 'data')
)
def plot_timeline(options, click_data, date):
    if options is None and click_data is None: return no_update, no_update
    if ctx.triggered_id == 'timeline_options':
        fig = make_timeline('Hide weekends & holidays' in options, 'Log scale' in options)
        if date is not None: fig.add_shape(**draw_rectangle(date))
        return fig, no_update

    new_date = click_data['points'][0]['x']
    fig = Patch()
    fig['layout']['shapes'] = [draw_rectangle(new_date)]
    return fig, new_date

@callback(
    Output('timeline_output', 'children'),
    Input('date_selection', 'data')
)
def date_summary(date: str):
    data = df_payments[df_payments['SETTLEMENT_DATE'] == date]
    return f'Chosen date: {date}, Unique clients: {data["CLIENT_ID"].nunique()}, Unique organizations: {data["ORGANIZATION_ID"].nunique()}'

@callback(
    Output('timeline_table', 'data'),
    Output('timeline_table', 'columns'),
    Output('timeline_info', 'hidden'),
    Input('date_selection', 'data')
)
def largest_payments_today(date: str):
    if date is None: return no_update, no_update, no_update
    data = (
        df_payments
        [df_payments['SETTLEMENT_DATE'] == date]
        .nlargest(5, 'AMOUNT')
        .merge(df_clients, on='CLIENT_ID')
        .merge(df_orgs, on='ORGANIZATION_ID')
        .merge(df_primary_orgs, on='PRIMARY_ORG_CODE')
        [['PRIMARY_ORGANIZATION', 'ORGANIZATION', 'AMOUNT', 'CLIENT_RECEIVER_NAME', 'SEBRA_PAY_CODE', 'REASON1', 'REASON2']]
        .round({'AMOUNT': 0})
    )
    data.columns = ['Primary Organization', 'Organization', 'Amount', 'Client', 'SEBRA Pay Code', 'Reason 1', 'Reason 2']
    return data.to_dict('records'), get_columns(data), False

@callback(
    Output('query_result', 'data'),
    Output('query_result', 'columns'),
    Output('query_feedback', 'children'),
    Input('submit', 'n_clicks'),
    State('query', 'value')
)
def run_query(n_clicks: int, query: str):
    try: query_result = pd.read_sql_query(query, con)
    except Exception as e: return [], [], 'Invalid query!'
    return query_result.to_dict('records'), [{'name': col, 'id': col} for col in query_result.columns], None

# TAB 3

@callback(
    Output('primary_orgs', 'figure'),
    Output('primary_org_selection', 'data'),
    Input('primary_orgs', 'clickData')
)
def select_primary_org(click_data):
    if click_data is None or click_data['points'][0]['curveNumber'] == 0: return no_update, no_update
    point = click_data['points'][0]
    primary_org = point['customdata'][0]
    new_figure = Patch()
    sizes = [6] * len(df_primary_orgs)
    sizes[point['pointIndex']] = 10
    new_figure['data'][1]['marker']['size'] = sizes
    opacities = [.5] * len(df_primary_orgs)
    opacities[point['pointIndex']] = 1
    new_figure['data'][1]['marker']['opacity'] = opacities
    return new_figure, primary_org

@callback(
    Output('primary_orgs_info', 'hidden'),
    Output('primary_orgs_table', 'data'),
    Output('primary_orgs_table', 'columns'),
    Input('primary_org_selection', 'data')
)
def primary_orgs_summary(primary_org: str):
    if primary_org is None: return no_update, no_update, no_update
    df = (
        df_payments
        .merge(df_orgs, on='ORGANIZATION_ID')
        .merge(df_primary_orgs, on='PRIMARY_ORG_CODE')
        .query('PRIMARY_ORGANIZATION == @primary_org')
        .merge(df_clients, on='CLIENT_ID')
        .groupby('CLIENT_RECEIVER_NAME', as_index=False)
        ['AMOUNT']
        .agg(['sum', 'size'])
        .nlargest(5, 'sum')
        .round({'sum': 0})
    )
    df.columns = ['Client', 'Total Amount', 'Total Payments']
    return False, df.to_dict('records'), get_columns(df)

@callback(
    Output('download_query', 'data'),
    Input('download_button', 'n_clicks'),
    State('query_result', 'data')
)
def download_data(n_clicks, data):
    if n_clicks is None: return None
    return dcc.send_data_frame(pd.DataFrame(data).to_csv, 'query.csv', index=False)
