from dash import Input, Output, State, dcc, html, dash_table, no_update, Patch, ctx, register_page, callback
from dash.exceptions import PreventUpdate
from dash.dash_table.Format import Format
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Literal, List, Union
import traceback
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

bg_letters = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЮЯ'

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
            html.Li(f'Number of unique clients: {df_payments["CLIENT_ID"].nunique():,}'),
            html.Li(f'Number of unique organizations: {df_payments["ORGANIZATION_ID"].nunique():,}')
        ])
    ]),
    dcc.Store(id='tab_query'),
    dcc.Tabs(
        children=[
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
                            html.P('Choose one from the group with rarely occuring codes:'),
                            dcc.RadioItems(id='sebra_other_radio')
                        ],
                        id='sebra_other_container',
                        hidden=True
                    ),
                    html.Div(
                        children=[
                            html.Div(id='sebra_summary'),
                            html.Br(), html.P('Most paying organizations'),
                            dash_table.DataTable(id='sebra_orgs_table'),
                            html.Br(), html.P('Most receiving clients'),
                            dash_table.DataTable(id='sebra_clients_table')
                        ],
                        id='sebra_summary_container',
                        hidden=True
                    )
                ],
                label='Pay Codes',
                value='SEBRA_PAY_CODE'
            ),
            dcc.Tab(
                children=[
                    dcc.Store(id='date_selection'),
                    html.Div([
                        html.H2('Payments Over Time'),
                        html.Label('Choose whether you want to select a single day or a range'),
                        dcc.RadioItems(id='timeline_radio', options=[{'label': 'Single Day', 'value': False}, {'label': 'Range', 'value': True}], value=False),
                        html.Br(), html.Label('Click on the graph to see more information about that date/range'),
                        dcc.Graph(id='timeline', figure=make_timeline()),
                        html.Label('Format the graph'),
                        dcc.Checklist(id='timeline_options', options=['Log scale', 'Hide weekends & holidays']),
                        html.Br()
                    ]),
                    html.Div(
                        children=[
                            html.Div(id='timeline_output'),
                            html.P('Largest payments'),
                            dash_table.DataTable(id='timeline_table')
                        ],
                        id='timeline_info',
                        hidden=True
                    )
                ],
                label='Timeline',
                value='SETTLEMENT_DATE'
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
                label='Primary Organizations',
                value='PRIMARY_ORG_CODE'
            ),
            dcc.Tab(
                children=[
                    html.Div([
                        html.H2('Payments Per Day of the Week'),
                        html.P('This graph shows the median number of payments as well as the median payment amount per day per weekday. Interestingly enough, it can be seen that the fewest yet largest payments are made on Tuesday'),
                        dcc.Graph(figure=compare_weekdays())
                    ]),
                    html.H2('Run an arbitrary SQL query on the data'),
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
                label='Misc',
                value='Misc'
        )
        ], 
        id='tabs'
    ),
    html.Div(
        children=[
            # ORGANIZATIONS
            html.Br(), html.Button('Find a specific organization', id='orgs_button'), html.Br(),
            html.Div(
                children=[
                    html.P(id='orgs_filter_summary'),
                    dcc.RadioItems(id='orgs_initials', inline=True),
                    html.Label('Minimum Total Amount:'), dcc.Input(id='orgs_min_amount', type='number', min=0, step=1_000),
                    html.Label('Minimum Number of Payments:'), dcc.Input(id='orgs_min_payments', type='number', min=0, step=1),
                    html.Br(),
                    html.Label('Maximum Total Amount:'), dcc.Input(id='orgs_max_amount', type='number', min=0, step=1_000),
                    html.Label('Maximum Number of Payments:'), dcc.Input(id='orgs_max_payments', type='number', min=0, step=1),
                    html.Br(), html.Button('Submit', id='submit_orgs_filter'), html.Br(),
                    html.P(id='orgs_filter_output_summary')
                ],
                id='filter_orgs',
                hidden=True
            ),
            html.Div(
                children=[
                    dcc.Dropdown(id='orgs_dropdown'),
                    html.Br(), dash_table.DataTable(id='individual_org_table'), html.Br()
                ],
                id='individual_org_container',
                hidden=True,
            ),

            # CLIENTS
            html.Br(), html.Button('Find a specific client', id='clients_button'), html.Br(),
            html.Div(
                children=[
                    html.P(id='clients_filter_summary'),
                    dcc.RadioItems(id='clients_initials', inline=True),
                    html.Label('Minimum Total Amount: '), dcc.Input(id='clients_min_amount', type='number', min=0, step=1_000),
                    html.Label('Minimum Number of Payments:'), dcc.Input(id='clients_min_payments', type='number', min=0, step=1),
                    html.Br(),
                    html.Label('Maximum Total Amount:'), dcc.Input(id='clients_max_amount', type='number', min=0, step=1_000),
                    html.Label('Maximum Number of Payments:'), dcc.Input(id='clients_max_payments', type='number', min=0, step=1),
                    html.Br(), html.Button('Submit', id='submit_clients_filter'), html.Br(),
                    html.P(id='clients_filter_output_summary')
                ],
                id='filter_clients',
                hidden=True
            ),
            html.Div(
                children=[
                    dcc.Dropdown(id='clients_dropdown'),
                    html.Br(), dash_table.DataTable(id='individual_client_table'), html.Br()
                ],
                id='individual_client_container',
                hidden=True,
            ),
        ],
        id='filter_container',
        hidden=True
    )
])

def get_columns(df: pd.DataFrame): 
    return (
        [{'name': col, 'id': col} for col in df.select_dtypes([object, 'datetime']).columns]
        +
        [{'name': col, 'id': col, 'type': 'numeric', 'format': Format().group(True)} for col in df.select_dtypes('number').columns]
    )

@callback(
    Output('filter_container', 'hidden'),
    Input('tabs', 'value'),
    Input('tab_query', 'data')
)
def update_filter(tab, query):
    if tab is None and query is None: return no_update
    return ctx.triggered_id == 'tabs' and query is None

@callback(
    Output('tab_query', 'data'),
    Input('tabs', 'value'),
    Input('code_selection', 'data'),
    Input('date_selection', 'data'),
    Input('primary_org_selection', 'data')
)
def set_query(tab, code, date, primary_org_code):
    if all((tab is None, code is None, date is None, primary_org_code is None)): return no_update
    if (ctx.triggered_id == 'tabs' and tab == 'SEBRA_PAY_CODE' and code is not None) or ctx.triggered_id == 'code_selection':
        return f'SEBRA_PAY_CODE == {code}'
    if (ctx.triggered_id == 'tabs' and tab == 'SETTLEMENT_DATE' and date is not None) or ctx.triggered_id == 'date_selection': 
        return f'SETTLEMENT_DATE == @pd.Timestamp("{date[0]}")' if len(date) == 1 else f'"{date[0]}" < SETTLEMENT_DATE < "{date[1]}"'
    if (ctx.triggered_id == 'tabs' and tab == 'PRIMARY_ORG_CODE' and primary_org_code is not None) or ctx.triggered_id == 'primary_org_selection': 
        return f'PRIMARY_ORG_CODE == {primary_org_code}'

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
def click_on_pie(click_data_pie, click_data_bar, radio_code, code_selection):

    # Initial call. No one has clicked on any of the graphs or the radio: don't do anything
    if click_data_pie is None and click_data_bar is None and radio_code is None: return [no_update] * 5
    
    # If the callback is triggered by clicking one of the 'other' codes: Don't change the plots
    if ctx.triggered_id == 'sebra_other_radio':
        fig_pie, fig_bar = no_update, no_update
        code = radio_code

    # If the callback is triggered by clicking on either graph
    else:

        # Extract the selected code
        code = (click_data_pie if ctx.triggered_id == 'pie' else click_data_bar)['points'][0]['label']

        # If the selected code is the same as the old one
        if code_selection is not None and code == str(code_selection): return [no_update] * 5

        # Update the graphs
        fig_pie = pull_pie_sector(code)
        fig_bar = highlight_bar(code)

    # If the callback is triggered by clicking the "other" section of the pie, render the radio
    if ctx.triggered_id == 'pie' and code == pies['other_label']:
        options = [
            {'label': f'{other_code}: {df_sebra.loc[other_code, "DESCRIPTION"]}', 'value': other_code}
            for other_code in pies['to_remove']
        ]
        return no_update, fig_pie, fig_bar, False, options
    
    return (
        int(code), fig_pie, fig_bar, 
        True if ctx.triggered_id == 'pie' else no_update, 
        [] if ctx.triggered_id == 'pie' else no_update,
    )

def pull_pie_sector(code: str) -> Patch:
    fig = Patch()
    pull = [0] * len(pies['labels'])
    if ctx.triggered_id == 'bar' and int(code) in pies['to_remove']:
        pull[pies['labels'].index(pies['other_label'])] = .2
    else:
        pull[pies['labels'].index(int(code) if code != pies['other_label'] else code)] = .2
    fig['data'][0]['pull'] = pull
    fig['data'][1]['pull'] = pull
    return fig

def highlight_bar(code: str) -> Patch:
    fig = Patch()
    colors = ['#636efa'] * len(pay_codes)
    if code == pies['other_label']: 
        for i in pies['to_remove']: colors[pay_codes.index(i)] = 'red'
    else:
        colors[pay_codes.index(int(code))] = 'red'
    fig['data'][0]['marker']['color'] = colors
    return fig

@callback(
    Output('sebra_summary', 'children'),
    Output('sebra_summary_container', 'hidden'),
    
    Output('sebra_orgs_table', 'data'),
    Output('sebra_orgs_table', 'columns'),

    Output('sebra_clients_table', 'data'),
    Output('sebra_clients_table', 'columns'),

    Input('code_selection', 'data')
)
def pie_summary(code: int):
    if code is None: return [no_update] * 6
    
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
        orgs_summary.to_dict('records'), get_columns(orgs_summary), 
        clients_summary.to_dict('records'), get_columns(clients_summary)
    )

# TAB 2

def draw_rectangle(x1: str, x2: Optional[str] = None) -> dict:
    get_datetime = lambda x: datetime(*map(int, x.split('-')))
    date1 = get_datetime(x1)
    date2 = date1 if x2 is None else get_datetime(x2)
    time_delta = timedelta(hours=12)
    return {
        'fillcolor': 'green',
        'layer': 'below',
        'line': {'width': 0},
        'opacity': 0.3,
        'type': 'rect',
        'x0': date1 - time_delta,
        'x1': date2 + time_delta,
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
    Input('timeline_radio', 'value'),
    State('date_selection', 'data')
)
def plot_timeline(options, click_data, range, date):
    
    if options is None and click_data is None: return no_update, no_update
    if ctx.triggered_id == 'timeline_options':
        fig = make_timeline('Hide weekends & holidays' in options, 'Log scale' in options)
        if date is not None: fig.add_shape(**draw_rectangle(*date))
        return fig, no_update

    fig = Patch()
    if ctx.triggered_id == 'timeline_radio':
        if range: return no_update, no_update
        fig['layout']['shapes'] = [draw_rectangle(date[0])]
        return fig, [date[0]]

    new_date = click_data['points'][0]['x']
    if range and date is not None: new_date = sorted((new_date, date[0]))
    else: new_date = [new_date]
    fig['layout']['shapes'] = [draw_rectangle(*new_date)]
    return fig, new_date

@callback(
    Output('timeline_table', 'data'),
    Output('timeline_table', 'columns'),
    Output('timeline_info', 'hidden'),
    Input('date_selection', 'data')
)
def largest_payments_today(date: List[str]):
    if date is None: return no_update, no_update, no_update
    data = (
        df_payments
        .query(f'SETTLEMENT_DATE == @pd.Timestamp("{date[0]}")' if len(date) == 1 else f'"{date[0]}" < SETTLEMENT_DATE < "{date[1]}"')
        .nlargest(5, 'AMOUNT')
        .merge(df_clients, on='CLIENT_ID')
        .merge(df_orgs, on='ORGANIZATION_ID')
        .merge(df_primary_orgs, on='PRIMARY_ORG_CODE')
        [['SETTLEMENT_DATE', 'PRIMARY_ORGANIZATION', 'ORGANIZATION', 'AMOUNT', 'CLIENT_RECEIVER_NAME', 'SEBRA_PAY_CODE', 'REASON1', 'REASON2']]
        .round({'AMOUNT': 0})
    )
    data['SETTLEMENT_DATE'] = data['SETTLEMENT_DATE'].dt.strftime('%d-%m-%Y')
    data.columns = ['Date', 'Primary Organization', 'Organization', 'Amount', 'Client', 'SEBRA Pay Code', 'Reason 1', 'Reason 2']
    return data.to_dict('records'), get_columns(data), False

# TAB 3

@callback(
    Output('primary_orgs', 'figure'),
    Output('primary_org_selection', 'data'),
    Output('primary_orgs_output', 'children'),
    Input('primary_orgs', 'clickData')
)
def select_primary_org(click_data):
    if click_data is None or click_data['points'][0]['curveNumber'] == 0: return [no_update] * 3
    point = click_data['points'][0]
    primary_org = point['customdata'][0]
    new_figure = Patch()
    sizes = [6] * len(df_primary_orgs)
    sizes[point['pointIndex']] = 10
    new_figure['data'][1]['marker']['size'] = sizes
    opacities = [.5] * len(df_primary_orgs)
    opacities[point['pointIndex']] = 1
    new_figure['data'][1]['marker']['opacity'] = opacities

    primary_org_code = df_primary_orgs[df_primary_orgs['PRIMARY_ORGANIZATION'] == primary_org].index[0]
    df = df_orgs.query('PRIMARY_ORG_CODE == @primary_org_code').merge(df_payments, on='ORGANIZATION_ID')
    summary = html.Div([
        html.P(f'You selected {primary_org}'),
        html.Ul([
            html.Li(f'Unique organizations: {df["ORGANIZATION"].nunique():,}'),
            html.Li(f'Unique clients: {df["CLIENT_ID"].nunique():,}'),
            html.Li(f'Total payments: {len(df):,}'),
            html.Li(f'Total payment amount: {round(df["AMOUNT"].sum()):,}')
        ])
    ])
    return new_figure, primary_org_code, summary

@callback(
    Output('primary_orgs_info', 'hidden'),
    Output('primary_orgs_table', 'data'),
    Output('primary_orgs_table', 'columns'),
    Input('primary_org_selection', 'data')
)
def primary_orgs_summary(primary_org_code: int):
    if primary_org_code is None: return no_update, no_update, no_update
    df = (
        df_payments
        .merge(df_orgs, on='ORGANIZATION_ID')
        .query('PRIMARY_ORG_CODE == @primary_org_code')
        .merge(df_clients, on='CLIENT_ID')
        .groupby('CLIENT_RECEIVER_NAME', as_index=False)
        ['AMOUNT']
        .agg(['sum', 'size'])
        .nlargest(5, 'sum')
        .round({'sum': 0})
    )
    df.columns = ['Client', 'Total Amount', 'Total Payments']
    return False, df.to_dict('records'), get_columns(df)

# SUMMARIES

@callback(
    Output('filter_orgs', 'hidden'),
    Output('orgs_filter_summary', 'children'),
    Output('orgs_initials', 'options'),
    Output('filter_clients', 'hidden'),
    Output('clients_filter_summary', 'children'),
    Output('clients_initials', 'options'),
    Input('orgs_button', 'n_clicks'),
    Input('clients_button', 'n_clicks'),
    Input('tabs', 'value'),
    Input('tab_query', 'data')
)
def enable_finding_organizations_or_clients(orgs: int, clients: int, tab: str, query: str):

    if orgs is None and clients is None: return [no_update] * 6
    if ctx.triggered_id in ('tabs', 'tab_query'): return True, no_update, [], True, no_update, []

    df_queried = df_payments.merge(df_orgs, on='ORGANIZATION_ID').merge(df_clients, on='CLIENT_ID').query(query)
    
    if ctx.triggered_id == 'orgs_button':
        unique_orgs_initials = np.sort(
            pd.Series(df_queried['ORGANIZATION'].unique())
            .str.upper().str.extract(f'([{bg_letters}])', expand=False).unique()
        )
        nunique_orgs = df_queried['ORGANIZATION_ID'].nunique()
        return (
            False, f'There are {nunique_orgs} unique organizations. You can filter them first', unique_orgs_initials, 
            no_update, no_update, no_update
        )
    
    else: 
        unique_clients_initials = np.sort(
            pd.Series(df_queried['CLIENT_RECEIVER_NAME'].unique())
            .str.upper().str.extract(f'([{bg_letters}])', expand=False).unique()
        )
        nunique_clients = df_queried['CLIENT_ID'].nunique()
        return (
            no_update, no_update, no_update,
            False, f'There are {nunique_clients} unique clients. You can filter them first', unique_clients_initials
        )

@callback(
    Output('individual_org_container', 'hidden'),
    Output('orgs_dropdown', 'options'),
    Output('orgs_dropdown', 'value'),
    Output('orgs_initials', 'value'),
    Output('orgs_filter_output_summary', 'children'),

    Input('submit_orgs_filter', 'n_clicks'),
    Input('tabs', 'value'),    
    Input('tab_query', 'data'),

    State('orgs_initials', 'value'),
    State('orgs_min_amount', 'value'),
    State('orgs_min_payments', 'value'),
    State('orgs_max_amount', 'value'),
    State('orgs_max_payments', 'value'),
)
def select_specifig_org(
    submit_org_filter: Optional[int], tab: str,
    global_query: str,
    initial: str,
    min_amount: Optional[int], min_payments: Optional[int], max_amount: Optional[int], max_payments: Optional[int] 
):
    # Initial call: All is None
    if submit_org_filter is None and global_query is None: return [no_update] * 5
    if ctx.triggered_id in ('tabs', 'tab_query'): return True, [], None, None, None

    df = df_payments.merge(df_orgs, on='ORGANIZATION_ID').merge(df_clients, on='CLIENT_ID').query(global_query)
    if initial is not None:
        df = df[df['ORGANIZATION'].str.upper().str.extract(f'([{bg_letters}])', expand=False) == initial]
    # If we filter the orgs
    aggregation: pd.DataFrame = (
        df
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
    return False, orgs, no_update, no_update, f'Filtered {len(orgs)} organizations'

@callback(
    Output('individual_client_container', 'hidden'),
    Output('clients_dropdown', 'options'),
    Output('clients_dropdown', 'value'),
    Output('clients_initials', 'value'),
    Output('clients_filter_output_summary', 'children'),

    Input('submit_clients_filter', 'n_clicks'),
    Input('tabs', 'value'),
    Input('tab_query', 'data'),

    State('clients_initials', 'value'),
    State('clients_min_amount', 'value'),
    State('clients_min_payments', 'value'),
    State('clients_max_amount', 'value'),
    State('clients_max_payments', 'value'),
)
def select_specifig_client(
    submit_client_filter: Optional[int], tab: str,
    global_query: str,
    initial: str,
    min_amount: Optional[int], min_payments: Optional[int], max_amount: Optional[int], max_payments: Optional[int] 
):
    # Initial call: All is None
    if submit_client_filter is None and global_query is None: return [no_update] * 5
    if ctx.triggered_id in ('tabs', 'tab_query'): return True, [], None, None, None
    
    df = df_payments.merge(df_orgs, on='ORGANIZATION_ID').merge(df_clients, on='CLIENT_ID').query(global_query)
    if initial is not None:
        df = df[df['CLIENT_RECEIVER_NAME'].str.upper().str.extract(f'([{bg_letters}])', expand=False) == initial]
    # If we filter the clients
    aggregation: pd.DataFrame = (
        df
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
    Output('individual_org_table', 'data'),
    Output('individual_org_table', 'columns'),
    Input('orgs_dropdown', 'value'),
    State('tab_query', 'data')
)
def display_org_info(org: Optional[str], query: str):

    if org is None: return [], []
    org_id = df_orgs.query('ORGANIZATION == @org').iloc[0].name
    data = (
        df_payments
        .query('ORGANIZATION_ID == @org_id')
        .merge(df_clients, on='CLIENT_ID')
        .merge(df_orgs, on='ORGANIZATION_ID')
        .query(query)
        [['SETTLEMENT_DATE', 'AMOUNT', 'CLIENT_RECEIVER_NAME', 'REASON1', 'REASON2', 'SEBRA_PAY_CODE']]
    )
    # Hacky way to avoid the datatable to show the hours, minutes, and seconds
    data['SETTLEMENT_DATE'] = data['SETTLEMENT_DATE'].dt.strftime('%d-%m-%Y')
    data.columns = ['Settlement Date', 'Payment Amount', 'Client Name', 'Reason 1', 'Reason 2', 'SEBRA Code']
    return data.to_dict('records'), get_columns(data)

@callback(
    Output('individual_client_table', 'data'),
    Output('individual_client_table', 'columns'),
    Input('clients_dropdown', 'value'),
    State('tab_query', 'data')
)
def display_client_info(client: Optional[str], query: str):
    
    if client is None: return [], []
    client_id = df_clients.query('CLIENT_RECEIVER_NAME == @client').iloc[0].name
    data = (
        df_payments
        .query('CLIENT_ID == @client_id')
        .merge(df_clients, on='CLIENT_ID')
        .merge(df_orgs, on='ORGANIZATION_ID')
        .merge(df_primary_orgs, on='PRIMARY_ORG_CODE')
        .query(query)
        [['SETTLEMENT_DATE', 'AMOUNT', 'ORGANIZATION', 'PRIMARY_ORGANIZATION', 'REASON1', 'REASON2', 'SEBRA_PAY_CODE']]
    )
    # Hacky way to avoid the datatable to show the hours, minutes, and seconds
    data['SETTLEMENT_DATE'] = data['SETTLEMENT_DATE'].dt.strftime('%d-%m-%Y')
    data.columns = ['Settlement Date', 'Payment Amount', 'Organization', 'Primary Organization', 'Reason 1', 'Reason 2', 'SEBRA Code']
    return data.to_dict('records'), get_columns(data)

# TAB 4

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

@callback(
    Output('download_query', 'data'),
    Input('download_button', 'n_clicks'),
    State('query_result', 'data')
)
def download_data(n_clicks, data):
    if n_clicks is None: return None
    return dcc.send_data_frame(pd.DataFrame(data).to_csv, 'query.csv', index=False)
