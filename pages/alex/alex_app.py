from dash import Input, Output, dcc, html, dash_table, no_update, register_page, callback
import pandas as pd
import sqlite3
import os

styles = [
    {'if': {'filter_query': '{res} = 1', 'column_id': 'score'}, 'backgroundColor': 'yellowgreen'},
    {'if': {'filter_query': '{res} = -1', 'column_id': 'score'}, 'backgroundColor': 'tomato'},
    {'if': {'filter_query': '{res} = 0', 'column_id': 'score'}, 'backgroundColor': 'silver'}
]

league_table_styles = [
    {'if': {'filter_query': '{zone} = `Promotion`', 'column_id': 'zone'}, 'backgroundColor': 'yellowgreen'},
    {'if': {'filter_query': '{zone} = `Play-offs (up)`', 'column_id': 'zone'}, 'backgroundColor': 'skyblue'},
    {'if': {'filter_query': '{zone} = `Relegation`', 'column_id': 'zone'}, 'backgroundColor': 'beige'}
]

con = sqlite3.connect(os.path.join('pages', 'alex', 'fwp.db'), check_same_thread=False)

teams = pd.read_sql('select * from team', con=con, index_col='id')

register_page(__name__, path='/spartans', name='Spartans', order=20, icon='fluent:math-formula-16-regular')

layout = html.Div([
    html.H1('Opponent Analysis (Combined Counties League Division One)'),
    html.H3(f'Database last updated at: {pd.read_sql("select * from last_updated_at", con=con).values[0][0]}'),
    html.H4(f'Work in progress ...'),
    html.Br(), html.H3('Choose a team:'),
    dcc.Dropdown(id='name', options=[{'value': id, 'label': name} for id, name in zip(teams.index, teams['name'])], value=3063),
    html.Br(), html.H3('Last 5 matches:'),
    dash_table.DataTable(id='all_games', style_data_conditional=styles),
    html.Br(), html.H3('Last 5 home matches:'),
    dash_table.DataTable(id='home_games', style_data_conditional=styles + [{'if': {'column_id': 'home_team'}, 'fontWeight': 'bold'}]),
    html.Br(), html.H3('Last 5 away matches:'),
    dash_table.DataTable(id='away_games', style_data_conditional=styles + [{'if': {'column_id': 'away_team'}, 'fontWeight': 'bold'}]),
    html.Br(), html.H3('Top 3 goalscorers:'),
    dash_table.DataTable(id='goals', style_data_conditional=styles),
    html.Br(), html.H3('League table:'),
    dash_table.DataTable(id='league_table', style_data_conditional=league_table_styles),
    html.Br(), html.H3('Sequences:'),
    dash_table.DataTable(id='sequences')
])

@callback(
    Output('all_games', 'data'),
    Output('all_games', 'columns'),
    Output('all_games', 'style_data_conditional'),
    Input('name', 'value'),
)
def get_last_5_games(team_id):
    if team_id is None: return no_update, no_update, no_update
    with open(os.path.join('pages', 'alex', 'common_queries', 'get_last_5_games.sql')) as f: query = f.read().replace('3063', str(team_id))
    df: pd.DataFrame = pd.read_sql(query, con=con)
    team_name = teams.at[team_id, 'name']
    new_styles = styles + [
        {'if': {'filter_query': f'{{home_team}} = `{team_name}`', 'column_id': 'home_team'}, 'fontWeight': 'bold'},
        {'if': {'filter_query': f'{{away_team}} = `{team_name}`', 'column_id': 'away_team'}, 'fontWeight': 'bold'}
    ]
    return df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns if i != 'res'], new_styles

@callback(
    Output('home_games', 'data'),
    Output('home_games', 'columns'),
    Input('name', 'value'),
)
def get_last_5_home_games(team_id):
    if team_id is None: return no_update, no_update
    with open(os.path.join('pages', 'alex', 'common_queries', 'get_last_5_home_games.sql')) as f: query = f.read().replace('3063', str(team_id))
    df: pd.DataFrame = pd.read_sql(query, con=con)
    return df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns if i != 'res']

@callback(
    Output('away_games', 'data'),
    Output('away_games', 'columns'),
    Input('name', 'value'),
)
def get_last_5_away_games(team_id):
    if team_id is None: return no_update, no_update
    with open(os.path.join('pages', 'alex', 'common_queries', 'get_last_5_away_games.sql')) as f: query = f.read().replace('3063', str(team_id))
    df: pd.DataFrame = pd.read_sql(query, con=con)
    return df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns if i != 'res']

@callback(
    Output('goals', 'data'),
    Output('goals', 'columns'),
    Input('name', 'value'),
)
def get_top_3_goalscorers(team_id):
    if team_id is None: return no_update, no_update
    with open(os.path.join('pages', 'alex', 'common_queries', 'get_top_3_goalscorers.sql')) as f: query = f.read().replace('3063', str(team_id))
    df: pd.DataFrame = pd.read_sql(query, con=con)
    return df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns]

@callback(
    Output('league_table', 'data'),
    Output('league_table', 'columns'),
    Output('league_table', 'style_data_conditional'),
    Input('name', 'value'),
)
def get_league_table(team_id):
    if team_id is None: return no_update, no_update, no_update
    with open(os.path.join('pages', 'alex', 'common_queries', 'get_league_table.sql')) as f: query = f.read()
    df: pd.DataFrame = pd.read_sql(query, con=con)
    team_name = teams.at[team_id, 'name']
    new_styles = league_table_styles + [{'if': {'filter_query': f'{{name}} = `{team_name}`'}, 'fontWeight': 'bold', 'backgroundColor': 'darkgrey'}]
    return df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns], new_styles

@callback(
    Output('sequences', 'data'),
    Output('sequences', 'columns'),
    Input('name', 'value'),
)
def get_sequences(team_id):
    if team_id is None: return no_update, no_update, no_update
    with open(os.path.join('pages', 'alex', 'common_queries', 'get_sequences.sql')) as f: query = f.read().replace('3063', str(team_id))
    df: pd.DataFrame = pd.read_sql(query, con=con).T.reset_index()
    df.columns = ['stat', 'value']
    return df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns]
