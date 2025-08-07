import requests
import pandas as pd
from tqdm import tqdm
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()
url = 'https://football-web-pages1.p.rapidapi.com/'
headers = {
	"x-rapidapi-key": os.getenv('API_KEY'),
	"x-rapidapi-host": "football-web-pages1.p.rapidapi.com"
}
league_params = {'comp': '119'}
con = sqlite3.connect('fwp.db')

def extract_competitions():
    """
    Source data for the individual competitions (id and name)
    To be ran only once
    """

    endpoint = url + 'competitions.json'
    response = requests.get(endpoint, headers=headers)
    df = pd.DataFrame(response.json()['competitions']).set_index('id').drop(columns='rounds')
    df.columns = df.columns.str.replace('-', '_')
    df.to_sql('competition', con=con, if_exists='replace')

def extract_team():
    """
    Source data for each team in the league (e.g. id, name, address, facebook page, etc.)
    To be ran only once
    """

    endpoint = url + 'teams.json'
    response = requests.get(endpoint, headers=headers, params=league_params).json()['teams']

    endpoint = url + 'team.json'
    team_ids = [team['id'] for team in response]
    team_infos = [requests.get(endpoint, headers=headers, params={'team': str(team_id)}).json()['team'] for team_id in tqdm(team_ids)]
    df = pd.DataFrame(team_infos).set_index('id')
    df.columns = df.columns.str.replace('-', '_')
    df.to_sql('team', con=con, if_exists='replace')

def extract_fixtures():
    """
    Source data for each scheduled match in the current month
    """

    endpoint = url + 'fixtures-results.json'
    response = requests.get(endpoint, headers=headers, params=league_params).json()['fixtures-results']['matches']
    matches = []
    for match in response:
        for team in ('home', 'away'):
            match[team+'-team-score'] = match[team+'-team']['score']
            match[team+'-team-half-time-score'] = match[team+'-team'].get('half-time-score', 0)
            match[team+'-team'] = match[team+'-team']['id']
        match['FT'] = match['status']['short'] == 'FT'
        match.pop('status')
        matches.append(match)

    df = pd.DataFrame(matches).set_index('id')
    df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.drop(columns=['time', 'competition'])
    df.columns = df.columns.str.replace('-', '_')
    df.to_sql('fixture', con=con, if_exists='replace')

def extract_table():
    """
    Source data for the current state of the league table
    """

    endpoint = url + 'league-table.json'
    response = requests.get(endpoint, headers=headers, params=league_params)
    teams = []
    for team in response.json()['league-table']['teams']:
        team.pop('name')
        team['goal-difference'] = team['all-matches']['goal-difference']
        for t in ('all-matches', 'away-matches', 'home-matches'):
            team[t+'-played'] = team[t]['played']
            team[t+'-won'] = team[t]['won']
            team[t+'-drawn'] = team[t]['drawn']
            team[t+'-lost'] = team[t]['lost']
            team.pop(t)
        teams.append(team)
    df = pd.DataFrame(teams).set_index('id')
    df.columns = df.columns.str.replace('-', '_')
    df.to_sql('league_table', con=con, if_exists='replace')

def extract_league_progress():
    """
    Source data for each team's current league progress based on past rounds
    """

    endpoint = url + 'league-progress.json'
    team_ids = pd.read_sql('select * from team', con=con, index_col='id').index
    teams = []
    for team_id in team_ids:
        team_progress = requests.get(endpoint, headers=headers, params={'team': str(team_id)}).json()['league-progress']['progress']
        teams.append(pd.DataFrame(team_progress).assign(team=team_id))
    df = pd.concat(teams)
    df.columns = df.columns.str.replace('-', '_')
    df.to_sql('progress', con=con, index=False, if_exists='replace')

def extract_goals():
    """
    Source data for each goal so far
    """

    endpoint = url + 'goalscorers.json'
    team_ids = pd.read_sql('select * from team', con=con, index_col='id').index
    goals = []
    for team_id in tqdm(team_ids):
        players = requests.get(endpoint, headers=headers, params={'team': str(team_id)}).json()['goalscorers']['players']
        for player in players:
            player_goals = [{'match': goal['match']['id'], 'minute': goal['minute']} for goal in player['goals']]
            goals.append(pd.DataFrame(player_goals).assign(player=player['id'], team=team_id))
    df = pd.concat(goals)
    df.columns = df.columns.str.replace('-', '_')
    df.to_sql('goal', con=con, index=False, if_exists='replace')

def extract_appearances():
    """
    Source data for each player's appearances so far (incl. substitutions)
    """

    endpoint = url + 'appearances.json'
    team_ids = pd.read_sql('select * from team', con=con, index_col='id').index
    dfs_appearances, player_names, subs = [], [], []
    for team_id in tqdm(team_ids):
        players = requests.get(endpoint, headers=headers, params={'team': str(team_id)}).json()['appearances']['players']
        for player in players:
            player_appearances = [{'match': appearance['match']['id'], 'shirt': appearance['shirt']} for appearance in player['appearances']]
            
            # Check for substitutions
            for appearance in player['appearances']:
                if 'replaced' not in appearance: continue
                sub = appearance['replaced']
                
                # Player id not given ...
                out = [
                    p['id'] for p in players for a in p['appearances'] 
                    if a['match']['id'] == appearance['match']['id'] and a['shirt'] == sub['shirt']
                ][0]

                subs.append({
                    'team_id': team_id, 'match': appearance['match']['id'], 'in': 
                    player['id'], 'minute': sub['minute'], 'out': out
                })

            dfs_appearances.append(pd.DataFrame(player_appearances).assign(player=player['id'], team=team_id))
            player_names.append({'id': player['id'], 'first-name': player['first-name'], 'last-name': player['last-name']})
    
    df = pd.concat(dfs_appearances)
    df.columns = df.columns.str.replace('-', '_')
    df.to_sql('appearance', con=con, index=False, if_exists='replace')

    df_players = pd.DataFrame(player_names).drop_duplicates().set_index('id')
    df_players.columns = df_players.columns.str.replace('-', '_')
    df_players.to_sql('player', con=con, if_exists='replace')

    df_subs = pd.DataFrame(subs)
    df_subs.columns = df_subs.columns.str.replace('-', '_')
    df_subs.to_sql('substitution', con=con, index=False, if_exists='replace')

def extract_sequences():
    """
    Source data for the teams' individual sequences (e.g. a running total of matches without conceding a goal)
    """
    endpoint = url + 'sequences.json'
    team_ids = pd.read_sql('select * from team', con=con, index_col='id').index
    seqs = []
    for team_id in tqdm(team_ids):
        sequences = requests.get(endpoint, headers=headers, params={'team': str(team_id)}).json()['sequences']['sequences']
        team_sequence = {'team': team_id}
        for sequence in sequences:
            if sequence['time-period'] != 'Current': continue
            key = sequence['description'].lower().replace(' ', '_') + '_' + sequence['type'].lower().replace(' ', '_')
            team_sequence[key] = sequence['matches']
        seqs.append(team_sequence)
    df = pd.DataFrame(seqs).set_index('team')
    df.columns = df.columns.str.replace('-', '_')
    df.to_sql('sequence', con=con, if_exists='replace')

# extract_appearances()
