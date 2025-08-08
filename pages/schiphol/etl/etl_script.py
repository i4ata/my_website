import requests
import re
import pandas as pd
from dotenv import load_dotenv
import sqlalchemy
import os
from typing import Literal
import time
from tqdm import tqdm

load_dotenv()
connection_str = 'mysql+pymysql://root:@flights_sql_server/schiphol'

def get_airports_coords() -> pd.DataFrame:
    url = 'https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets/airports-code@public/records'
    params = {
        'select': 'column_1, latitude, longitude',
        'limit': '100'
    }
    response = requests.get(url=url, params=params).json()
    response_list: list = response['results']
    for i in tqdm(range(response['total_count'] // 100 + 1)):
        response_list.extend(requests.get(url=url, params=params | {'offset': str((i+1)*100)}).json()['results'])
    return pd.DataFrame(response_list).rename(columns={'column_1': 'iata'}).round({'latitude': 6, 'longitude': 6})

def process_flights(flights_data: list[dict]) -> pd.DataFrame:

    flights_df = pd.DataFrame(flights_data)
    flights_df.to_csv('flights.csv')
    flights_df['destination'] = pd.json_normalize(flights_df['route'])['destinations'].str[0]
    flights_df = flights_df[[
        'id', 'scheduleDateTime', 'estimatedLandingTime', 'actualLandingTime', 'flightDirection', 
        'flightName', 'flightNumber', 'airlineCode', 'destination'
    ]]
    return flights_df

def process_airlines(airlines_data: list[dict]) -> pd.DataFrame:
    airlines_df = pd.DataFrame(airlines_data)
    return airlines_df

def process_destinations(destinations_data: list[dict]) -> pd.DataFrame:
    destinations_df = pd.DataFrame(destinations_data)
    destinations_df['publicName'] = pd.json_normalize(destinations_df['publicName'])['english']
    return destinations_df

def extract(endpoint: Literal['flights', 'destinations', 'airlines', 'aircrafttypes'] = 'flights') -> dict:

    url = 'https://api.schiphol.nl/public-flights/' + endpoint    
    headers = {
      'accept': 'application/json',
	  'resourceversion': 'v4',
      'app_id': os.getenv('APP_ID'),
	  'app_key': os.getenv('APP_KEY')
	}

    print('Reading page 0')
    response = requests.get(url=url, headers=headers)
    
    response_list: list = response.json()[endpoint]
    last_page_idx = int(re.search(r'page=(\d+)', requests.utils.parse_header_links(response.headers['link'])[1]['url']).groups()[0])
    for i in range(last_page_idx):
        if i != 0 and i % 10 == 0: print(f'Reading page {i+1}/{last_page_idx}')
        response_list.extend(requests.get(url=url, headers=headers, params={'page': str(i+1)}).json()[endpoint])
        if i != 0 and i % 100 == 0: 
            print('Waiting a minute to avoid too many requests ...')
            time.sleep(60)
    return {endpoint: response_list}

def load(df: pd.DataFrame, engine: sqlalchemy.engine) -> None:
    df.to_sql('flights', engine, if_exists='append', index=False)

if __name__ == '__main__':
    print('RUNNING THE SCRIPT')
    engine = sqlalchemy.create_engine(connection_str)
    with engine.connect() as con: con.execute(sqlalchemy.text('delete from flights'))
    flights_data = extract()['flights']
    flights_df = process_flights(flights_data)
    load(flights_df, engine)
