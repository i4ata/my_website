from dash import dcc, html, register_page, callback, Input, Output
from sqlalchemy import create_engine
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

SCHIPHOL_LON = 4.674
SCHIPHOL_LAT = 52.309

# engine = create_engine('mysql+pymysql://root:@flights_sql_server/schiphol')
# engine = create_engine('mysql+pymysql://root:@localhost:3306/schiphol')

register_page(__name__, path='/schiphol', name='Schiphol Airport ETL', order=2, icon='fluent-emoji:airplane')

df_flights: Optional[pd.DataFrame] = None
df_airlines: Optional[pd.DataFrame] = None
df_destinations: Optional[pd.DataFrame] = None

# Called when the page is loaded
@callback(
    Output('flights', 'figure'),
    Output('airlines', 'figure'),
    Output('destinations', 'figure'),
    Input('url', 'pathname'))
def load_page(pathname: Optional[str]):
    global df_flights, df_airlines, df_destinations

    engine = create_engine('mysql+pymysql://root:@flights_sql_server/schiphol')
    read_sql_table = lambda table, index: pd.read_sql(f'select * from {table}', engine, index_col=index)

    if pathname is not None and pathname == '/schiphol':
        df_flights = read_sql_table('flights', index='id')
        df_airlines = read_sql_table('airlines', index='nvls')
        df_destinations = read_sql_table('destinations', index='iata')
        return plot_flights(), plot_airlines(), plot_destinations()
    return {}, {}, {}

# https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(
    lon1: pd.Series | np.ndarray | float, 
    lat1: pd.Series | np.ndarray | float, 
    lon2: pd.Series | np.ndarray | float, 
    lat2: pd.Series | np.ndarray | float
) -> pd.Series | float:
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371
    return c * r

def plot_airlines() -> go.Figure:
    df = (
        df_flights
        .groupby(['airlineCode', 'flightDirection'])
        .size()
        .unstack(fill_value=0)
        .merge(df_airlines, left_on='airlineCode', right_on='nvls')
        .rename(columns={'publicName': 'Airline', 'A': 'Arrivals', 'D': 'Departures'})
    )
    fig = px.bar(
        data_frame=df.reset_index(), 
        y=['Arrivals', 'Departures'], 
        x='Airline', 
        title='Flights per Airline', 
        labels={
            'variable': 'Flight Direction',
            'value': 'Count'
        }
    )
    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
    return fig

def plot_flights() -> go.Figure:
    df = (
        df_flights
        .groupby([df_flights['scheduleDateTime'].dt.hour, 'flightDirection'])
        .size()
        .unstack()
        .reindex(range(24), fill_value=0)
        .rename(columns={'A': 'Arrivals', 'D': 'Departures'})
        .reset_index(names='hour')
    )
    fig = px.bar(
        df, x='hour', y=['Arrivals', 'Departures'], 
        title='Flights throughout the day', 
        labels={'hour': 'Hour of the day', 'value': 'Count', 'variable': 'Flight Direction'}
    )
    fig.update_xaxes(dtick=1)
    return fig

def plot_destinations() -> go.Figure:
    fig = go.Figure()
    df = df_destinations[df_destinations.index.isin(df_flights['destination'])]
    if df.empty: return fig
    fig.add_trace(go.Scattergeo(
        lon=[SCHIPHOL_LON],
        lat=[SCHIPHOL_LAT],
        mode='markers',
        marker=dict(
            size=5,
            color='red'
        ),
        hovertemplate='<b>SCHIPHOL</b><extra></extra>'
    ))

    df['distance'] = haversine(df['longitude'], df['latitude'], SCHIPHOL_LON, SCHIPHOL_LAT).round()
    df[['arrivals', 'departures']] = (
        df.merge(df_flights, left_on='iata', right_on='destination').groupby(['destination', 'flightDirection']).size().unstack(fill_value=0)
    )
    
    fig.add_trace(go.Scattergeo(
        lon=df['longitude'],
        lat=df['latitude'],
        mode='markers',
        marker=dict(
            size=3,
            color='orange'
        ),
        customdata=df[['publicName', 'distance', 'arrivals', 'departures']],
        hovertemplate='<b>%{customdata[0]}</b><br>Distance: %{customdata[1]:,}km<br>Arrivals: %{customdata[2]}<br>Departures: %{customdata[3]}<extra></extra>'
    ))

    lons = np.full(3*len(df), SCHIPHOL_LON)
    lons[1::3] = df['longitude']
    lons[2::3] = None

    lats = np.full(3*len(df), SCHIPHOL_LAT)
    lats[1::3] = df['latitude']
    lats[2::3] = None

    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='lines',
        hoverinfo='skip',
        line=dict(width=1, color='red'),
        opacity=.5
    ))
    fig.update_layout(
        showlegend = False,
        title='Flight Connections',
        geo = dict(
            showland=True,
            showcountries=True,
            showocean=True,
            landcolor='green',
            lakecolor='blue',
            oceancolor='blue',
            countrywidth = 0.5,
            projection_type='orthographic'
        ),
        width=1000,
        height=1000
    )
    return fig

with open('pages/schiphol/dash_app/text.md') as f: text = f.read()

layout = html.Div([
    dcc.Markdown(text, mathjax=True, link_target='_blank', dangerously_allow_html=True),
    dcc.Tabs([
        dcc.Tab(dcc.Graph(id='flights'), label='Flights'),
        dcc.Tab(dcc.Graph(id='airlines'), label='Airlines'),
        dcc.Tab(dcc.Graph(id='destinations'), label='Destinations')
    ])
])
