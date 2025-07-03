from datetime import datetime
from airflow.decorators import dag, task
import pandas as pd
import sqlalchemy
import sys
from datetime import datetime

sys.path.append('/opt/airflow')

from etl import etl_script
engine = sqlalchemy.create_engine(etl_script.connection_str)

@dag(
    schedule='0 7 * * *',
    start_date=datetime(2025, 7, 3),
    catchup=False,
    tags=['schiphol']
)
def schiphol_test_dag():

    @task()
    def clean_up() -> None:
        print('Removing old flights')
        with engine.connect() as con: con.execute(sqlalchemy.text('delete from flights'))

    @task()
    def extract() -> list[dict]:
        return etl_script.extract()['flights']

    @task()
    def transform(flights_data: list[dict]) -> pd.DataFrame:
        return etl_script.process_flights(flights_data)
    
    @task()
    def load(flights_df: pd.DataFrame) -> None:
        etl_script.load(flights_df, engine)

    flights_data = extract()
    flights_df = transform(flights_data)
    clean_up()
    load(flights_df)

schiphol_test_dag()
