docker compose down
docker volume rm schiphol_airflow_data schiphol_flights_data
docker compose up --build
