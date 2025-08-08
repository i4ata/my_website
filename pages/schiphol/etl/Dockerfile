FROM python:3.10-slim
WORKDIR /app
COPY ../start.sh /app/start.sh
COPY . /app
RUN apt-get update && apt-get install cron
RUN pip install pandas SQLAlchemy mysqlclient requests tqdm dotenv
EXPOSE 3306
CMD [ "python", "etl_script.py" ]
