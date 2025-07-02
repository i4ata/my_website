FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install pandas SQLAlchemy mysqlclient requests tqdm dotenv
EXPOSE 3306
CMD [ "python", "etl_script.py" ]
