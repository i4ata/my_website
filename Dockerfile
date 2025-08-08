FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app

# Weird ass chatgpt way to ensure scikit-survival is installed
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    cron \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt
RUN echo "0 3 * * * python -m pages.schiphol.etl.etl_script" | crontab -

COPY . /app
EXPOSE 5000
RUN chmod +x /app/start.sh
CMD /app/start.sh
