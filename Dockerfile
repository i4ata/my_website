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
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

COPY . /app
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "my_website:server"]
