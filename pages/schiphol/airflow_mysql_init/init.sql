CREATE DATABASE airflow CHARACTER SET utf8 COLLATE utf8_unicode_ci;
CREATE USER 'airflow' IDENTIFIED BY 'airflow';
GRANT ALL PRIVILEGES ON airflow.* TO 'airflow';
