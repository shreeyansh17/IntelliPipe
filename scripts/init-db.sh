#!/bin/bash
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE airflow;
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE airflow TO intellipipe;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO intellipipe;
EOSQL
echo "Databases airflow and mlflow created"
