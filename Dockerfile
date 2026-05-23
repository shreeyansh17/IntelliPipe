# ──────────────────────────────────────────────────────────────────────────────
# IntelliPipe Multi-Stage Dockerfile
# ──────────────────────────────────────────────────────────────────────────────
# Stages:
#   base      — shared Python deps (all services)
#   api       — FastAPI application server
#   pipeline  — Kafka producer + alert processor
#   airflow   — Apache Airflow scheduler/webserver
#   spark     — PySpark streaming consumer
# ──────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.11
ARG DEBIAN_CODENAME=bookworm

# ── base: shared runtime ──────────────────────────────────────────────────────

FROM python:${PYTHON_VERSION}-slim-${DEBIAN_CODENAME} AS base

LABEL org.opencontainers.image.vendor="IntelliPipe"
LABEL org.opencontainers.image.title="IntelliPipe Data Quality Platform"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app

# System deps (shared across all stages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}

# Non-root user for security
RUN groupadd -r intellipipe && useradd -r -g intellipipe -d ${APP_HOME} intellipipe

# Install shared Python dependencies
COPY requirements/base.txt requirements/base.txt
RUN pip install --upgrade pip && pip install -r requirements/base.txt

COPY src/ src/
COPY pyproject.toml .

# ── api: FastAPI application ──────────────────────────────────────────────────

FROM base AS api

COPY requirements/api.txt requirements/api.txt
RUN pip install -r requirements/api.txt

# Run DB migrations then start Uvicorn
COPY alembic.ini .
COPY src/db/migrations/ src/db/migrations/

RUN chown -R intellipipe:intellipipe ${APP_HOME}
USER intellipipe

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["sh", "-c", \
    "alembic upgrade head && \
     uvicorn src.api.main:app \
       --host 0.0.0.0 \
       --port 8080 \
       --workers 4 \
       --loop uvloop \
       --access-log \
       --log-level info"]

# ── pipeline: Kafka + alert processor ────────────────────────────────────────

FROM base AS pipeline

COPY requirements/pipeline.txt requirements/pipeline.txt
RUN pip install -r requirements/pipeline.txt

RUN chown -R intellipipe:intellipipe ${APP_HOME}
USER intellipipe

# Default cmd overridden in docker-compose
CMD ["python", "-m", "src.pipeline.kafka_producer"]

# ── airflow: orchestration ────────────────────────────────────────────────────

FROM base AS airflow

ARG AIRFLOW_VERSION=2.8.0
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt"

RUN pip install "apache-airflow==${AIRFLOW_VERSION}" \
    --constraint "${CONSTRAINT_URL}" \
    --no-cache-dir

RUN pip install \
    apache-airflow-providers-postgres \
    apache-airflow-providers-slack \
    apache-airflow-providers-http \
    --no-cache-dir

COPY airflow/dags/ /opt/airflow/dags/

ENV AIRFLOW_HOME=/opt/airflow

RUN chown -R intellipipe:intellipipe /opt/airflow
USER intellipipe

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8090/health || exit 1

# ── spark: PySpark streaming ──────────────────────────────────────────────────

FROM base AS spark

ARG SPARK_VERSION=3.5.0
ARG SCALA_VERSION=2.12
ARG HADOOP_VERSION=3

# Install Java (required for Spark)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

COPY requirements/spark.txt requirements/spark.txt
RUN pip install -r requirements/spark.txt

# Download Spark binaries
RUN curl -sL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    | tar -xz -C /opt && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Download Kafka Spark connector
RUN curl -sL "https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_${SCALA_VERSION}/${SPARK_VERSION}/spark-sql-kafka-0-10_${SCALA_VERSION}-${SPARK_VERSION}.jar" \
    -o /opt/spark/jars/spark-sql-kafka.jar

RUN chown -R intellipipe:intellipipe ${APP_HOME}
USER intellipipe

CMD ["python", "-m", "src.pipeline.spark_consumer"]
