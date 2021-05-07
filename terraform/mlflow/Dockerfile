FROM python:3.8-slim

RUN groupadd mlflow \
    && useradd -r --home-dir /home/mlflow --shell /bin/bash -g mlflow mlflow \
    && mkdir /home/mlflow \
    && chown -R mlflow:mlflow /home/mlflow \
    && chmod 755 /home/mlflow

USER mlflow

ENV PATH=/home/mlflow/.local/bin:$PATH
RUN pip install --user mlflow pymysql cryptography

WORKDIR /home/mlflow

ENTRYPOINT mlflow server -h 0.0.0.0 \
    --backend-store-uri=mysql+pymysql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME} \
    --default-artifact-root='./mlruns'  \
    && /bin/bash
