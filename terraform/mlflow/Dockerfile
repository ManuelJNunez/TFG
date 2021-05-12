FROM python:3.8-slim

RUN groupadd mlflow \
    && useradd -r --home-dir /home/mlflow --shell /bin/bash -g mlflow mlflow \
    && mkdir /home/mlflow \
    && chown -R mlflow:mlflow /home/mlflow \
    && chmod 755 /home/mlflow

USER mlflow

ENV PATH=/home/mlflow/.local/bin:$PATH
RUN pip install --user mlflow pymysql cryptography boto3

WORKDIR /home/mlflow

ENTRYPOINT mlflow server -h 0.0.0.0 \
    --backend-store-uri=mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@${MLFLOW_DBHOST}:${MYSQL_PORT}/${MYSQL_DATABASE} \
    --default-artifact-root=${MLFLOW_ARTIFACTS_URI}
