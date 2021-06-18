# docker run -it --rm -p 5000:5000 -v /local/path:/mlflow --name mlflow-server atcommons/mlflow-server

# The mlflow instance can be configured through environment variables

# BACKEND_URI: backend store where experiments and all metadata are stored. 
# You may provide a file path or a database URI. Defaults to a local sqlite database.
# ARTIFACT_ROOT: root directory of mlflow's artifact store. 
# Note that serving of files in the artifact store is not managed by the mlflow server, 
# so you have to provide a storage path that is accessible by the server as well as all clients. 
# While using local paths is possible, it is not very useful in practice. 
# Feasible backends are Amazon S3, Azure Blob Storage, Google Cloud Storage and more.



FROM python:3-slim
ARG MLFLOW_VERSION=1.17.0

RUN pip install --upgrade pip
WORKDIR /mlflow/
RUN pip install --no-cache-dir mlflow==$MLFLOW_VERSION psycopg2-binary scikit-learn boto3
EXPOSE 5000

ENV BACKEND_URI sqlite:////mlflow/mlflow.db
ENV ARTIFACT_ROOT /mlflow/artifacts

CMD mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --host 0.0.0.0 --port 5000