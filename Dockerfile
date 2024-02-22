%%writefile Dockerfile
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim
RUN pip install --no-cache-dir transformers==4.1.1 tensorflow==2.9.1 numpy==1.23.1 pydantic==1.9.1 torch==2.2.0

COPY model_artifact/ ./model_artifact

COPY main.py ./main.py
ENV FROM_PT=False
RUN chmod -R 777 model/
RUN ls -l

RUN ls -l model/
