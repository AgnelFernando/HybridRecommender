FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY artifacts/models/als/model.joblib /app/artifacts/models/als/model.joblib
COPY artifacts/id_maps.joblib /app/artifacts/id_maps.joblib
COPY artifacts/R_train.npz /app/artifacts/R_train.npz
COPY artifacts/features/hybrid_item_vecs.npy /app/artifacts/features/hybrid_item_vecs.npy
COPY data/processed/items.csv /app/data/processed/items.csv


ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8080"]