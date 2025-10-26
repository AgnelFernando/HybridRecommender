FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN pip install --no-cache-dir fastapi==0.115.0 "uvicorn[standard]==0.30.6" redis==5.0.8 faiss-cpu==1.8.0.post1 pydantic==2.9.2

COPY src /app/src
COPY artifacts /app/artifacts
COPY data/processed/items.csv /app/data/processed/items.csv

ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8080"]