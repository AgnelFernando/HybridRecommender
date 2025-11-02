.PHONY: run api docker build push test

run:
	uvicorn src.serve.api:app --reload --port 8000

api:
	curl -s localhost:8000/health | jq .

test:
	pytest -q

docker:
	docker build -t recs-mlops:latest -f infra/docker/Dockerfile .

monitor-drift:
	python -m src.monitoring.drift_job

compact-logs:
	python -c "from src.serve.serving_logger import compact_to_parquet; print(compact_to_parquet())"