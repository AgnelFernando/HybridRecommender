.PHONY: run api docker build push test

run:
\tuvicorn src.serve.api:app --reload --port 8000

api:
\tcurl -s localhost:8000/health | jq .

test:
\tpytest -q

docker:
\tdocker build -t recs-mlops:latest -f infra/docker/Dockerfile .

push:  
\t@echo "Handled in CI"