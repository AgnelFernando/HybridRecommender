import json, pathlib as p, os
from subprocess import check_call
from prefect import flow, task

ART = p.Path("artifacts")
PROMOTION_DELTA = float(os.getenv("PROMOTION_DELTA", "0.003"))  # ndcg@10 threshold

@task
def dvc_repro():
    check_call(["dvc", "repro", "-f"])

@task
def read_metrics():
    m = json.loads((ART / "metrics_val_hybrid.json").read_text())
    return m

@task
def improved_over_champion(m: dict):
    champ_path = ART / "champion_metrics.json"
    if champ_path.exists():
        champ = json.loads(champ_path.read_text())
        return (m.get("ndcg@10", 0.0) - champ.get("ndcg@10", 0.0)) >= PROMOTION_DELTA
    return True  

@task
def promote_model():
    (ART / "champion_metrics.json").write_text((ART / "metrics_val_hybrid.json").read_text())
    (ART / "champion_pointer.txt").write_text("hybrid-latest")  

@flow(name="weekly-dvc-retrain")
def retrain_flow():
    dvc_repro()
    m = read_metrics()
    if improved_over_champion(m):
        promote_model()

if __name__ == "__main__":
    retrain_flow()
