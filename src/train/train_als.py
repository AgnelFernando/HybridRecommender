import os, json, time
import mlflow
import numpy as np
from scipy.sparse import load_npz
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
import joblib, pathlib as p

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "./.mlflow")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("recs-mlops-baselines")
mlflow.log_artifact("mlflow_env.yml")

def recall_at_k(true_items, recs, k):
    hits = sum(1 for it in true_items if it in recs[:k])
    return hits / max(1, len(true_items))

def ndcg_at_k(true_items, recs, k):
    dcg=0.0
    for i,item in enumerate(recs[:k], start=1):
        if item in true_items:
            dcg += 1/np.log2(i+1)
    # ideal DCG
    ideal = sum(1/np.log2(i+1) for i in range(1, min(k, len(true_items))+1))
    return dcg / (ideal or 1.0)

def topn(model, user_id, user_items, N=100):
    # implicit expects item-user CSR for recommendations
    recs = model.recommend(user_id, user_items, N=N, filter_already_liked_items=True)
    return recs[0]

def build_true_items(R):
    # for evaluation we need per-user held-out items (nonzero cols)
    true = {}
    R = R.tocsr()
    for u in range(R.shape[0]):
        start, end = R.indptr[u], R.indptr[u+1]
        true[u] = set(R.indices[start:end].tolist())
    return true


def main():
    R_train = load_npz("artifacts/R_train.npz")
    R_val   = load_npz("artifacts/R_val.npz")
    Cui = R_train.tocsr()

    params = dict(factors=64, regularization=0.1, iterations=20)
    mlflow.end_run()
    with mlflow.start_run(run_name="ALS-implicit", ):
        mlflow.log_params(params)

        model = AlternatingLeastSquares(**params)
        t0=time.time() 
        model.fit(Cui)
        mlflow.log_metric("train_secs", time.time()-t0)

        true_val = build_true_items(R_val)
        user_items = R_train.tocsr()

        K = 10
        recalls, ndcgs = [], []
        for u in tqdm(range(R_val.shape[0])):
            if len(true_val[u])==0: 
                continue
            recs = topn(model, u, user_items[u], N=K)
            recalls.append(recall_at_k(true_val[u], recs, K))
            ndcgs.append(ndcg_at_k(true_val[u], recs, K))

        mlflow.log_metric("val_recall_10", float(np.mean(recalls)))
        mlflow.log_metric("val_ndcg_10", float(np.mean(ndcgs)))

        out = p.Path("artifacts/models/als")
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out/"model.joblib")
        mlflow.log_artifact(str(out/"model.joblib"))
        with open("artifacts/metrics_val.json","w") as f:
            json.dump({"recall_10":float(np.mean(recalls)),
                       "ndcg_10":float(np.mean(ndcgs))}, f, indent=2)
        mlflow.log_artifact("artifacts/metrics_val.json")

if __name__ == "__main__":
    main()
