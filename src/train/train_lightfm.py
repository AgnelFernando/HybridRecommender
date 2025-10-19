from lightfm import LightFM
from scipy.sparse import load_npz
import numpy as np, mlflow, os
from tqdm import tqdm

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",".mlflow"))
mlflow.set_experiment("recs-mlops-baselines")

def evaluate(model, R_train, R_val, K=10):
    n_users, n_items = R_val.shape
    R_val = R_val.tocsr(); R_train = R_train.tocsr()
    rec, ndc = [], []
    for u in tqdm(range(n_users)):
        true = set(R_val[u].indices)
        if not true: continue
        scores = model.predict(u, np.arange(n_items))
        # filter seen items
        seen = set(R_train[u].indices)
        order = np.argsort(-scores)
        recs = [i for i in order if i not in seen][:K]
        # metrics
        hit = sum(1 for i in true if i in recs)/max(1,len(true))
        dcg = sum((1/np.log2(i+2)) for i,item in enumerate(recs) if item in true)
        ideal = sum(1/np.log2(i+2) for i in range(min(K, len(true))))
        rec.append(hit); ndc.append(dcg/(ideal or 1))
    return float(np.mean(rec)), float(np.mean(ndc))

if __name__ == "__main__":
    R_train = load_npz("artifacts/R_train.npz")
    R_val   = load_npz("artifacts/R_val.npz")
    with mlflow.start_run(run_name="LightFM-warp"):
        model = LightFM(loss="warp", no_components=64)
        model.fit(R_train, epochs=20, num_threads=4)
        r, n = evaluate(model, R_train, R_val, K=10)
        mlflow.log_metric("val_recall_10", r)
        mlflow.log_metric("val_ndcg_10", n)