import os, json, pathlib as p
import numpy as np, joblib, mlflow
from tqdm import tqdm
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "./.mlflow"))
mlflow.set_experiment("recs-mlops-hybrid")

def recall_at_k(true_items, recs, k):
    hits = sum(1 for it in true_items if it in recs[:k])
    return hits / max(1, len(true_items))

def ndcg_at_k(true_items, recs, k):
    dcg = 0.0
    for i, it in enumerate(recs[:k], start=1):
        if it in true_items:
            dcg += 1.0 / np.log2(i + 1)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(true_items)) + 1))
    return dcg / (ideal or 1.0)

def main():
    art = p.Path("artifacts")
    art.mkdir(parents=True, exist_ok=True)

    R_train = load_npz(art / "R_train.npz").tocsr()
    R_val   = load_npz(art / "R_val.npz").tocsr()

    model   = joblib.load(art / "models/als/model.joblib")              
    hybrid  = np.load(art / "features/hybrid_item_vecs.npy")            

    cf_dim = model.factors
    assert hybrid.shape[0] == model.item_factors.shape[0], "Hybrid/item_factors item count mismatch"

    K = 10
    blend_cf, blend_cont = 0.7, 0.3

    with mlflow.start_run(run_name="hybrid-cosine"):
        mlflow.log_param("blend", f"{blend_cf}_cf_{blend_cont}_content")
        mlflow.log_param("K", K)

        recalls, ndcgs = [], []
        item_cf = model.item_factors                                         # [n_items, cf_dim]
        for u in tqdm(range(R_val.shape[0])):
            true = set(R_val[u].indices)
            if not true:
                continue

            cf_vec   = model.user_factors[u]                                 # [cf_dim]
            cf_score = cf_vec @ item_cf.T                                    # [n_items]
            cont_score = cosine_similarity(cf_vec.reshape(1, -1), hybrid[:, :cf_dim])[0]  # [n_items]

            scores = blend_cf * cf_score + blend_cont * cont_score
            # filter seen items
            seen = set(R_train[u].indices)
            order = np.argsort(-scores)
            recs = [i for i in order if i not in seen][:K]

            recalls.append(recall_at_k(true, recs, K))
            ndcgs.append(ndcg_at_k(true, recs, K))

        m_recall = float(np.mean(recalls) if recalls else 0.0)
        m_ndcg   = float(np.mean(ndcgs) if ndcgs else 0.0)

        mlflow.log_metric("val_recall_10", m_recall)
        mlflow.log_metric("val_ndcg_10", m_ndcg)

        metrics_path = art / "metrics_val_hybrid.json"
        with open(metrics_path, "w") as f:
            json.dump({"recall_10": m_recall, "ndcg_10": m_ndcg}, f, indent=2)

        mlflow.log_artifact(str(metrics_path))
        print(f"Hybrid Recall_10: {m_recall:.4f} | NDCG_10: {m_ndcg:.4f}")

if __name__ == "__main__":
    main()
