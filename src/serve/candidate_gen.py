import numpy as np, joblib, pathlib as p
import pandas as pd

art = p.Path("artifacts")
als_model = joblib.load(art / "models/als/model.joblib")         
id_maps   = joblib.load(art / "id_maps.joblib")                  
iid_map   = id_maps["iid_map"]                                   

embs   = np.load(art / "features/item_embeddings.npy")           
items  = pd.read_csv("data/processed/items.csv")                 
itemid_to_row = dict(zip(items["item_id"].tolist(), range(len(items))))

n_items_cf = als_model.item_factors.shape[0]
dim = embs.shape[1]
embs_aligned = np.zeros((n_items_cf, dim), dtype=embs.dtype)

for item_id, idx in iid_map.items():          
    row = itemid_to_row.get(item_id)
    if row is not None:
        embs_aligned[idx] = embs[row]

item_factors = als_model.item_factors
item_factors = item_factors / (np.linalg.norm(item_factors, axis=1, keepdims=True) + 1e-12)

hybrid = np.concatenate([item_factors, embs_aligned], axis=1)
np.save(art / "features/hybrid_item_vecs.npy", hybrid)
print("Hybrid item embeddings shape:", hybrid.shape)
