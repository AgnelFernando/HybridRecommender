import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pathlib as p
import joblib

model_name = "all-MiniLM-L6-v2"
out_dir = p.Path("artifacts/features"); out_dir.mkdir(parents=True, exist_ok=True)

items = pd.read_csv("data/processed/items.csv")
text = (items["title"] + " " + items["genres"].fillna("")).tolist()

print(f"Encoding {len(text)} movie texts...")
encoder = SentenceTransformer(model_name)
embs = encoder.encode(text, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
np.save(out_dir/"item_embeddings.npy", embs)

meta = {"model": model_name, "dim": embs.shape[1]}
joblib.dump(meta, out_dir/"meta.joblib")
print("Saved embeddings to", out_dir)