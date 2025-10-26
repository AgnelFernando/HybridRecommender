import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    rds = redis.Redis.from_url(REDIS_URL) if os.getenv("ENABLE_REDIS", "0") == "1" else None
except Exception:
    rds = None

try:
    import faiss
    USE_FAISS = os.getenv("USE_FAISS", "0") == "1"
except Exception:
    USE_FAISS = False

ART = Path("artifacts")

ID_MAPS = joblib.load(ART / "id_maps.joblib")                         
ALS = joblib.load(ART / "models/als/model.joblib")                    
ITEM_EMB = np.load(ART / "features/hybrid_item_vecs.npy")             
ITEMS_DF = pd.read_csv("data/processed/items.csv")[["item_id","title","genres"]]

uid_map = ID_MAPS["uid_map"]            
iid_map = ID_MAPS["iid_map"]             
rev_iid = {v:k for k,v in iid_map.items()}  

from scipy.sparse import load_npz
R_TRAIN = load_npz(ART / "R_train.npz").tocsr()
item_pop = np.asarray(R_TRAIN.sum(axis=0)).ravel()
popular_iids = np.argsort(-item_pop)

if USE_FAISS:
    dim = ITEM_EMB.shape[1]
    index = faiss.IndexFlatIP(dim)
    norms = np.linalg.norm(ITEM_EMB, axis=1, keepdims=True) + 1e-12
    index.add((ITEM_EMB / norms).astype("float32"))

def _seen_items(uid: int):
    return set(R_TRAIN[uid].indices) if uid < R_TRAIN.shape[0] else set()

def _to_catalog(iids: List[int], topk: int):
    rows = []
    for iid in iids[:topk]:
        item_id = rev_iid.get(iid)
        if item_id is None: 
            continue
        row = ITEMS_DF[ITEMS_DF.item_id == item_id]
        if row.empty:
            rows.append({"item_id": int(item_id), "title": None, "genres": None})
        else:
            rr = row.iloc[0]
            rows.append({"item_id": int(rr.item_id), "title": rr.title, "genres": rr.genres})
    return rows

def _cf_scores(uid: int):
    uvec = ALS.user_factors[uid]
    return uvec @ ALS.item_factors.T

def _content_scores(uid: int):
    cf_dim = ALS.factors
    uvec = ALS.user_factors[uid].reshape(1, -1)
    A = ITEM_EMB[:, :cf_dim]
    num = A @ uvec.T                       
    den = (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12) * (np.linalg.norm(uvec) + 1e-12)
    return (num / den).ravel()

def recommend_internal(uid: int, topk: int, blend_cf=0.7, blend_cont=0.3):
    cache_key = f"recs:{uid}:{topk}:{blend_cf:.2f}:{blend_cont:.2f}"
    if rds:
        cached = rds.get(cache_key)
        if cached:
            return json.loads(cached)

    seen = _seen_items(uid)
    cf = _cf_scores(uid)
    cont = _content_scores(uid)
    scores = blend_cf * cf + blend_cont * cont

    order = np.argsort(-scores)
    recs = [i for i in order if i not in seen][:topk]
    payload = _to_catalog(recs, topk)

    if rds:
        rds.setex(cache_key, int(os.getenv("REDIS_TTL_SECONDS", "600")), json.dumps(payload))
    return payload

def cold_start(topk: int):
    return _to_catalog(list(popular_iids), topk)

app = FastAPI(title="Hybrid Recommender as a Service", version="0.1.0")

class RecRequest(BaseModel):
    user_id: int = Field(..., description="Raw user id from dataset")
    topk: int = Field(10, ge=1, le=100)
    blend_cf: float = Field(0.7, ge=0.0, le=1.0)
    blend_cont: float = Field(0.3, ge=0.0, le=1.0)

class FeedbackEvent(BaseModel):
    user_id: int
    item_id: int
    action: str = Field(..., description="click|view|like|purchase")
    weight: Optional[float] = 1.0

@app.get("/health")
def health():
    return {"status": "ok", "items": int(ITEM_EMB.shape[0])}

@app.post("/recommendations")
def recommendations(req: RecRequest):
    if req.user_id not in uid_map:
        return {"user_id": req.user_id, "cold_start": True, "items": cold_start(req.topk)}

    uid = uid_map[req.user_id]
    if uid >= ALS.user_factors.shape[0]:
        return {"user_id": req.user_id, "cold_start": True, "items": cold_start(req.topk)}

    items = recommend_internal(uid, req.topk, req.blend_cf, req.blend_cont)
    return {"user_id": req.user_id, "cold_start": False, "items": items}

@app.post("/feedback")
def feedback(evt: FeedbackEvent):
    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
    with open(log_dir / "feedback.log", "a") as f:
        f.write(json.dumps(evt.model_dump()) + "\n")
    if rds:
        rds.lpush("feedback_stream", json.dumps(evt.model_dump()))
    return {"ok": True}

@app.get("/similar/{item_id}")
def similar_items(item_id: int, topk: int = 10):
    if not USE_FAISS:
        raise HTTPException(status_code=400, detail="FAISS disabled")
    if item_id not in iid_map:
        raise HTTPException(status_code=404, detail="Unknown item_id")
    iid = iid_map[item_id]
    vec = ITEM_EMB[iid:iid+1].astype("float32")
    norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
    vec = vec / norms
    D, I = index.search(vec, topk + 1)  
    iids = [int(x) for x in I[0] if int(x) != iid][:topk]
    return {"item_id": item_id, "similar": _to_catalog(iids, topk)}

