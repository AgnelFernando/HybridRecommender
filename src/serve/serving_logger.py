import json, uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import pandas as pd

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
JSONL_PATH = LOG_DIR / "serving_logs.jsonl"
PARQUET_PATH = Path("data/serving_logs.parquet")
PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)

def _flatten_items(items: List[Dict[str, Any]]) -> List[int]:
    # Your response items contain {item_id, title, genres}; keep ids for drift/metrics
    out: List[int] = []
    for it in items:
        iid = it.get("item_id")
        if isinstance(iid, (int, float)) and iid is not None:
            out.append(int(iid))
    return out

def log_recommendation_event(
    *,
    user_id: int,
    topk: int,
    strategy: str,             
    cold_start: bool,
    items: List[Dict[str, Any]],
    latency_ms: float,
    cache_hit_ratio: Optional[float],
    blend_cf: float,
    blend_cont: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    now = datetime.now(timezone.utc)
    row = {
        "request_id": str(uuid.uuid4()),
        "timestamp": now.isoformat(),
        "user_id": int(user_id),
        "topk": int(topk),
        "strategy": strategy,
        "cold_start": bool(cold_start),
        "items": _flatten_items(items),     
        "latency_ms": float(latency_ms),
        "cache_hit_ratio": None if cache_hit_ratio is None else float(cache_hit_ratio),
        "blend_cf": float(blend_cf),
        "blend_cont": float(blend_cont),
    }
    if extra:
        row.update(extra)

    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(_json_dumps(row) + "\n")

def compact_to_parquet(limit_rows: Optional[int] = None) -> str:
    """Convert/overwrite data/serving_logs.parquet from the JSONL file."""
    if not JSONL_PATH.exists():
        return f"Nothing to compact; {JSONL_PATH} does not exist."

    rows: List[Dict[str, Any]] = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit_rows and i >= limit_rows:
                break

    if not rows:
        return "No rows found in JSONL."

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    df.to_parquet(PARQUET_PATH, index=False)
    return f"Wrote {len(df)} rows to {PARQUET_PATH}"