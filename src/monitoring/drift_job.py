import pandas as pd
from pathlib import Path
from evidently import Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset

RAW = Path("data/serving_logs.parquet")
OUT = Path("artifacts/monitoring")
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"request_id", "timestamp", "items", "strategy"}  

def load_data():
    df = pd.read_parquet(RAW)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    cutoff = df["timestamp"].max() - pd.Timedelta(minutes=10)
    ref = df[df["timestamp"] < cutoff].copy()
    cur = df[df["timestamp"] >= cutoff].copy()

    if ref.empty or cur.empty:
        raise ValueError("Reference or current slice is empty. Adjust your 7-day window or ensure logs span >7 days.")

    keep = []
    for c in df.columns:
        if c in EXCLUDE:
            continue
        if ref[c].apply(lambda x: isinstance(x, (list, dict))).any() or cur[c].apply(lambda x: isinstance(x, (list, dict))).any():
            continue
        if ref[c].notna().any() and cur[c].notna().any():
            keep.append(c)

    ref = ref[keep]
    cur = cur[keep]
    return ref, cur

def main():
    ref, cur = load_data()
    per_column = "cache_hit_ratio" if "cache_hit_ratio" in ref.columns else "latency_ms"

    rep = Report(metrics=[
        DataDriftPreset(),
        ValueDrift(column=per_column),
    ])
    snap = rep.run(reference_data=ref, current_data=cur)

    html_path = OUT / "drift_report.html"
    snap.save_html(str(html_path))
    print(f"Saved {html_path}")

if __name__ == "__main__":
    main()