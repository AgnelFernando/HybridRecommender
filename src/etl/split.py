import pandas as pd, numpy as np, pathlib as p
out_dir = p.Path("data/interim")
out_dir.mkdir(parents=True, exist_ok=True)

r = pd.read_csv("data/processed/ratings.csv")
r = r.sort_values(["user_id","ts"])

def chrono_split(g):
    n = len(g)
    tr = int(0.8*n)
    va = int(0.9*n)
    g["split"] = "train"
    g.iloc[tr:va, g.columns.get_loc("split")] = "val"
    g.iloc[va:, g.columns.get_loc("split")] = "test"
    return g

r = r.groupby("user_id", group_keys=False).apply(chrono_split)

r[r.split=="train"].to_csv(out_dir/"train.csv", index=False)
r[r.split=="val"].to_csv(out_dir/"val.csv", index=False)
r[r.split=="test"].to_csv(out_dir/"test.csv", index=False)
print("Wrote splits to", out_dir)