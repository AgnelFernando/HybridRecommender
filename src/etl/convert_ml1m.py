import pandas as pd, pathlib as p

src_dir = p.Path("data/raw/ml-1m")
out_dir = p.Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

r = pd.read_csv(src_dir/"ratings.dat", sep="::", engine="python",
                encoding="latin-1", names=["user_id","item_id","rating","ts"])
m = pd.read_csv(src_dir/"movies.dat", sep="::", engine="python",
                encoding="latin-1", names=["item_id","title","genres"])

r.to_csv(out_dir/"ratings.csv", index=False)
m.to_csv(out_dir/"items.csv", index=False)
print("Wrote", out_dir/"ratings.csv", out_dir/"items.csv")