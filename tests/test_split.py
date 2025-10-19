import pandas as pd

def test_splits_exist():
    for name in ["train","val","test"]:
        df = pd.read_csv(f"data/interim/{name}.csv")
        assert {"user_id","item_id","rating","ts"}.issubset(df.columns)