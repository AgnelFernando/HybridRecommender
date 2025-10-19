import pandas as pd
from scipy.sparse import coo_matrix, save_npz
import pathlib as p
import joblib

interim = p.Path("data/interim")
artifacts = p.Path("artifacts") 
artifacts.mkdir(exist_ok=True)

def encode_ids(df):
    uids = pd.Index(df.user_id.unique()).sort_values()
    iids = pd.Index(df.item_id.unique()).sort_values()
    uid_map = {u:i for i,u in enumerate(uids)}
    iid_map = {i:j for j,i in enumerate(iids)}
    df["uid"] = df.user_id.map(uid_map)
    df["iid"] = df.item_id.map(iid_map)
    return df, uid_map, iid_map

train = pd.read_csv(interim/"train.csv")
train, uid_map, iid_map = encode_ids(train)

def apply_map(df, uid_map, iid_map):
    df = df[df.user_id.isin(uid_map) & df.item_id.isin(iid_map)].copy()
    df["uid"] = df.user_id.map(uid_map)
    df["iid"] = df.item_id.map(iid_map)
    return df

val = apply_map(pd.read_csv(interim/"val.csv"), uid_map, iid_map)
test = apply_map(pd.read_csv(interim/"test.csv"), uid_map, iid_map)

n_users = len(uid_map)
n_items = len(iid_map)
def to_csr(df):
    return coo_matrix((df.rating.values, (df.uid.values, df.iid.values)),
                      shape=(n_users, n_items)).tocsr()

R_train = to_csr(train)
R_val = to_csr(val)
R_test = to_csr(test)

save_npz(artifacts/"R_train.npz", R_train)
save_npz(artifacts/"R_val.npz", R_val)
save_npz(artifacts/"R_test.npz", R_test)

joblib.dump({"uid_map":uid_map, "iid_map":iid_map}, artifacts/"id_maps.joblib")
print("Saved matrices and id maps to artifacts/")
