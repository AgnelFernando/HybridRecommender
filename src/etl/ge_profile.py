import pandas as pd, great_expectations as ge, pathlib as p

ratings = pd.read_csv("data/processed/ratings.csv")
df = ge.from_pandas(ratings)
# basic expectations
df.expect_table_columns_to_match_set(["user_id","item_id","rating","ts"])
df.expect_column_values_to_not_be_null("user_id")
df.expect_column_values_to_not_be_null("item_id")
df.expect_column_values_to_be_between("rating", 1, 5)
res = df.validate()
print(res["statistics"])
assert res["success"], "GE validations failed!"
