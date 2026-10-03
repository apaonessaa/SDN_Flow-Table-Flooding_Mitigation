import pandas as pd

df = pd.read_csv("./dataset_mfr_015.csv")

df = df.sort_values(
    by=["scenario_id", "label", "round"],
    ascending=[True, True, True],
    kind="stable",
).reset_index(drop=True)

df.to_csv("./dataset_mfr_015_sorted.csv", index=False)