import pandas as pd

roster = pd.read_csv(
    "roster.tsv",
    sep="\t",
    header=None,
    names=["Sex","Class"],
    dtype=str
)

def norm_sex_roster(x):
    s = ("" if pd.isna(x) else str(x)).strip().upper()
    if s == "M": return "Male"
    if s == "F": return "Female"
    return pd.NA

roster["Sex"] = roster["Sex"].apply(norm_sex_roster)
roster = roster.dropna(subset=["Class","Sex"])

targets = roster.groupby("Class")["Sex"].value_counts().unstack(fill_value=0)
for col in ["Female","Male"]:
    if col not in targets.columns: targets[col] = 0
targets["Total"] = targets["Female"] + targets["Male"]
targets["Prop_Female"] = targets["Female"] / targets["Total"].where(targets["Total"]>0, 1)
targets["Prop_Male"]   = 1 - targets["Prop_Female"]
targets = targets.reset_index()

targets.to_csv("sex_targets_by_class.csv", index=False)
print(targets.head())
