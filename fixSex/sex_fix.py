import pandas as pd
import numpy as np
from pathlib import Path

def pick_col(df, candidates):
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols:
            return cols[key]
    for c in df.columns:
        if any(key in c.strip().lower() for key in candidates):
            return c
    return None

def norm_sex_any(x):
    s = ("" if pd.isna(x) else str(x)).strip().lower()
    if s in ("m","male","boy","0"): return "Male"
    if s in ("f","female","girl","1"): return "Female"
    if s in ("2","other"):            return pd.NA
    return pd.NA

survey_path  = Path("BF_survey.csv")
targets_path = Path("sex_targets_by_class.csv")
roster_path  = Path("roster.tsv")

if not survey_path.exists():
    raise FileNotFoundError(f"Could not find {survey_path.resolve()}")

survey = pd.read_csv(survey_path, dtype=str)

sex_col_candidates   = ["sex", "gender", "sexe"]
class_col_candidates = ["class", "classe", "class.", "classe.", "grade"]

sex_col   = pick_col(survey, sex_col_candidates)
class_col = pick_col(survey, class_col_candidates)

print("Columns found in survey:", list(survey.columns))
print("Detected columns -> Sex:", sex_col, "| Class:", class_col)

if sex_col is None:
    survey["Sex"] = pd.NA
    sex_col = "Sex"
else:
    if sex_col != "Sex":
        survey = survey.rename(columns={sex_col: "Sex"})
        sex_col = "Sex"

if class_col is None:
    raise KeyError("No 'Class/Classe' column found. Please add a class column or tell me its exact name.")

survey["Sex"] = survey["Sex"].apply(norm_sex_any)

survey[class_col] = survey[class_col].astype(str).str.strip()

if targets_path.exists():
    targets = pd.read_csv(targets_path)
    t_class_col = pick_col(targets, ["class","classe"])
    if t_class_col is None:
        raise KeyError("sex_targets_by_class.csv must have a 'Class'/'Classe' column.")
    if t_class_col != "Class":
        targets = targets.rename(columns={t_class_col: "Class"})
else:
    if not roster_path.exists():
        raise FileNotFoundError("Neither sex_targets_by_class.csv nor Rooster.tsv found.")
    roster = pd.read_csv(
        roster_path, sep="\t", header=None,
        names=["SchoolID","Name","DOB","Sex","Status","Class"],
        dtype=str
    )
    def norm_sex_roster(x):
        s = ("" if pd.isna(x) else str(x)).strip().upper()
        if s == "M": return "Male"
        if s == "F": return "Female"
        return pd.NA
    roster["Sex"] = roster["Sex"].apply(norm_sex_roster)
    roster["Class"] = roster["Class"].astype(str).str.strip()
    targets = roster.groupby("Class")["Sex"].value_counts().unstack(fill_value=0)
    for col in ["Female","Male"]:
        if col not in targets.columns: targets[col] = 0
    targets["Total"] = targets["Female"] + targets["Male"]
    targets["Prop_Female"] = targets["Female"] / targets["Total"].where(targets["Total"]>0, 1)
    targets["Prop_Male"]   = 1 - targets["Prop_Female"]
    targets = targets.reset_index()
    targets.to_csv(targets_path, index=False)
    print(f"Built targets from roster and saved -> {targets_path}")

need_cols = ["Class","Prop_Female","Prop_Male"]
for c in need_cols:
    if c not in targets.columns:
        raise KeyError(f"Targets file missing column: {c}")
targets = targets[need_cols]

survey = survey.merge(targets, left_on=class_col, right_on="Class", how="left")

rng = np.random.default_rng(42)

def impute_class(group):
    n = len(group)
    pF = group["Prop_Female"].iloc[0]
    if pd.isna(pF):
        return group
    target_f = int(round(pF * n))
    cur_f = (group["Sex"] == "Female").sum()
    cur_m = (group["Sex"] == "Male").sum()
    need_f = max(0, target_f - cur_f)

    candidates = group.index[group["Sex"].isna()].tolist()
    rng.shuffle(candidates)

    assign_f = candidates[:need_f]
    group.loc[assign_f, "Sex"] = "Female"
    group.loc[group["Sex"].isna(), "Sex"] = "Male"
    return group

survey = survey.groupby(class_col, group_keys=False).apply(impute_class)

survey = survey.drop(columns=[c for c in ["Prop_Female","Prop_Male","Class"] if c in survey.columns])

out_path = Path("BF_survey_sex_fixed.csv")
survey.to_csv(out_path, index=False)
print(f"✅ Done! Wrote {out_path}")
print(survey["Sex"].value_counts(dropna=False))
