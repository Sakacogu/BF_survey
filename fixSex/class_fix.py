import pandas as pd
import numpy as np

def pick_col(df, candidates):
    lowmap = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lowmap:
            return lowmap[key]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            return c
    return None

def norm_class(s):
    return ("" if pd.isna(s) else str(s)).strip().lower().replace(" ", "")

def norm_sex(x):
    s = ("" if pd.isna(x) else str(x)).strip().lower()
    if s in ("m","male","boy","0"): return "Male"
    if s in ("f","female","girl","1"): return "Female"
    if s in ("2","other",""): return pd.NA
    return pd.NA

survey = pd.read_csv("BF_survey.csv", dtype=str)
targets = pd.read_csv("sex_targets_by_class.csv")

survey_class_col = pick_col(survey, ["class","classe","grade","section"])
targets_class_col = pick_col(targets, ["class","classe","grade","section"])

if survey_class_col is None:
    raise KeyError("Could not find a class/grade column in the survey.")
if targets_class_col is None:
    raise KeyError("Could not find a class/grade column in the targets file.")

print("Detected survey class col:", survey_class_col)
print("Detected targets class col:", targets_class_col)

survey["Class_norm"]  = survey[survey_class_col].astype(str).map(norm_class)
targets["Class_norm"] = targets[targets_class_col].astype(str).map(norm_class)

if not {"Prop_Female","Prop_Male"}.issubset(set(targets.columns)):
    if {"Female","Male"}.issubset(set(targets.columns)):
        tot = targets["Female"].fillna(0).astype(float) + targets["Male"].fillna(0).astype(float)
        targets["Prop_Female"] = targets["Female"].fillna(0).astype(float) / tot.where(tot>0, 1)
        targets["Prop_Male"]   = 1 - targets["Prop_Female"]
    else:
        raise KeyError("Targets must have Prop_Female/Prop_Male or Female/Male columns.")

survey = survey.merge(
    targets[["Class_norm","Prop_Female","Prop_Male"]],
    on="Class_norm", how="left"
)

survey["Sex"] = survey.get("Sex", pd.Series([pd.NA]*len(survey))).map(norm_sex)
rng = np.random.default_rng(42)

if {"Female","Male"}.issubset(set(targets.columns)):
    total_F = targets["Female"].fillna(0).astype(float).sum()
    total_M = targets["Male"].fillna(0).astype(float).sum()
    pF_global = total_F / (total_F + total_M) if (total_F + total_M) > 0 else 0.5
else:
    pF_global = targets["Prop_Female"].dropna().mean() if "Prop_Female" in targets.columns else 0.5

survey["Prop_Female"] = survey["Prop_Female"].fillna(pF_global)
survey["Prop_Male"]   = 1 - survey["Prop_Female"]

def impute_group(g):
    miss = g.index[g["Sex"].isna()].tolist()
    if not miss:
        return g
    n = len(g)
    pF = float(g["Prop_Female"].iloc[0])
    target_F = int(round(pF * n))
    cur_F = (g["Sex"] == "Female").sum()
    need_F = max(0, min(len(miss), target_F - cur_F))
    rng.shuffle(miss)
    assign_F = miss[:need_F]
    assign_M = miss[need_F:]
    g.loc[assign_F, "Sex"] = "Female"
    g.loc[assign_M, "Sex"] = "Male"
    return g

survey = survey.groupby("Class_norm", group_keys=False).apply(impute_group)

out = survey.drop(columns=["Class_norm","Prop_Female","Prop_Male"])
out.to_csv("BF_survey_sex_fixed.csv", index=False)

print(out["Sex"].value_counts(dropna=False))
print("Missing values in Sex:", out["Sex"].isna().sum())
