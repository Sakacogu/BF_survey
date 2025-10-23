import pandas as pd
import numpy as np

# === CONFIG ===
INPUT = "BF_correct_survey_with_IDs_EN.xlsx"   # or your CSV file
OUTPUT = "bf_clean.csv"

# Column groups
A_cols = [f"A{i}" for i in range(1,16)]
H_cols = [f"H{i}" for i in range(1,13)]
T_cols = [f"T{i}" for i in range(1,13)]

# Background columns (adjust names to match your file)
bg_yesno = [
    "Do you go to school every day? (Yes/No)",
    "Do you share your phone with your family? (Yes/No)",
    "Do you have electricity at home? (Yes/No)",
    "Do your parents/guardians set rules about your phone? (Yes/No)",
]
sex_col = "Sex (Boy/Girl/Other)"
age_col = "Age (years)"
class_col = "Class"
purpose_col = "Main purpose of phone use (School/Communication/Games/Social Media/Music/Other)"
years_col = "T13: How long have you had a smartphone? (Years)"
months_col = "T14: How long have you had a smartphone? (Months)"

# Load (Numbers can export CSV; xlsx also works)
df = pd.read_excel(INPUT) if INPUT.lower().endswith("xlsx") else pd.read_csv(INPUT)

# Trim spaces in headers and values
df.columns = [c.strip() for c in df.columns]
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.strip()

# Ensure ID is string with leading zeros if looks numeric
if "ID" in df.columns:
    df["ID"] = df["ID"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(3)

# Map Yes/No -> 1/0 (case-insensitive)
yn_map = {"yes":1,"no":0,"oui":1,"non":0,"1":1,"0":0}
for c in bg_yesno:
    if c in df.columns:
        df[c+"_raw"] = df[c]
        df[c] = df[c].str.lower().map(yn_map)

# Coerce numeric scales
def coerce_int(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")

for c in A_cols + H_cols:
    if c in df.columns:
        df[c] = coerce_int(df[c])

for c in T_cols + [years_col, months_col, age_col]:
    if c in df.columns:
        df[c] = coerce_int(df[c])

# Clamp to allowed ranges and flag fixes
def clamp_and_flag(df, cols, low, high, prefix):
    for c in cols:
        if c not in df.columns: 
            continue
        bad = df[c].notna() & ((df[c] < low) | (df[c] > high))
        df[f"{prefix}{c}_was_bad"] = bad.astype(int)
        df.loc[df[c] < low, c] = low
        df.loc[df[c] > high, c] = high

clamp_and_flag(df, A_cols, 1, 4, "fix_")
clamp_and_flag(df, H_cols, 1, 4, "fix_")
clamp_and_flag(df, T_cols, 1, 7, "fix_")

# Handle age and duration sanity
if age_col in df.columns:
    df["fix_age_was_bad"] = ((df[age_col].notna()) & ((df[age_col] < 8) | (df[age_col] > 25))).astype(int)
    df.loc[df[age_col] < 8, age_col] = 8
    df.loc[df[age_col] > 25, age_col] = 25

# Months owned
if years_col in df.columns and months_col in df.columns:
    df["Months_owned"] = (df[years_col].fillna(0).astype("Int64") * 12 + df[months_col].fillna(0).astype("Int64")).astype("Int64")

# Reverse-scoring helpers
def reverse_1_to_4(x):
    return x.map({1:4, 2:3, 3:2, 4:1}).astype("Int64")

# Happiness reverse items for a "higher = more happiness" score
H_rev = {"H2","H4","H6","H9","H11"}  # depressed, never safe, prejudices, lonely, anxious
for h in H_rev:
    if h in df.columns:
        df[h+"_rev"] = reverse_1_to_4(df[h])

# Build Happiness_all_scored (mix of normal + reversed)
H_scored = []
for h in H_cols:
    if h in H_rev and (h+"_rev") in df.columns:
        H_scored.append(h+"_rev")
    else:
        H_scored.append(h)
df["Happiness_mean"] = df[H_scored].astype("float").mean(axis=1)

# Attention difficulty (as-is) and Attention control (reversed so higher=better)
df["Attention_difficulty_mean"] = df[[c for c in A_cols if c in df.columns]].astype("float").mean(axis=1)

for a in A_cols:
    if a in df.columns:
        df[a+"_rev"] = reverse_1_to_4(df[a])
A_rev_cols = [a+"_rev" for a in A_cols if (a+"_rev") in df.columns]
df["Attention_control_mean"] = df[A_rev_cols].astype("float").mean(axis=1)

# Phone aggregates
if set(T_cols).issubset(df.columns):
    df["Phone_general_mean"] = df[["T1","T2","T3","T4","T5","T6"]].astype("float").mean(axis=1)
    df["Phone_automatic_mean"] = df[["T7","T8","T9","T10","T11","T12"]].astype("float").mean(axis=1)
    df["Phone_overall_mean"] = df[T_cols].astype("float").mean(axis=1)

# Simple quality indicators
df["missing_attention_fraction"] = df[A_cols].isna().mean(axis=1)
df["missing_happiness_fraction"] = df[H_cols].isna().mean(axis=1)
df["missing_phone_fraction"] = df[T_cols].isna().mean(axis=1)

# Drop helper raw columns if you want a cleaner file (optional)
# df = df.drop(columns=[c for c in df.columns if c.endswith("_raw") or c.endswith("_was_bad") or c.endswith("_rev")], errors="ignore")

# Save
df.to_csv(OUTPUT, index=False)
print("Wrote", OUTPUT, "with", len(df), "rows")
