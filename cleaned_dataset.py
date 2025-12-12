
import os, numpy as np, pandas as pd
from scipy.stats import binomtest


BEHAV = "/Users/changyin/Downloads/LIFESPAN_agingAZ_all.csv"
YOUNG = "/Users/changyin/Downloads/YoungNeuropsych.csv"
OLD   = "/Users/changyin/Downloads/OlderNeuropsych.csv"
OUT   = "/Users/changyin/PyCharmMiscProject/outputs"
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.01        # p-threshold from the paper
MIN_LAST_TRIALS = 1 # require at least this many evaluable last trials per subject


EXTRA_YOUNG_REMOVE = []  # e.g., [105, 119]


beh = pd.read_csv(BEHAV)
yn  = pd.read_csv(YOUNG).assign(group="young")
od  = pd.read_csv(OLD).assign(group="old")
neuro = pd.concat([yn, od], ignore_index=True)

# Extract PID from subjectID
beh["pid"] = pd.to_numeric(beh["subjectID"].str.extract(r'behavior_(\d+)_')[0], errors="coerce")

for col in ["gameLength"] + [f"c{i}" for i in range(1,11)] + [f"r{i}" for i in range(1,11)]:
    if col in beh.columns:
        beh[col] = pd.to_numeric(beh[col], errors="coerce")

# Neuro numeric fields
for col in ["Participant ID","Age","MoCA","Education"]:
    if col in neuro.columns:
        neuro[col] = pd.to_numeric(neuro[col], errors="coerce")

df = beh.merge(neuro, left_on="pid", right_on="Participant ID", how="left", suffixes=("","_neuro"))

def paper_summary_line(d, grp):
    sub = d.drop_duplicates("pid").query("group == @grp")
    g = sub["Gender"].astype(str).str.upper()
    male = int((g=="M").sum()); female = int((g=="F").sum())
    age_m, age_sd = sub["Age"].mean(), sub["Age"].std()
    edu_m, edu_sd = sub["Education"].mean(), sub["Education"].std()
    return f"Male = {male}, Female = {female}, age: M = {age_m:.1f}, SD = {age_sd:.1f}, education: M = {edu_m:.1f} years, SD = {edu_sd:.1f}"

def counts_line(d, label):
    u = d.dropna(subset=["pid"]).groupby("group")["pid"].nunique()
    old_n = int(u.get("old",0)); yng_n = int(u.get("young",0))
    print(f"\n== {label} ==")
    print(f"Unique participants — old: {old_n}, young: {yng_n}")
    if old_n>0:  print("Old  → "  + paper_summary_line(d, "old"))
    if yng_n>0: print("Young → " + paper_summary_line(d, "young"))

# Exclusion 1: MoCA
df_moca = df[df["MoCA"] >= 26].copy()
counts_line(df,        "Raw (before exclusions)")
counts_line(df_moca,   "After MoCA ≥ 26")

# Exclusion 2: performance over the entire experiment
# Define correctness on the last free trial per game using SAMPLE MEANS up to t-1.
def last_free_trial_correct(row):
    gl = int(row["gameLength"]) if not pd.isna(row["gameLength"]) else -1
    if gl not in (5,10):  # only H1 (5) or H6 (10)
        return np.nan
    # last free trial index:
    t_last = 5 if gl == 5 else 10
    # choices and rewards up to t_last-1
    c = [row.get(f"c{i}") for i in range(1, t_last)]
    r = [row.get(f"r{i}") for i in range(1, t_last)]
    c_last = row.get(f"c{t_last}")

    idx1 = [i for i,ci in enumerate(c) if ci == 1]
    idx2 = [i for i,ci in enumerate(c) if ci == 2]
    if len(idx1)==0 or len(idx2)==0 or pd.isna(c_last):
        return np.nan

    mean1 = float(np.mean([r[i] for i in idx1])) if idx1 else np.nan
    mean2 = float(np.mean([r[i] for i in idx2])) if idx2 else np.nan
    if np.isnan(mean1) or np.isnan(mean2) or np.isclose(mean1, mean2):
        return np.nan  # tie or missing -> uninformative

    # correct if last choice matches higher sample mean
    if c_last == 1 and mean1 > mean2: return 1.0
    if c_last == 2 and mean2 > mean1: return 1.0
    if c_last in (1,2):               return 0.0
    return np.nan

# Compute correctness for all games (both horizons)
df_moca["last_correct"] = df_moca.apply(last_free_trial_correct, axis=1)

# Aggregate per participant across entire experiment
perf = (df_moca.groupby("pid", as_index=False)
              .agg(n_eval=("last_correct","count"),
                   k_corr=("last_correct","sum")))
perf["k_corr"] = perf["k_corr"].fillna(0).astype(int)

# Binomial test vs 0.5 (one-sided, greater), alpha=0.01
def pass_perf_row(n, k):
    if n < MIN_LAST_TRIALS:
        return False
    p = binomtest(k, n, 0.5, alternative="greater").pvalue
    return p < ALPHA

perf["pass"] = perf.apply(lambda r: pass_perf_row(int(r.n_eval), int(r.k_corr)), axis=1)

# Join back for diagnostics and filtering
pid_info = df_moca.drop_duplicates("pid")[["pid","group","Gender","Age","Education","MoCA"]]
perf_full = pid_info.merge(perf, on="pid", how="left")

pass_ids = set(perf_full.loc[perf_full["pass"]==True, "pid"].dropna().unique())
df_final = df_moca[df_moca["pid"].isin(pass_ids)].copy()



counts_line(df_final, f"After MoCA + performance binomial test (α={ALPHA}, pooled H1&H6 last trials)")

# Who got dropped by performance (after MoCA)
dropped_perf = sorted(set(df_moca["pid"].dropna().unique()) - set(df_final["pid"].dropna().unique()))
diag = perf_full[perf_full["pid"].isin(dropped_perf)][["pid","group","Age","MoCA","n_eval","k_corr","pass"]]
if not diag.empty:
    diag = diag.sort_values(["group","pid"])
    print("\nDropped by performance screen (after MoCA):")
    print(diag.to_string(index=False))
else:
    print("\nDropped by performance screen (after MoCA): (none)")


per_pid = (df_final.drop_duplicates("pid")[["pid","group","Gender","Age","Education","MoCA"]]
           .sort_values(["group","pid"]))
per_pid.to_csv(os.path.join(OUT, "participants_after_filters_p001.csv"), index=False)
df_final.to_csv(os.path.join(OUT, "behavior_clean_final_p001.csv"), index=False)
print("\nSaved:")
print(" -", os.path.join(OUT, "participants_after_filters_p001.csv"))
print(" -", os.path.join(OUT, "behavior_clean_final_p001.csv"))
import matplotlib.pyplot as plt
import numpy as np

# Use the per-participant table after filters
age_tbl = per_pid.dropna(subset=["Age", "group"]).copy()

lo = int(np.floor(age_tbl["Age"].min() // 5 * 5))
hi = int(np.ceil(age_tbl["Age"].max() // 5 * 5) + 1)
bins = np.arange(lo, hi, 2)  # 2-year bins

plt.figure(figsize=(6, 4.5))
for g, color in [("young", "#1f77b4"), ("old", "#d62728")]:
    vec = age_tbl.loc[age_tbl["group"] == g, "Age"].values
    if len(vec) == 0:
        continue
    plt.hist(vec, bins=bins, alpha=0.65, label=f"{g} (n={len(vec)})")

plt.xlabel("Age (years)")
plt.ylabel("Count")
plt.title("Age distribution after MoCA + performance filters")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "age_hist_by_group.png"), dpi=300)
print("Saved figure:", os.path.join(OUT, "age_hist_by_group.png"))
