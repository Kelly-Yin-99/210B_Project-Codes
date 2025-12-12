
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ------------------ CONFIG ------------------
CSV = "/Users/changyin/PyCharmMiscProject/outputs/behavior_clean_final_p001.csv"
SAVE_DIR = "/Users/changyin/PyCharmMiscProject/fig4_log"
os.makedirs(SAVE_DIR, exist_ok=True)

RT_MIN, RT_MAX = 0.1, 3.0

COLOR_H1 = "#1f77b4"  # blue
COLOR_H6 = "#ae0b05"  # red
GREY = "#b3b3b3"

TITLE_SIZE = 18
AXIS_SIZE = 18        # bump for visibility
TICK_SIZE = 14
LEGEND_SIZE = 14

MIN_TRIALS = 8        # min games per subject × horizon for regression

def col(df, name):
    return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)

def observed_means_first4(df):
    """Observed reward means & counts from instructed trials 1–4 (conditioned on chosen side)."""
    R = np.column_stack([col(df, f"r{i}") for i in range(1, 5)])  # rewards t1-4
    C = np.column_stack([col(df, f"c{i}") for i in range(1, 5)])  # choices t1-4 (1=left,2=right)
    C = np.where(np.isnan(C), 0, C)

    Lmask, Rmask = (C == 1), (C == 2)
    nL = Lmask.sum(axis=1).astype(float)
    nR = Rmask.sum(axis=1).astype(float)

    sumL = (R * Lmask).sum(axis=1)
    sumR = (R * Rmask).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        RL = np.where(nL > 0, sumL / nL, np.nan)
        RR = np.where(nR > 0, sumR / nR, np.nan)

    return RL, RR, nL, nR

def stars(p):
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return "ns"

def fit_beta(y, X):
    """OLS β = (X'X)^(-1) X'y with basic safety."""
    # drop rows with NaN
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y_use = y[m]
    X_use = X[m, :]
    if y_use.size < MIN_TRIALS or X_use.shape[0] <= X_use.shape[1]:
        return None
    # least squares
    beta, *_ = np.linalg.lstsq(X_use, y_use, rcond=None)
    return beta  # [β0, βR, βI]


def paired_scatter(ax, y1, y6, ylabel, title):
    """Paired horizon-1 vs horizon-6 scatter with mid-level significance bar."""
    y1 = np.asarray(y1, float)
    y6 = np.asarray(y6, float)

    # Only subjects with both horizons
    mask = np.isfinite(y1) & np.isfinite(y6)

    # grey connecting lines
    for idx in np.where(mask)[0]:
        ax.plot([1, 2], [y1[idx], y6[idx]],
                "-", color=GREY, lw=1, alpha=0.7, zorder=1)

    # scatter points
    ax.scatter(np.ones_like(y1[mask]), y1[mask],
               color=COLOR_H1, zorder=3)
    ax.scatter(np.ones_like(y6[mask]) * 2, y6[mask],
               color=COLOR_H6, zorder=3)

    # axis labels and title
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["1", "6"], fontsize=TICK_SIZE)
    ax.set_xlabel("horizon", fontsize=AXIS_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    # significance (paired t-test)
    if mask.sum() >= 3:
        t, p = stats.ttest_rel(y1[mask], y6[mask])
        if p < 0.05:
            all_y = np.r_[y1[mask], y6[mask]]
            ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)
            # place in the vertical middle of the data range
            mid = ymin + 0.5 * (ymax - ymin)
            bar_height = (ymax - ymin) * 0.02  # short horizontal bar

            # draw horizontal line
            ax.plot([1, 1, 2, 2],
                    [mid - bar_height, mid, mid, mid - bar_height],
                    color="k", lw=1.5)

            # stars directly above the line, still centered
            ax.text(1.5, mid + bar_height * 1.5, stars(p),
                    ha="center", va="bottom",
                    fontsize=AXIS_SIZE + 2, fontweight="bold")




df = pd.read_csv(CSV)

# subject IDs
subj = (df["subjectID"] if "subjectID" in df.columns else df["pid"]).astype(str).values

# horizon & RT on first free choice (trial 5)
gl = col(df, "gameLength").values
c5 = col(df, "c5").values
rt5 = col(df, "rt5").values

valid_rt5 = (rt5 > RT_MIN) & (rt5 < RT_MAX)

# rewards/info from instructed trials
RL, RR, nL, nR = observed_means_first4(df)


dR = RL - RR                   # left - right
dI = (nL - nR) / 2.0           # +1 if left more informative, -1 if right more informative

# choice code a: +1 for left, -1 for right
a = np.full_like(c5, np.nan, dtype=float)
a[c5 == 1] = +1.0
a[c5 == 2] = -1.0

# predictors for regression on trial 5
X_R = a * dR
X_I = a * dI

# - FIT β PER SUBJECT × HORIZON ------------------
betas = []  # list of dicts per subject

for sid in np.unique(subj):
    m_subj = (subj == sid)

    row = {"subjectID": sid,
           "b0_h1": np.nan, "bR_h1": np.nan, "bI_h1": np.nan,
           "b0_h6": np.nan, "bR_h6": np.nan, "bI_h6": np.nan}

    # # horizon 1 (gameLength==5)
    # m_h1 = m_subj & (gl == 5) & valid_rt5
    # if m_h1.sum() >= MIN_TRIALS:
    #     y = rt5[m_h1]
    #     X = np.column_stack([
    #         np.ones(m_h1.sum()),
    #         X_R[m_h1],
    #         X_I[m_h1]
    #     ])
    #     beta = fit_beta(y, X)
    #     if beta is not None:
    #         row["b0_h1"], row["bR_h1"], row["bI_h1"] = beta
    #
    # # horizon 6 (gameLength==10)
    # m_h6 = m_subj & (gl == 10) & valid_rt5
    # if m_h6.sum() >= MIN_TRIALS:
    #     y = rt5[m_h6]
    #     X = np.column_stack([
    #         np.ones(m_h6.sum()),
    #         X_R[m_h6],
    #         X_I[m_h6]
    #     ])
    #     beta = fit_beta(y, X)
    #     if beta is not None:
    #         row["b0_h6"], row["bR_h6"], row["bI_h6"] = beta
    # horizon 1 (gameLength==5)
    m_h1 = m_subj & (gl == 5) & valid_rt5
    if m_h1.sum() >= MIN_TRIALS:
        # use log RT as the dependent variable
        y = np.log(rt5[m_h1])
        X = np.column_stack([
            np.ones(m_h1.sum()),
            X_R[m_h1],
            X_I[m_h1]
        ])
        beta = fit_beta(y, X)
        if beta is not None:
            row["b0_h1"], row["bR_h1"], row["bI_h1"] = beta

    # horizon 6 (gameLength==10)
    m_h6 = m_subj & (gl == 10) & valid_rt5
    if m_h6.sum() >= MIN_TRIALS:
        # use log RT as the dependent variable
        y = np.log(rt5[m_h6])
        X = np.column_stack([
            np.ones(m_h6.sum()),
            X_R[m_h6],
            X_I[m_h6]
        ])
        beta = fit_beta(y, X)
        if beta is not None:
            row["b0_h6"], row["bR_h6"], row["bI_h6"] = beta

    betas.append(row)

par = pd.DataFrame(betas)
par_path = os.path.join(SAVE_DIR, "fig4_log_EFG_betas_per_subject.csv")
par.to_csv(par_path, index=False)

# ---------- ADD AGE & GROUP ----------
# Get age column name
age_col = "Age" if "Age" in df.columns else ("age" if "age" in df.columns else None)
if age_col is None:
    raise ValueError("No Age/age column found for grouping.")

# Map age per subject
df_tmp = df.copy()
df_tmp["subj"] = subj
age_by_subj = df_tmp.groupby("subj")[age_col].first()
par["age"] = par["subjectID"].map(age_by_subj)

# Define groups (edit bounds if needed)
YOUNG = (18, 35)
OLD   = (60, 80)

def assign_group(a):
    if pd.isna(a):
        return np.nan
    if YOUNG[0] <= a <= YOUNG[1]:
        return "young"
    if OLD[0] <= a <= OLD[1]:
        return "old"
    return "other"

par["group"] = par["age"].apply(assign_group)


par_anova = par[par["group"].isin(["young", "old"])].copy()



def mixed_anova_param(par_df, prefix, label):
    """
    prefix: 'b0', 'bR', or 'bI'
    label: pretty name for printing
    """
    df_long = pd.DataFrame({
        "subjectID": np.repeat(par_df["subjectID"].values, 2),
        "group":     np.repeat(par_df["group"].values, 2),
        "horizon":   np.tile([1, 6], len(par_df)),
        "beta": np.concatenate([
            par_df[f"{prefix}_h1"].values,
            par_df[f"{prefix}_h6"].values
        ])
    })
    df_long = df_long.dropna()

    # Mixed ANOVA: group (between) × horizon (within)
    aov = pg.mixed_anova(
        dv="beta",
        within="horizon",
        between="group",
        subject="subjectID",
        data=df_long
    )

    print(f"\n[Mixed ANOVA] {label}")
    print(aov[["Source", "F", "p-unc", "np2"]])

    # Post-hoc: between groups at each horizon + within-group horizon effect
    post = pg.pairwise_tests(
        dv="beta",
        between="group",
        within="horizon",
        subject="subjectID",
        data=df_long,
        padjust="bonf"
    )
    print(f"[Post-hoc tests] {label}")
    print(post)

    return aov, post

aov_b0, post_b0 = mixed_anova_param(par_anova, "b0", "β0 (baseline RT)")
aov_bR, post_bR = mixed_anova_param(par_anova, "bR", "βR (effect of aΔR)")
aov_bI, post_bI = mixed_anova_param(par_anova, "bI", "βI (effect of aΔI)")


fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))

# E: baseline RT β0
paired_scatter(
    axs[0],
    par["b0_h1"].values,
    par["b0_h6"].values,
    r"$\beta_0$",
    "baseline RT"
)

# F: effect of aΔR (βR)
paired_scatter(
    axs[1],
    par["bR_h1"].values,
    par["bR_h6"].values,
    r"$\beta_R$",
    "effect of $a\\Delta R$"
)

# G: effect of aΔI (βI)
paired_scatter(
    axs[2],
    par["bI_h1"].values,
    par["bI_h6"].values,
    r"$\beta_I$  (effect of $a\Delta I$)",
    "effect of $a\\Delta I$"
)

fig.tight_layout()
fig_out = os.path.join(SAVE_DIR, "figure4_EFG.png")
fig.savefig(fig_out, dpi=300)

print("Saved:")
print("-", par_path)
print("-", fig_out)


age_col = None
if "Age" in df.columns:
    age_col = "Age"
elif "age" in df.columns:
    age_col = "age"

if age_col is not None:
    # map subject -> age (first non-NaN occurrence)
    subj_age = {}
    for sid in par["subjectID"]:
        m = (subj == sid)
        ag = pd.to_numeric(df.loc[m, age_col], errors="coerce")
        ag = ag[np.isfinite(ag)]
        subj_age[sid] = float(ag.iloc[0]) if len(ag) else np.nan

    par["age"] = par["subjectID"].map(subj_age)

    def classify_group(a):
        if np.isnan(a):
            return "other"
        if 18 <= a <= 25:
            return "young"
        if 65 <= a <= 74:
            return "old"
        return "other"

    par["group"] = par["age"].apply(classify_group)
else:
   
    par["group"] = "all"


rows = []
for _, r in par.iterrows():
    sid = r["subjectID"]
    grp = r["group"]

    # horizon 1
    rows.append({
        "subject": sid,
        "group": grp,
        "horizon": "1",
        "b0": r["b0_h1"],
        "bR": r["bR_h1"],
        "bI": r["bI_h1"],
    })
    # horizon 6
    rows.append({
        "subject": sid,
        "group": grp,
        "horizon": "6",
        "b0": r["b0_h6"],
        "bR": r["bR_h6"],
        "bI": r["bI_h6"],
    })

df_betas = pd.DataFrame(rows)

complete_subj = (
    df_betas
    .dropna(subset=["b0", "bR", "bI"], how="all")
    .groupby("subject")["horizon"].nunique()
)
complete_subj = complete_subj[complete_subj == 2].index
df_betas = df_betas[df_betas["subject"].isin(complete_subj)]

# restrict to young vs old for the mixed ANOVA
df_betas_yo = df_betas[df_betas["group"].isin(["young", "old"])].copy()

# 3) Run mixed ANOVA using pingouin, if available
try:
    import pingouin as pg

    for dv, label in [("b0", "β0 (baseline RT)"),
                      ("bR", "βR (effect of aΔR)"),
                      ("bI", "βI (effect of aΔI)")]:
        sub = df_betas_yo[np.isfinite(df_betas_yo[dv])]
        if sub["subject"].nunique() < 3:
            print(f"\n[Mixed ANOVA] Not enough complete subjects for {label}.")
            continue

        print(f"\n[Mixed ANOVA] {label}")
        aov = pg.mixed_anova(
            dv=dv,
            within="horizon",
            between="group",
            subject="subject",
            data=sub
        )
        print(aov)

        # Optional pairwise follow-ups (comment out if you don't want spam)
        # post = pg.pairwise_ttests(
        #     dv=dv,
        #     within="horizon",
        #     between="group",
        #     subject="subject",
        #     padjust="bonf",
        #     data=sub
        # )
        # print(post)

except ImportError:
    print("\nTo run the mixed ANOVA, install pingouin:\n  pip install pingouin")
    print("Then re-run this script.")




import matplotlib.pyplot as plt
import pandas as pd
import os

SAVE_DIR = "/Users/changyin/PyCharmMiscProject/fig4_log"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- ANOVA SUMMARY (your numbers) ----------
anova_summary = pd.DataFrame([
    ["β₀ ",   40.89, 8.30e-09, 0.325, 0.354, 5.54e-01, 0.004, 14.62, 2.50e-04, 0.147],
    ["βᴿ", 0.028, 8.67e-01, 0.000, 41.60, 6.50e-09, 0.329,  8.43, 4.70e-03, 0.090],
    ["βᴵ",   3.34,  7.10e-02, 0.038, 13.39, 4.40e-04, 0.136,  0.21, 6.46e-01, 0.002],
], columns=[
    "Parameter",
    "F_group", "p_group", "η²_group",
    "F_horizon", "p_horizon", "η²_horizon",
    "F_interaction", "p_interaction", "η²_interaction"
])


for col in ["p_group", "p_horizon", "p_interaction"]:
    anova_summary[col] = anova_summary[col].map(lambda x: f"{x:.2e}")


fig, ax = plt.subplots(figsize=(14, 3.8))  # wider + taller
ax.axis("off")

tbl = ax.table(
    cellText=anova_summary.values,
    colLabels=anova_summary.columns,
    loc="center",
    cellLoc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 1.5)


for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#0c234b")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    elif c == 0 and r > 0:
        cell.get_text().set_weight("bold")

# highlight significant p-values (p < .05)
p_cols = [2, 5, 8]  # p_group, p_horizon, p_interaction
for r in range(1, anova_summary.shape[0] + 1):
    for pc in p_cols:
        txt = tbl[r, pc].get_text().get_text()
        try:
            p_val = float(txt)
        except ValueError:
            continue
        if p_val < 0.05:
            cell = tbl[r, pc]
            cell.set_facecolor("#fde0dd")
            cell.get_text().set_weight("bold")

fig.tight_layout(pad=0.5)
outpath = os.path.join(SAVE_DIR, "anova_summary_table.png")
fig.savefig(outpath, dpi=300, bbox_inches="tight")
print("Saved:", outpath)

