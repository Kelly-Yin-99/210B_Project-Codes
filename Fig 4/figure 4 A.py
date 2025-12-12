

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


CSV = "/Users/changyin/PyCharmMiscProject/outputs/behavior_clean_final_p001.csv"
SAVE_DIR = "/Users/changyin/PyCharmMiscProject/fig4"
os.makedirs(SAVE_DIR, exist_ok=True)


COLOR_H1 = "#1f77b4"  # blue
COLOR_H6 = "#ae0b05"  # red
GREY_SHADE = (0.92, 0.92, 0.92)
CREAM_SHADE = (0.98, 0.95, 0.88) # first free trial
TITLE_SIZE = 18
AXIS_SIZE = 20          # bumped for more visible axis labels
TICK_SIZE = 13
LEGEND_SIZE = 13
LW = 2.2
CAP = 3
MS = 6

RT_MIN, RT_MAX = 0.1, 3.0

# ------------------ HELPERS ------------------
def col(df, name):
    return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)

def observed_means_first4(df):

    R = np.column_stack([col(df, f"r{i}") for i in range(1, 5)])  # rewards on t1-4
    C = np.column_stack([col(df, f"c{i}") for i in range(1, 5)])  # choices on t1-4 (1=left, 2=right)
    C = np.where(np.isnan(C), 0, C)
    Lmask, Rmask = (C == 1), (C == 2)
    nL = Lmask.sum(1).astype(float); nR = Lmask.shape[1] - nL  # or recompute from Rmask
    sumL = (R * Lmask).sum(1);        sumR = (R * Rmask).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        RL = np.where(nL > 0, sumL / nL, np.nan)
        RR = np.where(nR > 0, sumR / nR, np.nan)
    return RL, RR, nL, nR

def sem(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return np.nan if a.size == 0 else stats.sem(a)

def line_with_err(ax, xs, m1, e1, m6, e6, label1, label6):
    ax.errorbar(xs, m1, yerr=e1, color=COLOR_H1, lw=LW, capsize=CAP,
                marker="o", ms=MS, label=label1)
    ax.errorbar(xs, m6, yerr=e6, color=COLOR_H6, lw=LW, capsize=CAP,
                marker="o", ms=MS, label=label6)

def shade_instructed_and_trial1(ax, n_instr=4):
    ax.axvspan(0.5, n_instr+0.5, color=GREY_SHADE, zorder=0)
    ax.axvspan(n_instr+0.5, n_instr+1.5, color=CREAM_SHADE, zorder=0)

def make_symmetric_bins(x, base_mask, valid_mask, n_bins=5):
    if hasattr(base_mask, "values"):
        base_mask = base_mask.values
    sel = base_mask & valid_mask & np.isfinite(x)
    if not np.any(sel):
        edges = np.linspace(-1, 1, n_bins + 1)
    else:
        maxabs = np.nanmax(np.abs(x[sel]))
        if maxabs == 0 or not np.isfinite(maxabs):
            maxabs = 1.0
        edges = np.linspace(-maxabs, maxabs, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers

def binned_rt(x, rt, cond_mask, horiz_mask, valid_mask, bin_edges):
    if hasattr(cond_mask, "values"):
        cond_mask = cond_mask.values
    if hasattr(horiz_mask, "values"):
        horiz_mask = horiz_mask.values
    sel = cond_mask & horiz_mask & valid_mask & np.isfinite(x)
    xb, rb = x[sel], rt[sel]
    means, errors = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        pick = (xb >= lo) & (xb < hi)
        vals = rb[pick]
        if vals.size:
            means.append(np.nanmean(vals))
            errors.append(sem(vals))
        else:
            means.append(np.nan)
            errors.append(np.nan)
    return np.array(means), np.array(errors)

# ------------------ LOAD ------------------
df = pd.read_csv(CSV)

rt_cols = [f"rt{i}" for i in range(1, 11) if f"rt{i}" in df.columns]
if len(rt_cols) < 5:
    raise ValueError("Need at least rt1..rt5 in the CSV for Fig. 4.")

RT = np.column_stack([col(df, c) for c in rt_cols])
T = RT.shape[1]

n_instr = 4
n_free = max(0, T - n_instr)

gl = col(df, "gameLength")
is_h1 = (gl == 5)
is_h6 = (gl == 10)

RT_valid = (RT > RT_MIN) & (RT < RT_MAX)

RL, RR, nL, nR = observed_means_first4(df)

is_unequal = (nL != nR)
is_equal = (nL == nR)

# ------------------ Panel A & B: trial-wise RT curves ------------------
def per_trial_stats(mask_condition, mask_horizon):
    m = (mask_condition & mask_horizon).values
    xs = np.arange(1, T+1)
    means, errors = [], []
    for t in range(T):
        sel = m & RT_valid[:, t]
        vals = RT[sel, t]
        means.append(np.nanmean(vals) if vals.size else np.nan)
        errors.append(sem(vals) if vals.size else np.nan)
    return xs, np.array(means), np.array(errors)

xs_A, m1_A, e1_A = per_trial_stats(is_unequal, is_h1)
_,    m6_A, e6_A = per_trial_stats(is_unequal, is_h6)

xs_B, m1_B, e1_B = per_trial_stats(is_equal, is_h1)
_,    m6_B, e6_B = per_trial_stats(is_equal, is_h6)

# ------------------ Panel C & D ------------------
if T < 5:
    raise ValueError("Need rt5 (first free choice) to build panels C/D.")
t5 = 4
rt5 = RT[:, t5]
valid_t5 = RT_valid[:, t5]

# Unequal: ΔR = R(high info) - R(low info)
dR_hi_low = np.where(nL < nR, RL - RR,
               np.where(nR < nL, RR - RL, np.nan))
bin_edges_C, bin_cent_C = make_symmetric_bins(dR_hi_low, is_unequal, valid_t5, n_bins=5)
mC_h1, eC_h1 = binned_rt(dR_hi_low, rt5, is_unequal, is_h1, valid_t5, bin_edges_C)
mC_h6, eC_h6 = binned_rt(dR_hi_low, rt5, is_unequal, is_h6, valid_t5, bin_edges_C)

# Equal: ΔR = R(left) - R(right)
dR_left_right = RL - RR
bin_edges_D, bin_cent_D = make_symmetric_bins(dR_left_right, is_equal, valid_t5, n_bins=5)
mD_h1, eD_h1 = binned_rt(dR_left_right, rt5, is_equal, is_h1, valid_t5, bin_edges_D)
mD_h6, eD_h6 = binned_rt(dR_left_right, rt5, is_equal, is_h6, valid_t5, bin_edges_D)

# ------------------ Plot Figure 4A–D ------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
(axA, axB), (axC, axD) = axs

# ---- A: unequal information [1 3] ----
shade_instructed_and_trial1(axA, n_instr=n_instr)
line_with_err(axA, xs_A, m1_A, e1_A, m6_A, e6_A, "horizon 1", "horizon 6")
axA.set_title("unequal information [1 3]", fontsize=AXIS_SIZE, fontweight="bold")
labels_A = [f"i{i}" for i in range(1, min(n_instr, T)+1)] + \
           [str(i) for i in range(1, max(0, T-n_instr)+1)]
axA.set_xticks(xs_A)
axA.set_xticklabels(labels_A, fontsize=TICK_SIZE)
axA.set_xlabel("trial number", fontsize=AXIS_SIZE, fontweight="bold")   # <-- added
axA.set_ylabel("RT", fontsize=AXIS_SIZE, fontweight="bold")
axA.legend(frameon=False, fontsize=LEGEND_SIZE)
axA.tick_params(labelsize=TICK_SIZE)

# ---- B: equal information [2 2] ----
shade_instructed_and_trial1(axB, n_instr=n_instr)
line_with_err(axB, xs_B, m1_B, e1_B, m6_B, e6_B, "horizon 1", "horizon 6")
axB.set_title("equal information [2 2]",fontsize=AXIS_SIZE, fontweight="bold")
labels_B = [f"i{i}" for i in range(1, min(n_instr, T)+1)] + \
           [str(i) for i in range(1, max(0, T-n_instr)+1)]
axB.set_xticks(xs_B)
axB.set_xticklabels(labels_B, fontsize=TICK_SIZE)
axB.set_xlabel("trial number", fontsize=AXIS_SIZE, fontweight="bold")   # <-- added

axB.tick_params(labelsize=TICK_SIZE)

# Match y-lims for top row
top_vals = np.r_[m1_A, m6_A, m1_B, m6_B]
if np.isfinite(top_vals).any():
    lo = np.nanmin(top_vals) - 0.05
    hi = np.nanmax(top_vals) + 0.05
    axA.set_ylim(lo, hi)
    axB.set_ylim(lo, hi)

# ---- C: RT vs R(high)-R(low) (unequal) ----
axC.errorbar(bin_cent_C, mC_h1, yerr=eC_h1, color=COLOR_H1, lw=LW,
             capsize=CAP, marker="o", ms=MS, label="horizon 1")
axC.errorbar(bin_cent_C, mC_h6, yerr=eC_h6, color=COLOR_H6, lw=LW,
             capsize=CAP, marker="o", ms=MS, label="horizon 6")
axC.set_xlabel("difference in mean reward\nR(high info) - R(low info)",
               fontsize=AXIS_SIZE, fontweight="bold")
axC.set_ylabel("RT",
               fontsize=AXIS_SIZE, fontweight="bold")
axC.tick_params(labelsize=TICK_SIZE)
axC.legend(frameon=False, fontsize=LEGEND_SIZE)

# ---- D: RT vs R(left)-R(right) (equal) ----
axD.errorbar(bin_cent_D, mD_h1, yerr=eD_h1, color=COLOR_H1, lw=LW,
             capsize=CAP, marker="o", ms=MS, label="horizon 1")
axD.errorbar(bin_cent_D, mD_h6, yerr=eD_h6, color=COLOR_H6, lw=LW,
             capsize=CAP, marker="o", ms=MS, label="horizon 6")
axD.set_xlabel("difference in mean reward\nR(left) - R(right)",
               fontsize=AXIS_SIZE, fontweight="bold")
axD.set_ylabel("RT",
               fontsize=AXIS_SIZE, fontweight="bold")
axD.tick_params(labelsize=TICK_SIZE)
axD.legend(frameon=False, fontsize=LEGEND_SIZE)

fig.tight_layout()
outpath = os.path.join(SAVE_DIR, "figure4_ABCD.png")
fig.savefig(outpath, dpi=300)
print("Saved:", outpath)
