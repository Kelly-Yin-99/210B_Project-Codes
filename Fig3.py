

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats


CSV = "/Users/changyin/PyCharmMiscProject/outputs/behavior_clean_final_p001.csv"
SAVE_DIR = "/Users/changyin/PyCharmMiscProject/fig3_CDE"
os.makedirs(SAVE_DIR, exist_ok=True)

RT_MIN, RT_MAX = 0.1, 3.0
COLOR_H1 = "#1f77b4"  # blue
COLOR_H6 = "#ae0b05"  # red
GREY = "#b3b3b3"
TITLE_SIZE, AXIS_SIZE, TICK_SIZE = 18, 16, 12
YOUNG = (18, 25); OLD = (65, 74)

# Bounds & tolerance
BND_A = (-100.0, 100.0)
BND_B = (-100.0, 100.0)
BND_SIG = (1e-6, 100.0)
BOUND_EPS = 1e-9

def pick_age_col(df): return "Age" if "Age" in df.columns else "age"
def col(df, name):   return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)

def observed_means_first4(df):

    R = np.column_stack([col(df, f"r{i}") for i in range(1, 5)])  # rewards on t1-4
    C = np.column_stack([col(df, f"c{i}") for i in range(1, 5)])  # choices on t1-4 (1=left, 2=right)
    C = np.where(np.isnan(C), 0, C)
    Lmask, Rmask = (C == 1), (C == 2)
    nL = Lmask.sum(1).astype(float); nR = Rmask.sum(1).astype(float)
    sumL = (R * Lmask).sum(1);        sumR = (R * Rmask).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        RL = np.where(nL > 0, sumL / nL, np.nan)
        RR = np.where(nR > 0, sumR / nR, np.nan)
    return RL, RR, nL, nR

def _stars(p):  # internal: return "", "*", "**", "***"
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else ""))

def _draw_sig(ax, x1, x2, y, p, star_size=22, color="k"):
    s = _stars(p)
    if not s:  # don't draw if not significant
        return
    ax.plot([x1, x1, x2, x2], [y*0.99, y, y, y*0.99], color=color, lw=1.6)
    ax.text((x1+x2)/2, y*1.02, s, ha="center", va="bottom",
            fontsize=star_size, fontweight="bold", color=color)


def nll_unequal(x, dR, dI, choose_right):
    # theta = [B, sigma, A]
    B, sigma, A = x
    if not (BND_SIG[0] <= sigma <= BND_SIG[1]):  # quick reject
        return 1e9
    z = (dR + A * dI + B) / (np.sqrt(2.0) * sigma)
    log_p   = -np.logaddexp(0.0, -z)
    log_1mp = -np.logaddexp(0.0,  z)
    y = choose_right.astype(float)
    return -np.nansum(y * log_p + (1.0 - y) * log_1mp)

def nll_equal(x, dR, choose_right):
    # theta = [B, sigma]
    B, sigma = x
    if not (BND_SIG[0] <= sigma <= BND_SIG[1]):
        return 1e9
    z = (dR + B) / (np.sqrt(2.0) * sigma)
    log_p   = -np.logaddexp(0.0, -z)
    log_1mp = -np.logaddexp(0.0,  z)
    y = choose_right.astype(float)
    return -np.nansum(y * log_p + (1.0 - y) * log_1mp)

def fit_subject_block(dR, dI, y_right, mode):
    """Fit one subject × (condition,horizon) block."""
    m = np.isfinite(dR) & np.isfinite(y_right)
    if mode == "unequal":
        m &= np.isfinite(dI) & (dI != 0)
        dI_use = dI[m]
    dR_use, y_use = dR[m], y_right[m]
    if dR_use.size < 8:  # require minimum trials to stabilize fits
        return None

    if mode == "unequal":
        x0 = np.array([0.0, 10.0, 0.0])  # [B, sigma, A]
        bnds = [BND_B, BND_SIG, BND_A]
        res = minimize(nll_unequal, x0, args=(dR_use, dI_use, y_use), method="L-BFGS-B", bounds=bnds)
        if not res.success:
            res = minimize(nll_unequal, np.array([5.0, 15.0, 5.0]), args=(dR_use, dI_use, y_use), method="L-BFGS-B", bounds=bnds)
        B, sigma, A = res.x
        hitA = (abs(A - BND_A[1]) <= BOUND_EPS) or (abs(A - BND_A[0]) <= BOUND_EPS)
        hitS = (abs(sigma - BND_SIG[1]) <= BOUND_EPS) or (abs(sigma - BND_SIG[0]) <= 1e-12)
        return {"A": A, "B": B, "sigma": sigma, "hit_A": hitA, "hit_sigma": hitS}
    else:
        x0 = np.array([0.0, 10.0])       # [B, sigma]
        bnds = [BND_B, BND_SIG]
        res = minimize(nll_equal, x0, args=(dR_use, y_use), method="L-BFGS-B", bounds=bnds)
        if not res.success:
            res = minimize(nll_equal, np.array([5.0, 15.0]), args=(dR_use, y_use), method="L-BFGS-B", bounds=bnds)
        B, sigma = res.x
        hitS = (abs(sigma - BND_SIG[1]) <= BOUND_EPS) or (abs(sigma - BND_SIG[0]) <= 1e-12)
        return {"A": 0.0, "B": B, "sigma": sigma, "hit_A": False, "hit_sigma": hitS}


df = pd.read_csv(CSV)
age_col = pick_age_col(df); age = col(df, age_col)

gl = col(df, "gameLength"); is_h1 = (gl == 5); is_h6 = (gl == 10)
c5 = col(df, "c5");        rt5 = col(df, "rt5")
valid = (rt5 > RT_MIN) & (rt5 < RT_MAX)
choose_right = (c5 == 2).astype(float)


RL, RR, nL, nR = observed_means_first4(df)


dR = RR - RL                        # right - left
dI = -(nR - nL) / 2.0               # +1 when RIGHT is more informative; -1 when LEFT is more informative

is_unequal = (nL != nR) & valid
is_equal   = (nL == nR) & valid

subj = (df["subjectID"] if "subjectID" in df.columns else df["pid"]).astype(str)

def fit_all(mode):
    out_h1, out_h6 = {}, {}
    for sid in np.unique(subj):
        m1 = (subj == sid) & (is_h1) & (is_unequal if mode=="unequal" else is_equal)
        m6 = (subj == sid) & (is_h6) & (is_unequal if mode=="unequal" else is_equal)
        if m1.any():
            res = fit_subject_block(dR[m1], dI[m1], choose_right[m1], mode)
            if res: out_h1[sid] = res
        if m6.any():
            res = fit_subject_block(dR[m6], dI[m6], choose_right[m6], mode)
            if res: out_h6[sid] = res
    return out_h1, out_h6

unequal_h1, unequal_h6 = fit_all("unequal")  # A,B,σ
equal_h1,   equal_h6   = fit_all("equal")    # B,σ (A fixed 0)


rows = []
for sid in np.unique(subj):
    ag  = float(age[subj==sid].iloc[0]) if np.any(subj==sid) else np.nan
    grp = "young" if YOUNG[0] <= ag <= YOUNG[1] else ("old" if OLD[0] <= ag <= OLD[1] else "other")
    r = {"subjectID": sid, "age": ag, "group": grp,
         "A_h1": np.nan, "A_h6": np.nan,
         "sigma_unequal_h1": np.nan, "sigma_unequal_h6": np.nan,
         "sigma_equal_h1": np.nan, "sigma_equal_h6": np.nan,
         "hitA_h1": False, "hitA_h6": False,
         "hitSigUnequal_h1": False, "hitSigUnequal_h6": False,
         "hitSigEqual_h1": False, "hitSigEqual_h6": False}
    if sid in unequal_h1:
        r["A_h1"]=unequal_h1[sid]["A"]; r["sigma_unequal_h1"]=unequal_h1[sid]["sigma"]
        r["hitA_h1"]=unequal_h1[sid]["hit_A"]; r["hitSigUnequal_h1"]=unequal_h1[sid]["hit_sigma"]
    if sid in unequal_h6:
        r["A_h6"]=unequal_h6[sid]["A"]; r["sigma_unequal_h6"]=unequal_h6[sid]["sigma"]
        r["hitA_h6"]=unequal_h6[sid]["hit_A"]; r["hitSigUnequal_h6"]=unequal_h6[sid]["hit_sigma"]
    if sid in equal_h1:
        r["sigma_equal_h1"]=equal_h1[sid]["sigma"]; r["hitSigEqual_h1"]=equal_h1[sid]["hit_sigma"]
    if sid in equal_h6:
        r["sigma_equal_h6"]=equal_h6[sid]["sigma"]; r["hitSigEqual_h6"]=equal_h6[sid]["hit_sigma"]
    rows.append(r)

par = pd.DataFrame(rows)
par_full_path = os.path.join(SAVE_DIR, "fig3_params_per_subject_full.csv")
par.to_csv(par_full_path, index=False)


par_clean = par.copy()
par_clean.loc[par_clean["hitA_h1"]==True, "A_h1"] = np.nan
par_clean.loc[par_clean["hitA_h6"]==True, "A_h6"] = np.nan
par_clean.loc[par_clean["hitSigUnequal_h1"]==True, "sigma_unequal_h1"] = np.nan
par_clean.loc[par_clean["hitSigUnequal_h6"]==True, "sigma_unequal_h6"] = np.nan
par_clean.loc[par_clean["hitSigEqual_h1"]==True,   "sigma_equal_h1"]   = np.nan
par_clean.loc[par_clean["hitSigEqual_h6"]==True,   "sigma_equal_h6"]   = np.nan
par_clean_path = os.path.join(SAVE_DIR, "fig3_params_per_subject_clean.csv")
par_clean.to_csv(par_clean_path, index=False)

def paired_scatter(ax, y1, y6, ylabel, title):
    x1, x6 = np.ones_like(y1), np.ones_like(y6)*2
    for i in range(len(y1)):
        if np.isfinite(y1[i]) and np.isfinite(y6[i]):
            ax.plot([1,2], [y1[i], y6[i]], "-", color=GREY, lw=1, alpha=0.8)
    ax.scatter(x1, y1, color=COLOR_H1, zorder=3, s=38)
    ax.scatter(x6, y6, color=COLOR_H6, zorder=3, s=38)
    ax.set_xticks([1,2]); ax.set_xticklabels(["1","6"], fontsize=TICK_SIZE+4, fontweight="bold")
    ax.set_xlabel("horizon", fontsize=AXIS_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=6)
    # paired t-test + bracket (only if significant)
    mask = np.isfinite(y1) & np.isfinite(y6)
    if mask.sum() >= 3:
        t, p = stats.ttest_rel(y1[mask], y6[mask], nan_policy="omit")
        if _stars(p):  # only draw when significant
            ymax = np.nanpercentile(np.r_[y1[mask], y6[mask]], 97) * 1.05
            _draw_sig(ax, 1, 2, ymax, p, star_size=24)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
paired_scatter(axs[0], par_clean["A_h1"].values, par_clean["A_h6"].values,
               "information bonus (points)", "directed [1 3]")
paired_scatter(axs[1], par_clean["sigma_unequal_h1"].values, par_clean["sigma_unequal_h6"].values,
               "decision noise (points)", "random [1 3]")
paired_scatter(axs[2], par_clean["sigma_equal_h1"].values, par_clean["sigma_equal_h6"].values,
               "decision noise (points)", "random [2 2]")
fig.tight_layout()
figCDE = os.path.join(SAVE_DIR, "fig3_CDE_paired_clean.png")
fig.savefig(figCDE, dpi=300)


def grouped_paired_by_group(df_, col_h1, col_h6, ylabel, title, out_path):
    """
    x = ['young','old']; within each group plot paired h1 (blue) vs h6 (red),
    grey lines connect h1->h6 per subject; per-group bracket only if p<.05.
    Ensures y-lims include all points AND any significance brackets.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x_pos = {"young": 1.0, "old": 2.0}
    jitter = 0.085

    # We'll record the highest y used by any bracket so we can set ylim safely.
    needed_bracket_top = -np.inf

    for grp, x in x_pos.items():
        d = df_.loc[df_["group"] == grp, [col_h1, col_h6]].dropna()
        if d.empty:
            continue
        y1, y6 = d[col_h1].values, d[col_h6].values

        # grey within-subject lines
        for i in range(len(y1)):
            ax.plot([x - jitter, x + jitter], [y1[i], y6[i]], "-", color=GREY, lw=1, alpha=0.8)

        # dots
        ax.scatter(np.full_like(y1, x - jitter), y1, color=COLOR_H1, zorder=3, s=40)
        ax.scatter(np.full_like(y6, x + jitter), y6, color=COLOR_H6, zorder=3, s=40)

        # paired t-test h1 vs h6 within group (draw only if significant)
        if np.isfinite(y1).sum() >= 3 and np.isfinite(y6).sum() >= 3:
            t, p = stats.ttest_rel(y1, y6, nan_policy="omit")
            if _stars(p):
                # propose a bracket y a bit above the local data for this group
                local_hi = np.nanmax(np.r_[y1, y6])
                bracket_y = local_hi + 0.06 * max(1.0, local_hi - np.nanmin(np.r_[y1, y6]))
                _draw_sig(ax, x - jitter, x + jitter, bracket_y, p, star_size=24)
                needed_bracket_top = max(needed_bracket_top, bracket_y * 1.06)  # small extra headroom

    # cosmetics
    ax.set_xticks([x_pos["young"], x_pos["old"]])
    ax.set_xticklabels(["young", "old"], fontsize=TICK_SIZE+4, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=AXIS_SIZE)
    ax.set_xlabel("age group", fontsize=AXIS_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=6)


    all_vals = df_[[col_h1, col_h6]].values.flatten()
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        # start from full min/max to guarantee all points fit
        data_lo, data_hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

        # also compute percentile range (to avoid ultra-outlier over-zoom)
        p1, p99 = np.nanpercentile(all_vals, [1, 99])
        lo = min(data_lo, p1)
        hi = max(data_hi, p99)

        span = max(1.0, hi - lo)
        pad  = 0.12 * span

        top_needed = max(hi + pad, needed_bracket_top if np.isfinite(needed_bracket_top) else -np.inf)
        ax.set_ylim(lo - pad, top_needed)

    ax.tick_params(labelsize=TICK_SIZE+2)
    # small safety margin for artists near the edge
    ax.margins(y=0.06)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# Directed A
grouped_paired_by_group(
    par_clean, "A_h1", "A_h6",
    "information bonus (points)", "directed [1 3] — grouped",
    os.path.join(SAVE_DIR, "fig_group_directed_A.png")
)

# Random [1 3] sigma (unequal)
grouped_paired_by_group(
    par_clean, "sigma_unequal_h1", "sigma_unequal_h6",
    "decision noise (points)", "random [1 3] — grouped",
    os.path.join(SAVE_DIR, "fig_group_random_13.png")
)

# Random [2 2] sigma (equal)
grouped_paired_by_group(
    par_clean, "sigma_equal_h1", "sigma_equal_h6",
    "decision noise (points)", "random [2 2] — grouped",
    os.path.join(SAVE_DIR, "fig_group_random_22.png")
)

print("Saved:")
print("-", figCDE)
print("-", par_full_path)
print("-", par_clean_path)
print("-", os.path.join(SAVE_DIR, "fig_group_directed_A.png"))
print("-", os.path.join(SAVE_DIR, "fig_group_random_13.png"))
print("-", os.path.join(SAVE_DIR, "fig_group_random_22.png"))


def _scatter_two_groups(ax, y_vals, o_vals, color):
    """
    Jittered scatter for two independent groups (no mean markers).
    Returns the highest y we placed a bracket at (or -inf if none).
    """
    x_left, x_right = 1.0, 2.0
    jitter = 0.06

    if y_vals.size:
        ax.scatter(np.full_like(y_vals, x_left) +
                   np.random.uniform(-jitter, jitter, size=y_vals.size),
                   y_vals, s=40, color=color, alpha=0.95, zorder=3)
    if o_vals.size:
        ax.scatter(np.full_like(o_vals, x_right) +
                   np.random.uniform(-jitter, jitter, size=o_vals.size),
                   o_vals, s=40, color=color, alpha=0.95, zorder=3)

    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(["young", "old"], fontsize=TICK_SIZE+4, fontweight="bold")

    bracket_top = -np.inf
    if y_vals.size >= 3 and o_vals.size >= 3:
        t, p = stats.ttest_ind(y_vals, o_vals, equal_var=False, nan_policy="omit")
        if _stars(p):
            local_lo = np.nanmin(np.r_[y_vals, o_vals])
            local_hi = np.nanmax(np.r_[y_vals, o_vals])
            span = max(1.0, local_hi - local_lo)
            # Put bracket comfortably above points to avoid title overlap
            y_br = local_hi + 0.20 * span
            _draw_sig(ax, x_left, x_right, y_br, p, star_size=25)
            bracket_top = y_br
    return bracket_top


def _get_group_vals(df_, col):
    d = df_.loc[df_[col].notna(), ["group", col]]
    y = d.loc[d["group"] == "young", col].astype(float).values
    o = d.loc[d["group"] == "old",   col].astype(float).values
    return y, o


def figure_group_vs_group_two_panels(df_, col_h1, col_h6, ylabel, suptitle, out_path,
                                     color_h1=COLOR_H1, color_h6=COLOR_H6):

    y1, o1 = _get_group_vals(df_, col_h1)
    y6, o6 = _get_group_vals(df_, col_h6)

    fig, axs = plt.subplots(1, 2, figsize=(11.2, 5.6), sharey=True)
    fig.suptitle(suptitle, fontsize=TITLE_SIZE+1, y=0.98)

    # Left: H1
    axs[0].set_title("h1", fontsize=TITLE_SIZE-2, pad=10)
    top0 = _scatter_two_groups(axs[0], y1, o1, color_h1)
    axs[0].set_ylabel(ylabel, fontsize=AXIS_SIZE)
    axs[0].set_xlabel("age group", fontsize=AXIS_SIZE)
    axs[0].tick_params(labelsize=TICK_SIZE+2)

    # Right: H6
    axs[1].set_title("h6", fontsize=TITLE_SIZE-2, pad=10)
    top1 = _scatter_two_groups(axs[1], y6, o6, color_h6)
    axs[1].set_xlabel("age group", fontsize=AXIS_SIZE)
    axs[1].tick_params(labelsize=TICK_SIZE+2)

    # Robust shared y-lims, with extra headroom above the highest bracket
    all_vals = np.r_[y1, o1, y6, o6].astype(float)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        lo_full, hi_full = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        p1, p99 = np.nanpercentile(all_vals, [1, 99])
        lo = min(lo_full, p1)
        hi = max(hi_full, p99)
        span = max(1.0, hi - lo)
        pad_bottom = 0.14 * span
        pad_top    = 0.25 * span  # generous top pad to keep brackets away from titles
        hi_needed  = max(hi + pad_top, top0 + 0.25*span, top1 + 0.25*span)
        for ax in axs:
            ax.set_ylim(lo - pad_bottom, hi_needed)
            ax.margins(y=0.02)

    # More top space so suptitle never crowds stars
    fig.subplots_adjust(top=0.86)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# 1) Directed [1,3]
out_directed = os.path.join(SAVE_DIR, "fig_groupdiff_directed_A_h1h6.png")
figure_group_vs_group_two_panels(
    par_clean, "A_h1", "A_h6",
    ylabel="information bonus (points)",
    suptitle="directed [1 3] — young vs old",
    out_path=out_directed,
    color_h1=COLOR_H1, color_h6=COLOR_H6
)

# 2) Random [1,3]
out_random13 = os.path.join(SAVE_DIR, "fig_groupdiff_random_13_sigma_h1h6.png")
figure_group_vs_group_two_panels(
    par_clean, "sigma_unequal_h1", "sigma_unequal_h6",
    ylabel="decision noise (points)",
    suptitle="random [1 3] — young vs old",
    out_path=out_random13,
    color_h1=COLOR_H1, color_h6=COLOR_H6
)

# 3) Random [2,2]
out_random22 = os.path.join(SAVE_DIR, "fig_groupdiff_random_22_sigma_h1h6.png")
figure_group_vs_group_two_panels(
    par_clean, "sigma_equal_h1", "sigma_equal_h6",
    ylabel="decision noise (points)",
    suptitle="random [2 2] — young vs old",
    out_path=out_random22,
    color_h1=COLOR_H1, color_h6=COLOR_H6
)

print("Saved:")
print("-", out_directed)
print("-", out_random13)
print("-", out_random22)

