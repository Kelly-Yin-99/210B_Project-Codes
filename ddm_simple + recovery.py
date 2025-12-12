# ddm_simple_fit_plot.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

# ------------------ CONFIG ------------------
CSV = "/Users/changyin/PyCharmMiscProject/outputs/behavior_clean_final_p001.csv"
SAVE_DIR = "/Users/changyin/PyCharmMiscProject/fig6_ddm_simple"
os.makedirs(SAVE_DIR, exist_ok=True)

RT_MIN, RT_MAX = 0.1, 3.0
MIN_TRIALS = 15  # per subj × horizon minimum

MU_MIN = 0.3     # minimum |baseline drift mu_c0| to keep
BETA_MIN = 0.5

YOUNG = (18, 25)
OLD = (65, 80)

# Colors & style
COLOR_H1 = "#1f77b4"           # horizon 1 = blue
COLOR_H6 = "#ae0b05"           # horizon 6 = red
GREY = "#b3b3b3"
HILITE_BETWEEN = (1.0, 0.96, 0.90)  # light orange (between-group sig)
HILITE_WITHIN = (1.0, 0.92, 0.96)   # light pink (within-group only)

TITLE_SIZE = 15
AXIS_SIZE = 13
TICK_SIZE = 11

# ------------------ HELPERS ------------------
def col(df, name):
    return pd.to_numeric(df.get(name, np.nan), errors="coerce")

def pick_age_col(df):
    return "Age" if "Age" in df.columns else "age"

def observed_means_first4(df):
    """Observed means for left/right on instructed trials 1–4, conditioned on choices."""
    R = np.column_stack([col(df, f"r{i}") for i in range(1, 5)])
    C = np.column_stack([col(df, f"c{i}") for i in range(1, 5)])  # 1=left,2=right
    C = np.where(np.isnan(C), 0, C)

    Lmask, Rmask = (C == 1), (C == 2)
    nL = Lmask.sum(1).astype(float)
    nR = Rmask.sum(1).astype(float)

    sumL = (R * Lmask).sum(1)
    sumR = (R * Rmask).sum(1)

    with np.errstate(divide="ignore", invalid="ignore"):
        RL = np.where(nL > 0, sumL / nL, np.nan)
        RR = np.where(nR > 0, sumR / nR, np.nan)

    return RL, RR, nL, nR

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def stars(p):
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return ""

# ------------------ WFPT PDF ------------------
def wfpt_pdf(t, v, a, w, t0, eps=1e-12, K_max=50):
    """
    Approximate pdf of hitting upper bound of Wiener process.
    """
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    a = np.asarray(a, float)
    w = np.asarray(w, float)

    t, v, a, w = np.broadcast_arrays(t, v, a, w)
    out = np.zeros_like(t)

    valid = t > t0
    if not np.any(valid):
        return out

    tt = t[valid] - t0
    v_val = v[valid]
    a_val = np.clip(a[valid], 1e-4, None)
    w_val = np.clip(w[valid], 1e-4, 1 - 1e-4)

    min_tt = float(max(np.min(tt), 1e-4))
    max_a = float(np.max(a_val))
    K = int(np.ceil(np.sqrt(max_a**2 / (2.0 * np.pi * min_tt))))
    K = max(5, min(K, K_max))

    k = np.arange(-K, K+1, dtype=float)[:, None]
    ww = w_val[None, :] + 2.0 * k

    num = (a_val[None, :] * ww - v_val[None, :] * tt[None, :])**2
    den = 2.0 * tt[None, :]
    expo = np.exp(-num / den)

    coef = a_val / (np.sqrt(2.0 * np.pi) * tt**1.5)
    s = np.sum(ww * expo, axis=0)

    pdf_val = coef * s
    pdf_val = np.maximum(pdf_val, eps)
    out[valid] = pdf_val
    return out

# ------------------ SIMPLE DDM NLL ------------------
def ddm_nll_simple(params, dR, choice, rt):
    """
    Simpler DDM:
      mu = c0_mu                      (constant)
      a  = exp(c0_be + cR_be * dR)    (depends on ΔR)
      w  = 0.5                        (no bias)
      T0 = softplus(log_T0)

    params = [c0_mu, c0_be, cR_be, log_T0]
    """
    c0_mu, c0_be, cR_be, log_T0 = params

    T0 = np.log1p(np.exp(log_T0))
    mu = np.full_like(dR, c0_mu, dtype=float)

    beta_lin = c0_be + cR_be * dR
    a = np.exp(beta_lin)

    valid = (
        (rt > RT_MIN) & (rt < RT_MAX) &
        np.isfinite(mu) & np.isfinite(a) &
        np.isfinite(dR) & np.isin(choice, [1, 2])
    )

    if valid.sum() < MIN_TRIALS:
        return 1e9

    mu = mu[valid]
    a = a[valid]
    rt_use = rt[valid]
    ch = choice[valid].astype(int)

    a = np.clip(a, 1e-4, 50.0)
    w = np.full_like(a, 0.5)

    mask_L = (ch == 1)
    mask_R = (ch == 2)

    if np.any(mask_L):
        pdf_L = wfpt_pdf(rt_use[mask_L], mu[mask_L], a[mask_L], w[mask_L], T0)
        if (pdf_L <= 0).any():
            return 1e9
    else:
        pdf_L = np.array([], float)

    if np.any(mask_R):
        pdf_R = wfpt_pdf(rt_use[mask_R], -mu[mask_R], a[mask_R], 1.0 - w[mask_R], T0)
        if (pdf_R <= 0).any():
            return 1e9
    else:
        pdf_R = np.array([], float)

    nll = - (np.log(pdf_L).sum() + np.log(pdf_R).sum())
    return nll

# ------------------ LOAD DATA ------------------
df = pd.read_csv(CSV)

subj = (df["subjectID"] if "subjectID" in df.columns else df["pid"]).astype(str).values
age_col = pick_age_col(df)
age = col(df, age_col).values

gl = col(df, "gameLength").values
c5 = col(df, "c5").values
rt5 = col(df, "rt5").values

valid5 = (rt5 > RT_MIN) & (rt5 < RT_MAX)

RL, RR, nL, nR = observed_means_first4(df)
dR = RL - RR

choice = c5

# ------------------ FIT SIMPLE MODEL PER SUBJECT × HORIZON ------------------
rows = []

for sid in np.unique(subj):
    m_subj = (subj == sid)
    this_age = age[m_subj][0] if np.any(m_subj) else np.nan

    if YOUNG[0] <= this_age <= YOUNG[1]:
        group = "young"
    elif OLD[0] <= this_age <= OLD[1]:
        group = "old"
    else:
        group = "other"

    row = {"subjectID": sid, "age": this_age, "group": group}

    for horizon, tag in [(5, "h1"), (10, "h6")]:
        m = (
            m_subj &
            (gl == horizon) &
            valid5 &
            np.isfinite(dR) &
            np.isin(choice, [1, 2])
        )

        if m.sum() < MIN_TRIALS:
            for base in ["mu_c0", "be_c0", "be_cR", "T0"]:
                row[f"{base}_{tag}"] = np.nan
            continue

        dR_h = dR[m]
        ch_h = choice[m].astype(int)
        rt_h = rt5[m].astype(float)

        # initial guess
        x0 = np.array([
            0.0,                  # c0_mu
            np.log(1.0),          # c0_be (log boundary ~ 1)
            0.0,                  # cR_be (no ΔR effect start)
            np.log(np.exp(0.3) - 1.0)  # log_T0 for T0~0.3
        ])

        bnds = [
            (-5, 5),              # c0_mu
            (-2, 4),              # c0_be
            (-0.2, 0.2),          # cR_be
            (np.log(1e-3), np.log(1.0))  # T0 in ~[0.001,1]s
        ]

        res = minimize(ddm_nll_simple, x0,
                       args=(dR_h, ch_h, rt_h),
                       method="L-BFGS-B", bounds=bnds)

        if (not res.success) or np.isinf(res.fun) or np.isnan(res.fun):
            # retry with small jitter
            res = minimize(ddm_nll_simple,
                           x0 + np.random.normal(0, 0.1, size=x0.size),
                           args=(dR_h, ch_h, rt_h),
                           method="L-BFGS-B", bounds=bnds)

        if (not res.success) or np.isinf(res.fun) or np.isnan(res.fun):
            for base in ["mu_c0", "be_c0", "be_cR", "T0"]:
                row[f"{base}_{tag}"] = np.nan
        else:
            c0_mu, c0_be, cR_be, log_T0 = res.x
            T0 = np.log1p(np.exp(log_T0))

            row[f"mu_c0_{tag}"] = c0_mu
            row[f"be_c0_{tag}"] = c0_be
            row[f"be_cR_{tag}"] = cR_be
            row[f"T0_{tag}"] = T0

    rows.append(row)

par = pd.DataFrame(rows)
par_path = os.path.join(SAVE_DIR, "ddm_params_simple_per_subject.csv")
par.to_csv(par_path, index=False)
print("Saved:", par_path)

# ------------------ PLOTTING HELPERS ------------------
def get_groups(par_df, base_name):
    y = par_df[par_df["group"] == "young"]
    o = par_df[par_df["group"] == "old"]
    return (y[f"{base_name}_h1"].values,
            y[f"{base_name}_h6"].values,
            o[f"{base_name}_h1"].values,
            o[f"{base_name}_h6"].values)

def sig_bar(ax, x1, x2, y, p, rel_h=0.03):
    s = stars(p)
    if s == "":
        return False
    ymin, ymax = ax.get_ylim()
    h = (ymax - ymin) * rel_h
    # keep inside axes
    if y + h * 1.8 > ymax:
        y = ymax - h * 2.0
    ax.plot([x1, x1, x2, x2],
            [y, y + h, y + h, y],
            color="k", lw=1.3)
    ax.text((x1 + x2) / 2.0, y + h * 1.25, s,
            ha="center", va="bottom",
            fontsize=TICK_SIZE, fontweight="bold")
    return True

def plot_param(ax, y1, y6, o1, o6, coef_label):
    """
    x positions:
      1: young_1 (blue)
      2: young_6 (red)
      3: old_1   (blue)
      4: old_6   (red)
    """
    my = np.isfinite(y1) & np.isfinite(y6)
    mo = np.isfinite(o1) & np.isfinite(o6)

    # lines within subjects
    for i in np.where(my)[0]:
        ax.plot([1, 2], [y1[i], y6[i]], color=GREY, alpha=0.5, lw=1)
    for i in np.where(mo)[0]:
        ax.plot([3, 4], [o1[i], o6[i]], color=GREY, alpha=0.5, lw=1)

    # points
    ax.scatter(np.full(my.sum(), 1), y1[my], color=COLOR_H1, zorder=3)
    ax.scatter(np.full(my.sum(), 2), y6[my], color=COLOR_H6, zorder=3)
    ax.scatter(np.full(mo.sum(), 3), o1[mo], color=COLOR_H1, zorder=3)
    ax.scatter(np.full(mo.sum(), 4), o6[mo], color=COLOR_H6, zorder=3)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["young\n1", "young\n6", "old\n1", "old\n6"],
                       fontsize=TICK_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    # y-lims with extra space for bars
    allv = np.r_[y1[my], y6[my], o1[mo], o6[mo]]
    if allv.size:
        lo, hi = np.nanpercentile(allv, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(allv) - 0.5, np.nanmax(allv) + 0.5
        pad = (hi - lo) * 0.35
        ax.set_ylim(lo - pad, hi + pad)

    ymin, ymax = ax.get_ylim()
    H = ymax - ymin

    # top label = parameter name
    ax.text(0.5, 0.98, coef_label,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=AXIS_SIZE, fontweight="bold")

    any_within = False
    any_between = False

    # within young
    if my.sum() >= 3:
        t, p = stats.ttest_rel(y1[my], y6[my])
        if sig_bar(ax, 1, 2, ymax - 0.26 * H, p):
            any_within = True

    # within old
    if mo.sum() >= 3:
        t, p = stats.ttest_rel(o1[mo], o6[mo])
        if sig_bar(ax, 3, 4, ymax - 0.26 * H, p):
            any_within = True

    # young1 vs old1
    if my.sum() >= 3 and mo.sum() >= 3:
        t, p = stats.ttest_ind(y1[my], o1[mo], equal_var=False)
        if sig_bar(ax, 1, 3, ymax - 0.46 * H, p):
            any_between = True

    # young6 vs old6
    if my.sum() >= 3 and mo.sum() >= 3:
        t, p = stats.ttest_ind(y6[my], o6[mo], equal_var=False)
        if sig_bar(ax, 2, 4, ymax - 0.66 * H, p):
            any_between = True

    # panel shading
    if any_between:
        ax.set_facecolor(HILITE_BETWEEN)
    elif any_within:
        ax.set_facecolor(HILITE_WITHIN)
    else:
        ax.set_facecolor("white")

# ------------------ PLOT: 2x2 SIMPLE PARAM GRID ------------------
fig, axs = plt.subplots(2, 2, figsize=(9, 8))
axs = axs.ravel()

# 1) drift rate baseline c0^mu
y1, y6, o1, o6 = get_groups(par, "mu_c0")
plot_param(axs[0], y1, y6, o1, o6, r"$c_0^\mu$ (drift baseline)")

# 2) boundary baseline c0^beta
y1, y6, o1, o6 = get_groups(par, "be_c0")
plot_param(axs[1], y1, y6, o1, o6, r"$c_0^\beta$ (threshold baseline)")

# 3) boundary ΔR effect c_R^beta
y1, y6, o1, o6 = get_groups(par, "be_cR")
plot_param(axs[2], y1, y6, o1, o6, r"$c_R^\beta$ (ΔR effect on threshold)")

# 4) non-decision time T0
y1, y6, o1, o6 = get_groups(par, "T0")
plot_param(axs[3], y1, y6, o1, o6, r"$T_0$ (non-decision time)")

# Row labels on left
row_labels = [r"", r""]
# (Grid is small; optional to add global title)
fig.tight_layout()
fig_path = os.path.join(SAVE_DIR, "figure6_DDM_simple_young_old.png")
fig.savefig(fig_path, dpi=300)
print("Saved:", fig_path)
