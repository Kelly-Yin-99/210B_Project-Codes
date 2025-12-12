import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats


CSV = "/Users/changyin/PyCharmMiscProject/outputs/behavior_clean_final_p001.csv"

SAVE_DIR = "/Users/changyin/PyCharmMiscProject/fig6_ddm"
os.makedirs(SAVE_DIR, exist_ok=True)


SAVE_DIR_HIER = SAVE_DIR + "_hier"
os.makedirs(SAVE_DIR_HIER, exist_ok=True)

RT_MIN, RT_MAX = 0.1, 3.0
MIN_TRIALS = 15

YOUNG = (18, 25)
OLD = (65, 80)

# Colors
COLOR_H1 = "#1f77b4"
COLOR_H6 = "#ae0b05"           # horizon 6 = red
GREY = "#b3b3b3"
HILITE_BETWEEN = (1.0, 0.96, 0.90)  # light orange
HILITE_WITHIN = (1.0, 0.92, 0.96)   # light pink

TITLE_SIZE = 15
AXIS_SIZE = 13
TICK_SIZE = 11

# ------------------ BASIC HELPERS ------------------
def col(df, name):
    return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)

def pick_age_col(df):
    return "Age" if "Age" in df.columns else "age"

def observed_means_first4(df):

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
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return ""

# ------------------ WIENER FPT PDF ------------------
def wfpt_pdf(t, v, a, w, t0, eps=1e-12, K_max=50):

    t = np.asarray(t, float)
    v = np.asarray(v, float)
    a = np.asarray(a, float)
    w = np.asarray(w, float)

    t, v, a, w = np.broadcast_arrays(t, v, a, w)
    out = np.zeros_like(t, dtype=float)

    valid = t > t0
    if not np.any(valid):
        return out

    tt = t[valid] - t0
    v_valid = v[valid]
    a_valid = np.clip(a[valid], 1e-4, None)
    w_valid = np.clip(w[valid], 1e-4, 1.0 - 1e-4)

    min_tt = float(max(np.min(tt), 1e-4))
    max_a = float(np.max(a_valid))
    K = int(np.ceil(np.sqrt(max_a ** 2 / (2.0 * np.pi * min_tt))))
    K = max(5, min(K, K_max))

    k = np.arange(-K, K + 1, dtype=float)[:, None]  # (2K+1, 1)
    ww = w_valid[None, :] + 2.0 * k                # (2K+1, n_valid)

    num = (a_valid[None, :] * ww - v_valid[None, :] * tt[None, :]) ** 2
    den = 2.0 * tt[None, :]
    expo = np.exp(-num / den)

    coef = a_valid / (np.sqrt(2.0 * np.pi) * tt ** 1.5)
    s = np.sum(ww * expo, axis=0)

    pdf_valid = coef * s
    pdf_valid = np.maximum(pdf_valid, eps)

    out[valid] = pdf_valid
    return out

# ------------------ DDM NEGATIVE LOG-LIKELIHOOD ------------------
def ddm_nll(params, dR, dI, choice, rt):
    """
    NLL for single subject × horizon.

    params = [c0_mu, cR_mu, cI_mu,
              c0_be, cR_be, cI_be,
              c0_al, cR_al, cI_al,
              log_T0]
    """
    (c0_mu, cR_mu, cI_mu,
     c0_be, cR_be, cI_be,
     c0_al, cR_al, cI_al,
     log_T0) = params

    # trial-wise parameters
    mu = c0_mu + cR_mu * dR + cI_mu * dI

    beta_lin = c0_be + cR_be * dR + cI_be * dI
    beta = np.exp(beta_lin)

    alpha_lin = c0_al + cR_al * dR + cI_al * dI
    alpha = 2.0 * sigmoid(alpha_lin) - 1.0
    z = (alpha + 1.0) / 2.0 * beta

    T0 = np.log1p(np.exp(log_T0))

    valid = (rt > RT_MIN) & (rt < RT_MAX) & np.isfinite(mu) & np.isfinite(beta) & np.isfinite(z)
    if valid.sum() < MIN_TRIALS:
        return 1e9

    mu = mu[valid]
    beta = beta[valid]
    z = z[valid]
    rt_use = rt[valid]
    ch = choice[valid]

    w = z / beta

    # left choices
    mask_L = (ch == 1)
    if np.any(mask_L):
        pdf_L = wfpt_pdf(rt_use[mask_L], mu[mask_L], beta[mask_L], w[mask_L], T0)
    else:
        pdf_L = np.array([], float)

    # right choices (symmetry)
    mask_R = (ch == 2)
    if np.any(mask_R):
        pdf_R = wfpt_pdf(rt_use[mask_R], -mu[mask_R], beta[mask_R],
                         1.0 - w[mask_R], T0)
    else:
        pdf_R = np.array([], float)

    if (pdf_L <= 0).any() or (pdf_R <= 0).any():
        return 1e9

    return - (np.log(pdf_L).sum() + np.log(pdf_R).sum())

# ============================================================
# 1. LOAD DATA AND FIT PER SUBJECT × HORIZON (AS BEFORE)
# ============================================================
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
dI = (nL - nR) / 2.0
choice = c5

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
        m = (m_subj &
             (gl == horizon) &
             valid5 &
             np.isfinite(dR) & np.isfinite(dI) &
             np.isin(choice, [1, 2]))

        # Not enough trials → all NaN for this (sid, horizon)
        if m.sum() < MIN_TRIALS:
            for name in ["mu_c0", "mu_cR", "mu_cI",
                         "be_c0", "be_cR", "be_cI",
                         "al_c0", "al_cR", "al_cI",
                         "T0"]:
                row[f"{name}_{tag}"] = np.nan
            continue

        dR_h = dR[m]
        dI_h = dI[m]
        ch_h = choice[m].astype(int)
        rt_h = rt5[m].astype(float)

        x0 = np.array([
            0.0, 0.0, 0.0,            # mu
            np.log(1.0), 0.0, 0.0,    # beta
            0.0, 0.0, 0.0,            # alpha
            np.log(np.exp(0.3) - 1.0) # T0 ~ 0.3
        ])

        bnds = [(-5, 5), (-0.05, 0.05), (-0.05, 0.05),
                (-2, 4), (-0.2, 0.2), (-0.2, 0.2),
                (-2, 2), (-0.2, 0.2), (-0.2, 0.2),
                (np.log(1e-3), np.log(1.0))]

        res = minimize(ddm_nll, x0,
                       args=(dR_h, dI_h, ch_h, rt_h),
                       method="L-BFGS-B",
                       bounds=bnds)

        if (not res.success) or np.isinf(res.fun) or np.isnan(res.fun):
            # retry with jitter
            res = minimize(ddm_nll,
                           x0 + np.random.normal(0, 0.1, size=x0.size),
                           args=(dR_h, dI_h, ch_h, rt_h),
                           method="L-BFGS-B",
                           bounds=bnds)

        if (not res.success) or np.isinf(res.fun) or np.isnan(res.fun):
            for name in ["mu_c0", "mu_cR", "mu_cI",
                         "be_c0", "be_cR", "be_cI",
                         "al_c0", "al_cR", "al_cI",
                         "T0"]:
                row[f"{name}_{tag}"] = np.nan
        else:
            (mu_c0, mu_cR, mu_cI,
             be_c0, be_cR, be_cI,
             al_c0, al_cR, al_cI,
             log_T0) = res.x
            T0 = np.log1p(np.exp(log_T0))

            row[f"mu_c0_{tag}"] = mu_c0
            row[f"mu_cR_{tag}"] = mu_cR
            row[f"mu_cI_{tag}"] = mu_cI

            row[f"be_c0_{tag}"] = be_c0
            row[f"be_cR_{tag}"] = be_cR
            row[f"be_cI_{tag}"] = be_cI

            row[f"al_c0_{tag}"] = al_c0
            row[f"al_cR_{tag}"] = al_cR
            row[f"al_cI_{tag}"] = al_cI

            row[f"T0_{tag}"] = T0

    rows.append(row)

par = pd.DataFrame(rows)
par_path = os.path.join(SAVE_DIR, "ddm_params_per_subject.csv")
par.to_csv(par_path, index=False)

print("Saved original (non-shrunk) params to:", par_path)


param_cols = [c for c in par.columns
              if any(c.startswith(pfx) for pfx in ["mu_", "be_", "al_", "T0"])
              and c not in ["mu", "be", "al"]]

def shrink_1d(x, strength=0.2):

    x = np.asarray(x, float)
    mask = np.isfinite(x)
    if mask.sum() <= 1:
        return x

    m = x[mask].mean()
    s2 = x[mask].var(ddof=1)

    # heuristic "noise" term: scaled by strength
    # when s2 is small, w -> small (strong shrink). when large, w -> ~1 (weak).
    noise = strength
    w = s2 / (s2 + noise) if s2 > 0 else 0.0
    w = np.clip(w, 0.0, 1.0)

    x_shrunk = x.copy()
    x_shrunk[mask] = m + w * (x[mask] - m)
    return x_shrunk

par_shrink = par.copy()

# apply shrinkage separately by group and horizon for each parameter column
for col_name in param_cols:
    # detect horizon tag
    if col_name.endswith("_h1"):
        tag = "_h1"
    elif col_name.endswith("_h6"):
        tag = "_h6"
    else:
        # not a horizon-specific column; skip
        continue

    base = col_name.replace(tag, "")

    for g in ["young", "old"]:
        m = (par["group"] == g)
        if m.sum() <= 1:
            continue

        x = par.loc[m, col_name].values
        x_shrunk = shrink_1d(x, strength=0.2)  # you can tune strength
        par_shrink.loc[m, col_name] = x_shrunk

# save shrunk parameters
par_shrink_path = os.path.join(SAVE_DIR_HIER, "ddm_params_shrunk.csv")
par_shrink.to_csv(par_shrink_path, index=False)
print("Saved shrunk (hierarchical-style) params to:", par_shrink_path)

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
    if y + h * 1.6 >= ymax:
        y = ymax - h * 1.8
    ax.plot([x1, x1, x2, x2],
            [y, y + h, y + h, y],
            color="k", lw=1.3)
    ax.text((x1 + x2) / 2.0, y + h * 1.25, s,
            ha="center", va="bottom",
            fontsize=TICK_SIZE, fontweight="bold")
    return True

def plot_param(ax, y1, y6, o1, o6, coef_label):
    my = np.isfinite(y1) & np.isfinite(y6)
    mo = np.isfinite(o1) & np.isfinite(o6)

    # grey within-subject lines
    for i in np.where(my)[0]:
        ax.plot([1, 2], [y1[i], y6[i]], color=GREY, alpha=0.5, lw=1)
    for i in np.where(mo)[0]:
        ax.plot([3, 4], [o1[i], o6[i]], color=GREY, alpha=0.5, lw=1)

    ax.scatter(np.full(my.sum(), 1), y1[my], color=COLOR_H1, zorder=3)
    ax.scatter(np.full(my.sum(), 2), y6[my], color=COLOR_H6, zorder=3)
    ax.scatter(np.full(mo.sum(), 3), o1[mo], color=COLOR_H1, zorder=3)
    ax.scatter(np.full(mo.sum(), 4), o6[mo], color=COLOR_H6, zorder=3)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["young\n1", "young\n6", "old\n1", "old\n6"],
                       fontsize=TICK_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    allv = np.r_[y1[my], y6[my], o1[mo], o6[mo]]
    if allv.size:
        lo, hi = np.nanpercentile(allv, [2, 98])
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        pad = (hi - lo) * 0.4
        ax.set_ylim(lo - pad, hi + pad)

    ymin, ymax = ax.get_ylim()
    H = ymax - ymin

    # parameter label
    ax.text(0.5, 0.98, coef_label,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=AXIS_SIZE, fontweight="bold")

    any_within = False
    any_between = False

    # within-group
    if my.sum() >= 3:
        t, p = stats.ttest_rel(y1[my], y6[my])
        if sig_bar(ax, 1, 2, ymax - 0.26 * H, p):
            any_within = True
    if mo.sum() >= 3:
        t, p = stats.ttest_rel(o1[mo], o6[mo])
        if sig_bar(ax, 3, 4, ymax - 0.26 * H, p):
            any_within = True

    # between-group
    if my.sum() >= 3 and mo.sum() >= 3:
        t, p = stats.ttest_ind(y1[my], o1[mo], equal_var=False)
        if sig_bar(ax, 1, 3, ymax - 0.46 * H, p):
            any_between = True
        t, p = stats.ttest_ind(y6[my], o6[mo], equal_var=False)
        if sig_bar(ax, 2, 4, ymax - 0.66 * H, p):
            any_between = True

    # shading
    if any_between:
        ax.set_facecolor(HILITE_BETWEEN)
    elif any_within:
        ax.set_facecolor(HILITE_WITHIN)
    else:
        ax.set_facecolor("white")


fig, axs = plt.subplots(4, 3, figsize=(11, 11))
axs = axs.reshape(4, 3)

# Row 1: μ
for j, base in enumerate(["mu_c0", "mu_cR", "mu_cI"]):
    y1, y6, o1, o6 = get_groups(par_shrink, base)
    label = [r"$c_0^\mu$", r"$c_R^\mu$", r"$c_I^\mu$"][j]
    plot_param(axs[0, j], y1, y6, o1, o6, label)

# Row 2: β
for j, base in enumerate(["be_c0", "be_cR", "be_cI"]):
    y1, y6, o1, o6 = get_groups(par_shrink, base)
    label = [r"$c_0^\beta$", r"$c_R^\beta$", r"$c_I^\beta$"][j]
    plot_param(axs[1, j], y1, y6, o1, o6, label)

# Row 3: α
for j, base in enumerate(["al_c0", "al_cR", "al_cI"]):
    y1, y6, o1, o6 = get_groups(par_shrink, base)
    label = [r"$c_0^\alpha$", r"$c_R^\alpha$", r"$c_I^\alpha$"][j]
    plot_param(axs[2, j], y1, y6, o1, o6, label)

# Row 4: T0
y1, y6, o1, o6 = get_groups(par_shrink, "T0")
plot_param(axs[3, 0], y1, y6, o1, o6, r"$T_0$")
axs[3, 1].axis("off")
axs[3, 2].axis("off")

# column headers
col_titles = ["baseline", "effect of ΔR", "effect of ΔI"]
for j, title in enumerate(col_titles):
    pos = axs[0, j].get_position()
    x_center = pos.x0 + pos.width / 2.0
    plt.gcf().text(x_center, 0.98, title,
                   ha="center", va="top",
                   fontsize=TITLE_SIZE + 1, fontweight="bold")

# row labels
row_labels = [r"drift rate $\mu$", r"threshold $\beta$",
              r"bias $\alpha$", r"non-decision time $T_0$"]
for i, label in enumerate(row_labels):
    pos = axs[i, 0].get_position()
    y_center = pos.y0 + pos.height / 2.0
    plt.gcf().text(0.03, y_center, label,
                   ha="right", va="center",
                   fontsize=TITLE_SIZE + 1, fontweight="bold", rotation=90)

plt.subplots_adjust(left=0.09, right=0.99, top=0.94, bottom=0.06,
                    wspace=0.35, hspace=0.55)

fig_path = os.path.join(SAVE_DIR_HIER, "figure6_DDM_young_old_hier.png")
fig.savefig(fig_path, dpi=300)

print("Saved shrunk figure to:", fig_path)
