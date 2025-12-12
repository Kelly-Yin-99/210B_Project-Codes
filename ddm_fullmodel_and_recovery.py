import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats


CSV_BEHAV = "/Users/changyin/PyCharmMiscProject/outputs/behavior_clean_final_p001.csv"
CSV_PARAMS = "/Users/changyin/PyCharmMiscProject/fig6_ddm/ddm_params_per_subject.csv"
SAVE_DIR = "/Users/changyin/PyCharmMiscProject/fig6_ddm"
os.makedirs(SAVE_DIR, exist_ok=True)

RT_MIN, RT_MAX = 0.1, 3.0
MIN_TRIALS = 15
DT = 0.002
MAX_T = 3.0


def col(df, name):
    return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)

def observed_means_first4(df):
    R = np.column_stack([col(df, f"r{i}") for i in range(1, 5)])
    C = np.column_stack([col(df, f"c{i}") for i in range(1, 5)])
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

    k = np.arange(-K, K + 1, dtype=float)[:, None]
    ww = w_valid[None, :] + 2.0 * k

    num = (a_valid[None, :] * ww - v_valid[None, :] * tt[None, :]) ** 2
    den = 2.0 * tt[None, :]
    expo = np.exp(-num / den)

    coef = a_valid / (np.sqrt(2.0 * np.pi) * tt ** 1.5)
    s = np.sum(ww * expo, axis=0)

    pdf_valid = coef * s
    pdf_valid = np.maximum(pdf_valid, eps)

    out[valid] = pdf_valid
    return out


def ddm_nll(params, dR, dI, choice, rt):
    (c0_mu, cR_mu, cI_mu,
     c0_be, cR_be, cI_be,
     c0_al, cR_al, cI_al,
     log_T0) = params

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

    mask_L = (ch == 1)
    mask_R = (ch == 2)

    if np.any(mask_L):
        pdf_L = wfpt_pdf(rt_use[mask_L], mu[mask_L], beta[mask_L], w[mask_L], T0)
        if (pdf_L <= 0).any():
            return 1e9
    else:
        pdf_L = np.array([], float)

    if np.any(mask_R):
        pdf_R = wfpt_pdf(rt_use[mask_R], -mu[mask_R], beta[mask_R],
                         1.0 - w[mask_R], T0)
        if (pdf_R <= 0).any():
            return 1e9
    else:
        pdf_R = np.array([], float)

    return - (np.log(pdf_L).sum() + np.log(pdf_R).sum())



def simulate_ddm_trial(mu, a, z, T0, dt=DT, max_t=MAX_T):
    """
    Simulate one DDM trial.
    Upper bound at a, lower at 0, start at z.
    Returns: (choice:1/2, rt)
    """
    x = z
    t = 0.0
    while t < max_t:
        x += mu * dt + np.random.normal(0.0, np.sqrt(dt))
        t += dt
        if x >= a:
            return 1, T0 + t
        if x <= 0:
            return 2, T0 + t
    # bailout: choose by side of midline
    choice = 1 if x > a / 2 else 2
    return choice, T0 + max_t

def simulate_dataset(design_df, par_df):

    df = design_df.copy()
    df["c5_sim"] = np.nan
    df["rt5_sim"] = np.nan

    # pre-index parameters
    par_df = par_df.set_index("subjectID")

    for idx, row in df.iterrows():
        sid = str(row["subjectID"] if "subjectID" in row else row["pid"])
        if sid not in par_df.index:
            continue
        gl = int(row["gameLength"])

        tag = "h1" if gl == 5 else "h6" if gl == 10 else None
        if tag is None:
            continue

        # get ΔR, ΔI for this game
        dR = row["dR"]
        dI = row["dI"]

        # pull parameter vector
        try:
            p = par_df.loc[sid]
        except KeyError:
            continue

        key = lambda base: p.get(f"{base}_{tag}", np.nan)

        (c0_mu, cR_mu, cI_mu,
         c0_be, cR_be, cI_be,
         c0_al, cR_al, cI_al,
         T0) = (
            key("mu_c0"), key("mu_cR"), key("mu_cI"),
            key("be_c0"), key("be_cR"), key("be_cI"),
            key("al_c0"), key("al_cR"), key("al_cI"),
            key("T0")
        )

        if np.isnan(T0):
            continue

        mu = c0_mu + cR_mu * dR + cI_mu * dI

        beta_lin = c0_be + cR_be * dR + cI_be * dI
        a = np.exp(beta_lin)

        alpha_lin = c0_al + cR_al * dR + cI_al * dI
        alpha = 2.0 * sigmoid(alpha_lin) - 1.0
        z = (alpha + 1.0) / 2.0 * a

        # sanity: clip
        a = max(a, 1e-3)
        z = min(max(z, 1e-4), a - 1e-4)

        ch_sim, rt_sim = simulate_ddm_trial(mu, a, z, T0)
        df.at[idx, "c5_sim"] = ch_sim
        df.at[idx, "rt5_sim"] = rt_sim

    return df


beh = pd.read_csv(CSV_BEHAV)

subj = (beh["subjectID"] if "subjectID" in beh.columns else beh["pid"]).astype(str)
beh["subjectID"] = subj

gl = col(beh, "gameLength")
c5 = col(beh, "c5")
rt5 = col(beh, "rt5")

RL, RR, nL, nR = observed_means_first4(beh)
dR = RL - RR
dI = (nL - nR) / 2.0

design = pd.DataFrame({
    "subjectID": subj,
    "gameLength": gl,
    "dR": dR,
    "dI": dI,
})

# only rows where original model was fit (first free, valid stuff)
mask_design = np.isfinite(dR) & np.isfinite(dI) & np.isin(gl, [5, 10])
design = design[mask_design].reset_index(drop=True)


par_true = pd.read_csv(CSV_PARAMS)

sim = simulate_dataset(design, par_true)

rows_rec = []
unique_subj = par_true["subjectID"].astype(str).unique()

for sid in unique_subj:
    m_subj = (sim["subjectID"].astype(str) == sid)
    row = {"subjectID": sid}

    for horizon, tag in [(5, "h1"), (10, "h6")]:
        m = m_subj & (sim["gameLength"] == horizon) & np.isfinite(sim["dR"]) & np.isfinite(sim["dI"])
        # use simulated choice/rt
        ch = sim.loc[m, "c5_sim"].values.astype(float)
        rt = sim.loc[m, "rt5_sim"].values.astype(float)
        dR_h = sim.loc[m, "dR"].values
        dI_h = sim.loc[m, "dI"].values

        valid = (rt > RT_MIN) & (rt < RT_MAX) & np.isin(ch, [1, 2])
        if valid.sum() < MIN_TRIALS:
            for base in ["mu_c0","mu_cR","mu_cI",
                         "be_c0","be_cR","be_cI",
                         "al_c0","al_cR","al_cI",
                         "T0"]:
                row[f"{base}_{tag}"] = np.nan
            continue

        ch = ch[valid].astype(int)
        rt = rt[valid]
        dR_use = dR_h[valid]
        dI_use = dI_h[valid]

        x0 = np.array([
            0.0, 0.0, 0.0,
            np.log(1.0), 0.0, 0.0,
            0.0, 0.0, 0.0,
            np.log(np.exp(0.3)-1.0)
        ])

        bnds = [(-5,5), (-0.05,0.05), (-0.05,0.05),
                (-2,4), (-0.2,0.2), (-0.2,0.2),
                (-2,2), (-0.2,0.2), (-0.2,0.2),
                (np.log(1e-3), np.log(1.0))]

        res = minimize(ddm_nll, x0,
                       args=(dR_use, dI_use, ch, rt),
                       method="L-BFGS-B",
                       bounds=bnds)

        if not res.success:
            for base in ["mu_c0","mu_cR","mu_cI",
                         "be_c0","be_cR","be_cI",
                         "al_c0","al_cR","al_cI",
                         "T0"]:
                row[f"{base}_{tag}"] = np.nan
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

    rows_rec.append(row)

par_rec = pd.DataFrame(rows_rec)
par_rec_path = os.path.join(SAVE_DIR, "ddm_params_recovered.csv")
par_rec.to_csv(par_rec_path, index=False)


def collect_true_rec(base):
    true_vals = []
    rec_vals = []
    for _, row in par_true.iterrows():
        sid = str(row["subjectID"])
        r = par_rec[par_rec["subjectID"] == sid]
        if r.empty:
            continue
        r = r.iloc[0]
        for tag in ["h1","h6"]:
            t = row.get(f"{base}_{tag}", np.nan)
            v = r.get(f"{base}_{tag}", np.nan)
            if np.isfinite(t) and np.isfinite(v):
                true_vals.append(t)
                rec_vals.append(v)
    return np.array(true_vals), np.array(rec_vals)

def add_panel(ax, base, label):
    t, r = collect_true_rec(base)
    ax.scatter(t, r, s=25, alpha=0.7)
    if t.size > 0:
        lo = min(t.min(), r.min())
        hi = max(t.max(), r.max())
        if lo == hi:
            lo -= 0.1
            hi += 0.1
        pad = 0.1 * (hi - lo)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", lw=1)
        ax.set_xlim(lo-pad, hi+pad)
        ax.set_ylim(lo-pad, hi+pad)

        r_val, p_val = stats.pearsonr(t, r) if t.size > 1 else (np.nan, np.nan)
        rmse = np.sqrt(np.mean((t - r) ** 2)) if t.size > 0 else np.nan
        ax.text(0.05, 0.95,
                f"r = {r_val:.2f}\nRMSE = {rmse:.2f}",
                transform=ax.transAxes,
                ha="left", va="top", fontsize=9)
    ax.set_xlabel(f"true {label}")
    ax.set_ylabel(f"recovered {label}")

fig, axs = plt.subplots(4, 3, figsize=(10, 12))
axs = axs.ravel()

panels = [
    ("mu_c0", r"$c_0^\mu$ (baseline μ)"),
    ("mu_cR", r"$c_R^\mu$ (ΔR effect μ)"),
    ("mu_cI", r"$c_I^\mu$ (ΔI effect μ)"),
    ("be_c0", r"$c_0^\beta$ (baseline β)"),
    ("be_cR", r"$c_R^\beta$ (ΔR effect β)"),
    ("be_cI", r"$c_I^\beta$ (ΔI effect β)"),
    ("al_c0", r"$c_0^\alpha$ (baseline α)"),
    ("al_cR", r"$c_R^\alpha$ (ΔR effect α)"),
    ("al_cI", r"$c_I^\alpha$ (ΔI effect α)"),
    ("T0",    r"$T_0$"),
]

for i, (base, lab) in enumerate(panels):
    add_panel(axs[i], base, lab)

# hide unused axes if any
for j in range(len(panels), len(axs)):
    axs[j].axis("off")

fig.suptitle("DDM parameter recovery: true vs recovered", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig_path = os.path.join(SAVE_DIR, "ddm_param_recovery.png")
fig.savefig(fig_path, dpi=300)

print("Saved:")
print("-", par_rec_path)
print("-", fig_path)
