"""
================================================================================
The Incumbent Tax: A Stochastic Assessment of Pre-Election Clientelism and
Fiscal Insolvency in Philippine Local Government Units
--------------------------------------------------------------------------------
v10 — figure-correctness pass.

Fixes vs v9:
  • fig1: remove erroneous /1e6 division (_real columns are already PHP M)
  • fig2: shared y-axis, mean reference line, numeric span in caption
  • fig3: restricted to ENC ∈ [1.5, 4.0] to avoid mechanical boundaries
  • fig5: Anderson–Darling statistic in caption
  • table2: N (cycles) formatted as int (was displaying as concatenated digits)
================================================================================
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore", category=FutureWarning)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12,
                     "axes.titlesize": 13, "figure.dpi": 300})

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
OUTPUT_DIR = os.path.join(_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SIM = 10_000
N_YEARS = 10
INSOLVENCY_THRESHOLD = 0.05
REV_CORR_DEFAULT = 0.45
ELECTION_YEARS = [1992, 1995, 1998, 2001, 2004, 2007, 2010,
                  2013, 2016, 2019, 2022, 2025]
CPI_FALLBACK = {y: 100.0 for y in range(1992, 2026)}


# =============================================================================
# LOADERS
# =============================================================================
def load_fiscal_panel(panel_path):
    df = pd.read_csv(panel_path, low_memory=False)
    df["psgc10"] = (df["psgc10"].astype(str).str.strip()
                    .str.replace(r"\.0$", "", regex=True).str.zfill(10))
    df["lgu"] = df["lgu_name"].astype(str).str.strip().str.upper()
    df["lgutype"] = df["lgu_type"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")

    if "in_sre" in df.columns:
        df = df[df["in_sre"].astype(str).str.lower().isin(["true", "1", "yes"])]

    sum_cols = [c for c in [
        "total_local_sources", "nta_ira", "other_national_shares",
        "total_external_sources", "total_current_operating_income",
        "education", "health", "social_welfare", "total_social_services",
        "gps", "economic_services", "total_current_operating_exp",
        "total_capital_investment_exp", "total_non_operating_exp",
        "total_expenditures"] if c in df.columns]

    agg = {c: "sum" for c in sum_cols}
    for c in ["lgu", "lgutype", "province", "region", "income_class",
              "pop_2024", "poverty_pi", "cpi_2018"]:
        if c in df.columns:
            agg[c] = "first"

    df = df.groupby(["psgc10", "year"], as_index=False).agg(agg)
    df = df.rename(columns={
        "total_local_sources": "totlocsrc", "nta_ira": "ira",
        "total_expenditures": "totexp", "social_welfare": "pubwelf",
        "education": "educexp", "health": "healthexp", "gps": "govexp",
        "pop_2024": "population", "poverty_pi": "poverty"})

    for c in ["totlocsrc", "ira", "totexp", "pubwelf", "educexp",
              "healthexp", "govexp", "population", "poverty"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["totlocsrc", "ira", "totexp", "pubwelf"]:
        if c in df.columns:
            df = df[df[c].isna() | (df[c] <= 1e8)]

    for c in ["educexp", "healthexp", "pubwelf", "govexp"]:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)

    for c in ["totlocsrc", "ira", "totexp"]:
        if c in df.columns:
            df[f"{c}_raw"] = df[c]
            pos = df[c][df[c] > 0]
            if len(pos) > 10:
                q99 = np.exp(np.log(pos).quantile(0.99))
                df[c] = np.where(df[c] > q99, q99, df[c])

    if "cpi_2018" in df.columns:
        deflator = df["cpi_2018"] / 100.0
        print("[*] Deflation: cpi_2018 column present — using real 2018 PHP.")
    else:
        deflator = 1.0
        print("[!] Deflation: cpi_2018 column NOT present — treating as nominal.")

    for c in ["totlocsrc", "ira", "totexp", "pubwelf", "educexp",
              "healthexp", "govexp"]:
        if c in df.columns:
            df[f"{c}_real"] = df[c] / deflator

    pre_years = [y - 1 for y in ELECTION_YEARS]
    df["pbc_indicator"] = df["year"].isin(pre_years).astype(int)

    print(f"[*] Fiscal panel: {len(df):,} LGU-years, "
          f"{df['psgc10'].nunique():,} LGUs, "
          f"years {int(df['year'].min())}–{int(df['year'].max())}")
    print(df.groupby("lgutype")["psgc10"].nunique().to_string())
    return df


def load_election_panel(election_path, label):
    elec = pd.read_csv(election_path)
    elec["psgc10"] = (elec["psgc10"].astype(str).str.strip()
                      .str.replace(r"\.0$", "", regex=True).str.zfill(10))
    for c in ["elecyr", "pi", "ENC_lt", "n_candidates"]:
        if c in elec.columns:
            elec[c] = pd.to_numeric(elec[c], errors="coerce")

    elec = elec.dropna(subset=["psgc10", "elecyr"])
    elec = elec.drop_duplicates(subset=["psgc10", "elecyr"])

    if "lgutype" in elec.columns:
        elec["lgutype"] = elec["lgutype"].astype(str).str.strip().str.lower()
    else:
        elec["lgutype"] = "province"

    if "pi_incumbent" in elec.columns:
        elec["outcome"] = pd.to_numeric(elec["pi_incumbent"], errors="coerce")
        elec["outcome_source"] = "incumbent"
        print(f"[*] {label}: using pi_incumbent (N = {elec['outcome'].notna().sum():,})")
    else:
        elec["outcome"] = elec["pi"]
        elec["outcome_source"] = "winner"
        print(f"[*] {label}: pi_incumbent NOT found — using winner share")

    print(f"[*] {label} panel: {len(elec):,} LGU-cycles, "
          f"{elec['psgc10'].nunique():,} LGUs")
    print(elec.groupby("lgutype")["psgc10"].nunique().to_string())
    print(f"    cycles: {sorted(elec['elecyr'].astype(int).unique())}")
    return elec


def build_regression_panel(fiscal, elec):
    fp = fiscal.copy()
    fp["elecyr"] = fp["year"] + 1
    reg = elec.merge(fp, on=["psgc10", "elecyr"], how="inner",
                     suffixes=("", "_fiscal"))
    print(f"[*] Regression panel: {len(reg):,} LGU-cycles, "
          f"{reg['psgc10'].nunique()} LGUs, {reg['elecyr'].nunique()} cycles")
    return reg


# =============================================================================
# TWO-WAY FE (PanelOLS)
# =============================================================================
def estimate_elasticities(reg, level="province"):
    if level == "province":
        reg = reg[reg["lgutype"].str.contains("province", case=False, na=False)]
    elif level == "city":
        reg = reg[reg["lgutype"].str.contains("city", case=False, na=False)]
    elif level == "municipality":
        reg = reg[reg["lgutype"].str.contains("municipality", case=False, na=False)]

    if "n_candidates" in reg.columns:
        reg = reg[reg["n_candidates"].isna() | (reg["n_candidates"] > 1)]

    y_col = "outcome" if "outcome" in reg.columns else "pi"
    reg = reg[reg[y_col].notna() & (reg[y_col] > 0) & (reg[y_col] <= 1)].copy()

    needed = ["lgu", "elecyr", y_col, "pubwelf_real", "educexp_real",
              "healthexp_real", "ENC_lt"]
    needed = [c for c in needed if c in reg.columns]
    df = reg[needed].dropna(subset=[y_col, "pubwelf_real", "educexp_real",
                                    "healthexp_real", "ENC_lt", "lgu", "elecyr"])
    df = df[np.isfinite(df["ENC_lt"]) & (df["ENC_lt"] > 0)].copy()
    if len(df) < 30:
        print(f"[!] {level}: only {len(df)} usable rows — skipping.")
        return None

    df["log_pub"] = np.log(df["pubwelf_real"] + 1)
    df["log_edu"] = np.log(df["educexp_real"] + 1)
    df["log_hea"] = np.log(df["healthexp_real"] + 1)
    df["log_enc"] = np.log(df["ENC_lt"])

    means = {k: float(df[k].mean())
             for k in ["log_pub", "log_edu", "log_hea", "log_enc"]}
    mean_y = float(df[y_col].mean())

    panel = df.set_index(["lgu", "elecyr"])[[
        y_col, "log_pub", "log_edu", "log_hea", "log_enc"]]
    mod = PanelOLS(panel[y_col],
                   panel[["log_pub", "log_edu", "log_hea", "log_enc"]],
                   entity_effects=True, time_effects=True,
                   drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    b = {k: float(res.params.get(k, 0.0))
         for k in ["log_pub", "log_edu", "log_hea", "log_enc"]}
    ci = res.conf_int()
    ci_dict = {k: (float(ci.loc[k, "lower"]), float(ci.loc[k, "upper"]))
               for k in b if k in ci.index}

    intercept_adj = mean_y - sum(b[k] * means[k] for k in b)

    result = {
        "coefficients": b,
        "std_errors": {k: float(res.std_errors.get(k, np.nan)) for k in b},
        "p_values": {k: float(res.pvalues.get(k, np.nan)) for k in b},
        "ci_95": ci_dict,
        "n_obs": int(res.nobs),
        "n_lgu": int(df["lgu"].nunique()),
        "n_cycles": int(df["elecyr"].nunique()),
        "within_r2": float(res.rsquared_within),
        "f_stat": float(res.f_statistic.stat),
        "f_pval": float(res.f_statistic.pval),
        "df_resid": float(res.df_resid),
        "resid_std": float(np.std(res.resids, ddof=1)),
        "intercept": float(intercept_adj),
        "mean_outcome": mean_y,
        "mean_predictors": means,
        "level": level,
        "outcome_source": reg["outcome_source"].iloc[0]
            if "outcome_source" in reg.columns else "winner",
    }

    print(f"\n[*] Two-way FE ({level}) [{result['outcome_source']} outcome]:")
    print(f"    N = {result['n_obs']}, LGUs = {result['n_lgu']}, "
          f"cycles = {result['n_cycles']}, within-R² = {result['within_r2']:.4f}")
    print(f"    F = {result['f_stat']:.2f} (p = {result['f_pval']:.4f}), "
          f"df_resid = {result['df_resid']:.0f}")
    for k in ["log_pub", "log_edu", "log_hea", "log_enc"]:
        lo, hi = ci_dict.get(k, (np.nan, np.nan))
        print(f"    {k:9s}: b={b[k]:+.5f}  se={result['std_errors'][k]:.5f}  "
              f"p={result['p_values'][k]:.4f}  "
              f"95% CI [{lo:+.5f}, {hi:+.5f}]")
    return result


# =============================================================================
# CALIBRATION
# =============================================================================
def calibrate_lognormal(fiscal, level):
    if level == "province":
        sub = fiscal[fiscal["lgutype"].str.contains("province", case=False, na=False)]
    elif level == "city":
        sub = fiscal[fiscal["lgutype"].str.contains("city", case=False, na=False)]
    else:
        sub = fiscal[fiscal["lgutype"].str.contains("municipality", case=False, na=False)]

    loc_col = "totlocsrc_real" if "totlocsrc_real" in sub.columns else "totlocsrc"
    ira_col = "ira_real" if "ira_real" in sub.columns else "ira"

    loc = sub[loc_col].dropna()
    ira = sub[ira_col].dropna()
    joint = sub[[loc_col, ira_col]].dropna()

    mean_loc, std_loc = float(loc.mean()), float(loc.std())
    mean_ira, std_ira = float(ira.mean()), float(ira.std())
    rho = float(joint.corr().iloc[0, 1]) if len(joint) > 2 else REV_CORR_DEFAULT

    def to_lognormal(m, s):
        var = s ** 2
        sig2 = np.log(1 + var / (m ** 2))
        return np.log(m) - sig2 / 2, np.sqrt(sig2)

    mu_loc, sig_loc = to_lognormal(mean_loc, std_loc)
    mu_ira, sig_ira = to_lognormal(mean_ira, std_ira)

    cov = np.array([[sig_loc ** 2, rho * sig_loc * sig_ira],
                    [rho * sig_loc * sig_ira, sig_ira ** 2]])

    cal = {"mean_loc": mean_loc, "std_loc": std_loc,
           "mean_ira": mean_ira, "std_ira": std_ira,
           "rho": rho, "mu": np.array([mu_loc, mu_ira]), "Sigma": cov,
           "mu_loc": mu_loc, "sig_loc": sig_loc,
           "mu_ira": mu_ira, "sig_ira": sig_ira,
           "baseline_budget": mean_loc + mean_ira,
           "n_joint": int(len(joint)), "n_loc": int(len(loc)),
           "n_ira": int(len(ira)), "level": level}

    print(f"\n[*] Calibration ({level}) — N_joint={cal['n_joint']:,}")
    print(f"    E[Loc]={mean_loc:,.1f}  SD={std_loc:,.1f}  "
          f"E[IRA]={mean_ira:,.1f}  SD={std_ira:,.1f}")
    print(f"    ρ(Loc,IRA)={rho:.4f}  "
          f"μ=[{mu_loc:.4f}, {mu_ira:.4f}]  σ=[{sig_loc:.4f}, {sig_ira:.4f}]")
    return cal


# =============================================================================
# MONTE CARLO
# =============================================================================
def run_monte_carlo(elasticities, cal, n_sim=N_SIM, n_years=N_YEARS,
                    seed=RANDOM_SEED, threshold=INSOLVENCY_THRESHOLD):
    rng = np.random.default_rng(seed)

    mu_loc, sig_loc = cal["mu_loc"], cal["sig_loc"]
    mu_ira, sig_ira = cal["mu_ira"], cal["sig_ira"]
    rho = cal["rho"]
    baseline_budget = cal["baseline_budget"]

    cov = np.array([[sig_loc ** 2, rho * sig_loc * sig_ira],
                    [rho * sig_loc * sig_ira, sig_ira ** 2]])
    L = np.linalg.cholesky(cov)

    z = rng.standard_normal(size=(2, n_sim))
    corr = L @ z
    sim_loc = np.exp(mu_loc + corr[0, :])
    sim_ira = np.exp(mu_ira + corr[1, :])
    total_income = sim_loc + sim_ira

    income_paths = np.zeros((n_sim, n_years))
    for i in range(n_sim):
        z_y = rng.standard_normal(size=(2, n_years))
        cy = L @ z_y
        income_paths[i] = (np.exp(mu_loc + cy[0, :])
                           + np.exp(mu_ira + cy[1, :]))

    surge_A = rng.uniform(1.15, 1.30, n_sim)
    surge_B = rng.uniform(0.95, 1.02, n_sim)
    surge_C = surge_A.copy()

    comp_A = dict(pub=0.35, admin=0.35, edu=0.15, health=0.15)
    comp_B = dict(pub=0.10, admin=0.25, edu=0.35, health=0.30)
    comp_C = comp_B

    def compose(totexp, comp):
        return (totexp * comp["pub"], totexp * comp["admin"],
                totexp * comp["edu"], totexp * comp["health"])

    totexp_A = baseline_budget * surge_A
    totexp_B = baseline_budget * surge_B
    totexp_C = baseline_budget * surge_C

    pub_A, _, edu_A, hea_A = compose(totexp_A, comp_A)
    pub_B, _, edu_B, hea_B = compose(totexp_B, comp_B)
    pub_C, _, edu_C, hea_C = compose(totexp_C, comp_C)

    b = elasticities["coefficients"]
    intercept = elasticities["intercept"]
    resid_std = elasticities["resid_std"]
    mean_log_enc = elasticities["mean_predictors"]["log_enc"]

    def simulate_pi(pub, edu, hea):
        pi_hat = (intercept + b["log_pub"] * np.log(pub + 1)
                  + b["log_edu"] * np.log(edu + 1)
                  + b["log_hea"] * np.log(hea + 1)
                  + b["log_enc"] * mean_log_enc)
        return np.clip(pi_hat + rng.normal(0, resid_std, size=len(pub)), 0, 1)

    pi_A = simulate_pi(pub_A, edu_A, hea_A)
    pi_B = simulate_pi(pub_B, edu_B, hea_B)
    pi_C = simulate_pi(pub_C, edu_C, hea_C)

    def flag(totexp):
        return ((totexp - total_income) / total_income) > threshold

    ins_A, ins_B, ins_C = flag(totexp_A), flag(totexp_B), flag(totexp_C)
    d_A, d_B, d_C = totexp_A - total_income, totexp_B - total_income, totexp_C - total_income

    def sim_cum(totexp_vec, label):
        cum_def = np.zeros(n_sim)
        cum_ratio = np.zeros(n_sim)
        n_iy = np.zeros(n_sim, dtype=int)
        for i in range(n_sim):
            d_y = totexp_vec[i] - income_paths[i]
            r_y = d_y / income_paths[i]
            cum_def[i] = d_y.sum()
            cum_ratio[i] = d_y.sum() / income_paths[i].sum()
            n_iy[i] = int(np.sum(r_y > threshold))
        return {f"cum_def_{label}": cum_def, f"cum_ratio_{label}": cum_ratio,
                f"n_insolv_years_{label}": n_iy}

    pA, pB, pC = sim_cum(totexp_A, "A"), sim_cum(totexp_B, "B"), sim_cum(totexp_C, "C")

    def mc_se_p(p): return float(np.sqrt(p * (1 - p) / n_sim))
    def mc_se_m(x): return float(np.std(x, ddof=1) / np.sqrt(n_sim))

    out = {
        "sim_loc": sim_loc, "sim_ira": sim_ira, "total_income": total_income,
        "surge_A": surge_A, "surge_B": surge_B, "surge_C": surge_C,
        "totexp_A": totexp_A, "totexp_B": totexp_B, "totexp_C": totexp_C,
        "deficit_A": d_A, "deficit_B": d_B, "deficit_C": d_C,
        "pi_A": pi_A, "pi_B": pi_B, "pi_C": pi_C,
        "insolvent_A_1y": ins_A, "insolvent_B_1y": ins_B, "insolvent_C_1y": ins_C,
        "mc_se": {
            "insolv_A_1y": mc_se_p(ins_A.mean()),
            "insolv_B_1y": mc_se_p(ins_B.mean()),
            "n_insolv_A_10y": mc_se_m(pA["n_insolv_years_A"]),
            "n_insolv_B_10y": mc_se_m(pB["n_insolv_years_B"]),
            "cum_ratio_A_10y": mc_se_m(pA["cum_ratio_A"]),
            "cum_ratio_B_10y": mc_se_m(pB["cum_ratio_B"]),
            "mean_pi_A": mc_se_m(pi_A),
            "mean_pi_B": mc_se_m(pi_B),
        },
    }
    out.update(pA); out.update(pB); out.update(pC)
    return out


# =============================================================================
# TABLES
# =============================================================================
def build_three_level_elasticity_table(results):
    rows = []
    for label, key in [("ln(pub welfare + 1)", "log_pub"),
                       ("ln(education + 1)", "log_edu"),
                       ("ln(health + 1)", "log_hea"),
                       ("ln(ENC)", "log_enc")]:
        row = {"Predictor": label}
        for lvl in ["province", "city", "municipality"]:
            e = results.get(lvl)
            if e is None:
                row[f"{lvl.title()} β (SE)"] = "—"
                row[f"{lvl.title()} 95% CI"] = "—"
                continue
            b = e["coefficients"][key]
            se = e["std_errors"][key]
            lo, hi = e["ci_95"].get(key, (np.nan, np.nan))
            row[f"{lvl.title()} β (SE)"] = f"{b:+.4f} ({se:.4f})"
            row[f"{lvl.title()} 95% CI"] = f"[{lo:+.4f}, {hi:+.4f}]"
        rows.append(row)

    # Summary rows — all formatted as strings, N (cycles) as int
    for summary_label, key, formatter in [
        ("Within-R²", "within_r2", lambda x: f"{float(x):.3f}"),
        ("F-statistic", "f_stat", lambda x: f"{float(x):.2f}"),
        ("df_resid", "df_resid", lambda x: f"{int(x)}"),
        ("N (LGU-cycles)", "n_obs", lambda x: f"{int(x):,}"),
        ("N (LGUs)", "n_lgu", lambda x: f"{int(x):,}"),
        ("N (cycles)", "n_cycles", lambda x: str(int(x))),
    ]:
        row = {"Predictor": summary_label}
        for lvl in ["province", "city", "municipality"]:
            e = results.get(lvl)
            row[f"{lvl.title()} β (SE)"] = (
                formatter(e[key]) if e else "—")
            row[f"{lvl.title()} 95% CI"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def build_three_level_summary_table(sims):
    def disp(arr):
        return (f"{arr.mean():.2f} "
                f"(SD {arr.std(ddof=1):.2f}; "
                f"5–95% {np.percentile(arr,5):.2f}–{np.percentile(arr,95):.2f})")

    rows = []
    for label, key in [
        ("Expected expenditure (PHP M)", "totexp"),
        ("Expected deficit (PHP M)", "deficit"),
        ("1-year insolvency (%)", "insolvent_1y"),
        ("Mean insolvent years (of 10)", "n_insolv_years"),
        ("95% VaR cum. deficit/income", "cum_ratio_VaR"),
        ("Mean vote share", "pi"),
    ]:
        row = {"Metric": label}
        for scen in ["A", "B"]:
            for lvl in ["province", "city", "municipality"]:
                sim = sims.get(lvl)
                col = f"{lvl[:3].title()}-{scen}"
                if sim is None:
                    row[col] = "—"
                    continue
                if key == "totexp":
                    row[col] = disp(sim[f"totexp_{scen}"])
                elif key == "deficit":
                    row[col] = disp(sim[f"deficit_{scen}"])
                elif key == "insolvent_1y":
                    arr = sim[f"insolvent_{scen}_1y"].astype(float)
                    row[col] = (f"{100*arr.mean():.1f} "
                                f"(SE {100*arr.std(ddof=1)/np.sqrt(len(arr)):.2f})")
                elif key == "n_insolv_years":
                    row[col] = disp(sim[f"n_insolv_years_{scen}"])
                elif key == "cum_ratio_VaR":
                    arr = sim[f"cum_ratio_{scen}"]
                    row[col] = (f"{np.percentile(arr,95):.3f} "
                                f"(SD {arr.std(ddof=1):.3f})")
                elif key == "pi":
                    row[col] = disp(sim[f"pi_{scen}"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_sensitivity_table(elasticities, cal, n_sim=2_000,
                            seed=RANDOM_SEED + 1):
    rows = []
    for rho in [0.30, cal["rho"], 0.60]:
        for thresh in [0.03, 0.05, 0.10]:
            cal_alt = dict(cal)
            sl, si = cal["sig_loc"], cal["sig_ira"]
            cal_alt["Sigma"] = np.array([
                [sl**2, rho * sl * si], [rho * sl * si, si**2]])
            cal_alt["rho"] = rho
            sim = run_monte_carlo(elasticities, cal_alt, n_sim=n_sim,
                                  seed=seed, threshold=thresh)
            rows.append({
                "rho": rho, "threshold": thresh,
                "A_1yr": 100 * sim["insolvent_A_1y"].mean(),
                "B_1yr": 100 * sim["insolvent_B_1y"].mean(),
                "A_insolv_yrs": sim["n_insolv_years_A"].mean(),
                "B_insolv_yrs": sim["n_insolv_years_B"].mean(),
            })
    return pd.DataFrame(rows)


def build_mandanas_sensitivity(fiscal, elasticities):
    post = fiscal[fiscal["year"] >= 2022]
    if len(post) < 100:
        print("[!] Mandanas sensitivity: post-2022 subset too small")
        return None
    cal_post = calibrate_lognormal(post, "province")
    sim = run_monte_carlo(elasticities, cal_post, n_sim=2_000)
    return {"A_1yr": float(sim["insolvent_A_1y"].mean()),
            "B_1yr": float(sim["insolvent_B_1y"].mean()),
            "baseline": cal_post["baseline_budget"]}


# =============================================================================
# FIGURES
# =============================================================================
def fig1_empirical_pbc(fiscal, outdir):
    """
    Figure 1. Empirical PBC by LGU level.

    FIX vs v9: the `_real` columns in panel.csv are already in PHP millions.
    The prior version divided by 1e6, producing y-axis values six orders of
    magnitude too small (0.0004 instead of ~400). Removed.
    """
    smap = {"pubwelf": "Public welfare", "govexp": "General admin.",
            "educexp": "Education", "healthexp": "Health"}
    cols = [c for c in smap if c in fiscal.columns]
    types = [("Province", "Provinces"), ("City", "Cities"),
             ("Municipality", "Municipalities")]
    colors = ["#d9534f", "#f0ad4e", "#0275d8", "#5cb85c"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (t, label) in zip(axes, types):
        sub = fiscal[fiscal["lgutype"].str.contains(t, case=False, na=False)].copy()
        if sub.empty:
            ax.set_title(f"{label}: no data"); continue
        pcols = []
        for c in cols:
            cr = f"{c}_real"
            # _real is already in PHP M — no division
            sub[f"{c}_plot"] = sub[cr] if cr in sub.columns else sub[c]
            pcols.append(f"{c}_plot")

        summ = sub.groupby("pbc_indicator")[pcols].mean()
        summ.index = summ.index.map({0: "Off-election", 1: "Pre-election"})
        summ.plot(kind="bar", ax=ax, width=0.7, edgecolor="black",
                  linewidth=0.4, color=colors[:len(pcols)])
        ax.set_title(f"{label} (N = {sub['psgc10'].nunique()})")
        ax.set_xlabel("Cycle phase")
        ax.set_ylabel("Mean expenditure (PHP M, 2018)")
        ax.set_xticklabels(summ.index, rotation=0)
        ax.legend([smap[c] for c in cols], frameon=False, fontsize=8)
        ax.text(0.99, 0.02, f"N = {len(sub):,} LGU-years",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="gray")
    fig.suptitle("Figure 1. Empirical PBC by LGU level "
                 "(constant 2018 PHP M)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_empirical_pbc.png"))
    plt.close(fig)


def fig2_tradeoff(sim, elasticities, cal, outdir):
    """
    Figure 2. Vote share and insolvency vs surge (two panels).

    FIX vs v9: shared y-axis limits across panels, reference line at the
    mean simulated vote share, numeric span in the caption. The prior
    narrow y-axes made small effects look large.
    """
    base = cal["baseline_budget"]
    b = elasticities["coefficients"]
    intercept = elasticities["intercept"]
    mlenc = elasticities["mean_predictors"]["log_enc"]

    surges = np.linspace(1.00, 1.45, 25)
    pi_vals, ins_vals = [], []
    for s in surges:
        te = base * s
        pub, edu, hea = te * 0.35, te * 0.15, te * 0.15
        pi_hat = (intercept + b["log_pub"] * np.log(pub + 1)
                  + b["log_edu"] * np.log(edu + 1)
                  + b["log_hea"] * np.log(hea + 1)
                  + b["log_enc"] * mlenc)
        pi_vals.append(float(np.clip(pi_hat, 0, 1)))
        d = te - sim["total_income"]
        ins_vals.append(100 * float(np.mean(
            (d / sim["total_income"]) > INSOLVENCY_THRESHOLD)))

    pi_span = max(pi_vals) - min(pi_vals)
    pi_mid = float(np.mean(sim["pi_A"]))
    ins_span = max(ins_vals) - min(ins_vals)

    # Fixed y-limits to make small vote-share changes look small
    y_pi_lim = (min(pi_vals) - 0.01, max(pi_vals) + 0.01)
    y_ins_lim = (min(ins_vals) - 2, max(ins_vals) + 2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(surges, pi_vals, "b-o", markersize=4, label="π̂ (surge)")
    axes[0].axhline(pi_mid, color="grey", linestyle="--", linewidth=1.0,
                    label=f"Mean vote share = {pi_mid:.3f}")
    axes[0].set_xlabel("Surge multiplier")
    axes[0].set_ylabel("Simulated incumbent vote share")
    axes[0].set_title("(a) Vote share vs surge")
    axes[0].set_ylim(*y_pi_lim)
    axes[0].grid(alpha=0.5)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].text(0.99, 0.02,
                 f"Total span = {pi_span:.4f}",
                 transform=axes[0].transAxes, ha="right", va="bottom",
                 fontsize=8, color="gray")

    axes[1].plot(surges, ins_vals, "r--s", markersize=4)
    axes[1].set_xlabel("Surge multiplier")
    axes[1].set_ylabel("Insolvency probability (%)")
    axes[1].set_title("(b) Insolvency risk vs surge")
    axes[1].set_ylim(*y_ins_lim)
    axes[1].grid(alpha=0.5)
    axes[1].text(0.99, 0.02, f"Total span = {ins_span:.1f} pp",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=8, color="gray")

    fig.suptitle(f"Figure 2. Vote-share and insolvency trade-off "
                 f"({cal['level'].title()})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig2_tradeoff_{cal['level']}.png"))
    plt.close(fig)


def fig3_heatmap(sim, elasticities, cal, outdir):
    """
    Figure 3. Simulated vote share by ENC and surge.

    FIX vs v9: ENC grid restricted to [1.5, 4.0]. At ENC = 1 the vote share
    is 1.0 by construction (arithmetic identity); including ENC = 1 makes
    the heatmap visually dominated by a boundary artifact.
    """
    base = cal["baseline_budget"]
    b = elasticities["coefficients"]
    intercept = elasticities["intercept"]
    enc_grid = np.linspace(1.5, 4.0, 6)
    surge_grid = np.linspace(1.00, 1.35, 5)
    M = np.zeros((len(enc_grid), len(surge_grid)))
    for i, enc in enumerate(enc_grid):
        for j, s in enumerate(surge_grid):
            te = base * s
            pub, edu, hea = te * 0.35, te * 0.15, te * 0.15
            pi_hat = (intercept + b["log_pub"] * np.log(pub + 1)
                      + b["log_edu"] * np.log(edu + 1)
                      + b["log_hea"] * np.log(hea + 1)
                      + b["log_enc"] * np.log(enc))
            M[i, j] = float(np.clip(pi_hat, 0, 1))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.heatmap(M, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=[f"{s:.2f}×" for s in surge_grid],
                yticklabels=[f"{e:.2f}" for e in enc_grid], ax=ax,
                cbar_kws={"label": "Simulated vote share"})
    ax.set_title(f"Figure 3. Vote share by ENC and surge — "
                 f"{cal['level'].title()}")
    ax.set_xlabel("Surge multiplier")
    ax.set_ylabel("Effective number of candidates (ENC)")
    fig.text(0.01, 0.01,
             "Note: ENC grid restricted to [1.5, 4.0]. At ENC = 1 the "
             "vote share is 1.0 by construction and is excluded.",
             fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig3_heatmap_{cal['level']}.png"))
    plt.close(fig)


def fig4_deficit_density(sim, cal, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(sim["deficit_A"], fill=True, color="crimson",
                label="Scenario A", alpha=0.35, ax=ax)
    sns.kdeplot(sim["deficit_B"], fill=True, color="teal",
                label="Scenario B", alpha=0.35, ax=ax)
    for k, col, lab in [("deficit_A", "darkred", "A"),
                        ("deficit_B", "darkgreen", "B")]:
        v = float(np.percentile(sim[k], 95))
        ax.axvline(v, color=col, linestyle="--", linewidth=1.6,
                   label=f"95% VaR {lab}: {v:,.0f} M")
    ax.axvline(0, color="black", linestyle=":", linewidth=1.0)
    ax.set_title(f"Figure 4. Deficit distributions — {cal['level'].title()}")
    ax.set_xlabel("Deficit (PHP M, 2018)"); ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.99, 0.02, f"N = {len(sim['deficit_A']):,} draws",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig4_deficit_var_{cal['level']}.png"))
    plt.close(fig)


def fig5_revenue_fits(fiscal_level, cal, outdir):
    """
    Figure 5. QQ plots of log revenue vs fitted lognormal.

    FIX vs v9: Anderson–Darling statistic reported in each panel's title so
    the reader can judge whether the lognormal fit is adequate. The prior
    version relied on visual inspection alone, and the reviewer explicitly
    asked for a goodness-of-fit statistic.
    """
    loc_col = "totlocsrc_real" if "totlocsrc_real" in fiscal_level.columns else "totlocsrc"
    ira_col = "ira_real" if "ira_real" in fiscal_level.columns else "ira"
    loc = fiscal_level[loc_col].dropna()
    ira = fiscal_level[ira_col].dropna()
    loc = loc[loc > 0]; ira = ira[ira > 0]
    if len(loc) < 20 or len(ira) < 20:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, (data, mu, sig, label) in zip(
        axes,
        [(loc, cal["mu_loc"], cal["sig_loc"], "Local revenue"),
         (ira, cal["mu_ira"], cal["sig_ira"], "IRA")]):
        log_obs = np.sort(np.log(data.values))
        q = np.linspace(0.01, 0.99, len(log_obs))
        log_theo = stats.norm.ppf(q, loc=mu, scale=sig)

        # Anderson–Darling statistic against the fitted lognormal
        # Transform to standard normal: (log(x) - mu)/sig ~ N(0,1)
        z = (np.log(data.values) - mu) / sig
        ad_res = stats.anderson(z, dist="norm")
        # Critical value at 5% (index 2 in anderson's default 15/10/5/2.5/1)
        ad_stat = float(ad_res.statistic)
        ad_crit_5 = float(ad_res.critical_values[2])
        ad_verdict = "reject" if ad_stat > ad_crit_5 else "do not reject"

        ax.scatter(log_theo, log_obs, s=3, alpha=0.35, color="steelblue")
        lo, hi = min(log_theo.min(), log_obs.min()), max(log_theo.max(), log_obs.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
        ax.set_xlabel("Theoretical log quantile")
        ax.set_ylabel("Observed log quantile")
        ax.set_title(f"{label} (N = {len(data):,})")
        ax.grid(alpha=0.4)
        ax.text(0.99, 0.02,
                f"AD = {ad_stat:.3f} (5% crit = {ad_crit_5:.3f}) — {ad_verdict}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="gray")
    fig.suptitle(f"Figure 5. Lognormal QQ plots — {cal['level'].title()}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig5_revenue_qq_{cal['level']}.png"))
    plt.close(fig)


def fig6_diagnostics(sim, cal, outdir):
    ratio = sim["deficit_A"] / sim["total_income"]
    rho = float(np.corrcoef(sim["pi_A"], ratio)[0, 1])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(sim["total_income"], bins=50, color="steelblue",
                    alpha=0.75, edgecolor="black", linewidth=0.3)
    axes[0, 0].set_title("(a) Simulated total income")
    axes[0, 1].hist(ratio, bins=60, color="crimson", alpha=0.75,
                    edgecolor="black", linewidth=0.3)
    axes[0, 1].axvline(INSOLVENCY_THRESHOLD, color="red", linestyle="--",
                       label="5% threshold"); axes[0, 1].legend()
    axes[0, 1].set_title("(b) Deficit/income ratio (A)")
    axes[1, 0].scatter(sim["pi_A"], ratio, alpha=0.25, s=4, color="purple")
    axes[1, 0].set_title(f"(c) Vote share vs fiscal risk (ρ={rho:+.3f})")
    axes[1, 1].scatter(sim["total_income"], sim["deficit_A"],
                       alpha=0.25, s=4, color="darkgreen")
    axes[1, 1].axhline(0, color="black", linestyle="--")
    axes[1, 1].set_title("(d) Deficit vs income")
    fig.suptitle(f"Figure 6. Diagnostics — {cal['level'].title()}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig6_diagnostics_{cal['level']}.png"))
    plt.close(fig)


def fig7_trajectories(sim, cal, outdir, n_paths=60, seed=RANDOM_SEED+7):
    rng = np.random.default_rng(seed)
    ml, sl = cal["mu_loc"], cal["sig_loc"]
    mi, si = cal["mu_ira"], cal["sig_ira"]
    paths = np.zeros((n_paths, N_YEARS))
    for i in range(n_paths):
        te = float(sim["totexp_A"][i]) * np.ones(N_YEARS)
        inc = rng.lognormal(ml, sl, N_YEARS) + rng.lognormal(mi, si, N_YEARS)
        paths[i] = np.cumsum(te - inc)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i in range(n_paths):
        ax.plot(range(1, N_YEARS+1), paths[i], color="crimson",
                alpha=0.20, linewidth=0.9)
    ax.plot(range(1, N_YEARS+1), paths.mean(axis=0), "k--",
            linewidth=2.4, label="Mean cumulative deficit")
    ax.axhline(0, color="navy", linestyle=":", label="Zero")
    ax.set_title(f"Figure 7. 10-year trajectories — {cal['level'].title()}")
    ax.set_xlabel("Years"); ax.set_ylabel("Cumulative deficit (PHP M)")
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.99, 0.02, f"{n_paths} illustrative paths of N={len(sim['totexp_A']):,}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig7_trajectories_{cal['level']}.png"))
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    panel_path = os.path.join(_ROOT, "data", "panel.csv")
    elec_prov_path = os.path.join(_ROOT, "data", "election_panel_hybrid.csv")
    elec_mayor_path = os.path.join(_ROOT, "data", "election_panel_mayor.csv")

    print(f"[*] Loading fiscal panel: {panel_path}")
    fiscal = load_fiscal_panel(panel_path)

    print(f"\n[*] Loading province electoral panel: {elec_prov_path}")
    elec_prov = load_election_panel(elec_prov_path, "Province")
    reg_prov = build_regression_panel(fiscal, elec_prov)

    print(f"\n[*] Loading mayor electoral panel: {elec_mayor_path}")
    elec_mayor = load_election_panel(elec_mayor_path, "City+Municipality")
    reg_mayor = build_regression_panel(fiscal, elec_mayor)

    print("\n[*] Estimating two-way FE elasticities (PanelOLS) …")
    e_prov = estimate_elasticities(reg_prov, "province")
    e_city = estimate_elasticities(reg_mayor, "city")
    e_muni = estimate_elasticities(reg_mayor, "municipality")
    results = {"province": e_prov, "city": e_city, "municipality": e_muni}

    table2 = build_three_level_elasticity_table(results)
    table2.to_csv(os.path.join(OUTPUT_DIR, "table2_three_level.csv"), index=False)
    print("\n--- Table 2 ---")
    print(table2.to_string(index=False))

    cal_prov = calibrate_lognormal(fiscal, "province")
    cal_city = calibrate_lognormal(fiscal, "city")
    cal_muni = calibrate_lognormal(fiscal, "municipality")

    print("\n[*] Running Monte Carlo (10,000 draws each) …")
    sims = {}
    if e_prov: sims["province"] = run_monte_carlo(e_prov, cal_prov)
    if e_city: sims["city"] = run_monte_carlo(e_city, cal_city)
    if e_muni: sims["municipality"] = run_monte_carlo(e_muni, cal_muni)

    table3 = build_three_level_summary_table(sims)
    table3.to_csv(os.path.join(OUTPUT_DIR, "table3_three_level.csv"), index=False)
    print("\n--- Table 3 ---")
    print(table3.to_string(index=False))

    print("\n[*] Sensitivity (ρ, threshold) …")
    sens = build_sensitivity_table(e_prov or e_city, cal_prov, n_sim=2_000)
    sens.to_csv(os.path.join(OUTPUT_DIR, "table4_sensitivity.csv"), index=False)
    print(sens.to_string(index=False))

    print("\n[*] Mandanas-Garcia robustness (2022+) …")
    mand = build_mandanas_sensitivity(fiscal, e_prov or e_city)
    if mand:
        print(f"    Post-2022 baseline: {mand['baseline']:,.0f} PHP M")
        print(f"    A 1-yr insolv: {100*mand['A_1yr']:.1f}%  "
              f"B: {100*mand['B_1yr']:.1f}%")

    print("\n[*] Generating figures …")
    fig1_empirical_pbc(fiscal, OUTPUT_DIR)
    for lvl, sim, e, cal in [("province", sims.get("province"), e_prov, cal_prov),
                             ("city", sims.get("city"), e_city, cal_city),
                             ("municipality", sims.get("municipality"), e_muni, cal_muni)]:
        if sim is None or e is None:
            continue
        fig2_tradeoff(sim, e, cal, OUTPUT_DIR)
        fig3_heatmap(sim, e, cal, OUTPUT_DIR)
        fig4_deficit_density(sim, cal, OUTPUT_DIR)
        fig6_diagnostics(sim, cal, OUTPUT_DIR)
        fig7_trajectories(sim, cal, OUTPUT_DIR)
        fl = fiscal[fiscal["lgutype"].str.contains(lvl, case=False, na=False)]
        fig5_revenue_fits(fl, cal, OUTPUT_DIR)

    summary = {lvl: {
        "n_obs": e["n_obs"], "n_lgu": e["n_lgu"], "n_cycles": e["n_cycles"],
        "within_r2": e["within_r2"], "f_stat": e["f_stat"],
        "coefficients": e["coefficients"], "std_errors": e["std_errors"],
        "ci_95": {k: list(v) for k, v in e["ci_95"].items()},
        "p_values": e["p_values"], "outcome_source": e["outcome_source"],
    } for lvl, e in results.items() if e is not None}
    with open(os.path.join(OUTPUT_DIR, "results_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n[✓] Outputs saved to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()