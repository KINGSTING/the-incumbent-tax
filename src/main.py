"""
================================================================================
The Incumbent Tax: A Stochastic Assessment of Pre-Election Clientelism and
Fiscal Insolvency in Philippine Local Government Units
--------------------------------------------------------------------------------
v5 — three-level analysis (Province / City / Municipality).

  • Province electoral panel: election_panel_hybrid.csv
      - COMELEC 1992-2004 + OpenHalalan 2007-2025 (hybrid)
  • City + Municipality electoral panel: election_panel_mayor.csv
      - OpenHalalan 2007-2025 (mayoral races)
  • Three-level Table 2 (elasticities) and Table 3 (Monte Carlo)
  • Level filter applied inside estimate_elasticities() and
    calibrate_lognormal()
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# GLOBAL SETTINGS
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.titlesize": 14, "figure.dpi": 300,
})

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SIM = 10_000
N_YEARS = 10
INSOLVENCY_THRESHOLD = 0.05
REV_CORR_DEFAULT = 0.45

ELECTION_YEARS = [1992, 1995, 1998, 2001, 2004, 2007, 2010,
                  2013, 2016, 2019, 2022, 2025]

CPI_FALLBACK = {year: 100.0 for year in range(1992, 2026)}


# =============================================================================
# 1. DATA LOADERS
# =============================================================================
def load_fiscal_panel(panel_path: str) -> pd.DataFrame:
    df = pd.read_csv(panel_path, low_memory=False)

    df["psgc10"] = (df["psgc10"].astype(str).str.strip()
                    .str.replace(r"\.0$", "", regex=True).str.zfill(10))
    df["lgu"] = df["lgu_name"].astype(str).str.strip().str.upper()
    df["lgutype"] = df["lgu_type"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")

    if "in_sre" in df.columns:
        df = df[df["in_sre"].astype(str).str.lower()
                  .isin(["true", "1", "yes"])]

    sum_cols = [c for c in [
        "total_local_sources", "nta_ira", "other_national_shares",
        "total_external_sources", "total_current_operating_income",
        "education", "health", "social_welfare", "total_social_services",
        "gps", "economic_services", "total_current_operating_exp",
        "total_capital_investment_exp", "total_non_operating_exp",
        "total_expenditures",
    ] if c in df.columns]

    agg = {c: "sum" for c in sum_cols}
    for c in ["lgu", "lgutype", "province", "region", "income_class",
              "pop_2024", "poverty_pi"]:
        if c in df.columns:
            agg[c] = "first"

    df = df.groupby(["psgc10", "year"], as_index=False).agg(agg)

    df = df.rename(columns={
        "total_local_sources": "totlocsrc", "nta_ira": "ira",
        "total_expenditures": "totexp", "social_welfare": "pubwelf",
        "education": "educexp", "health": "healthexp", "gps": "govexp",
        "pop_2024": "population", "poverty_pi": "poverty",
    })

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
            positive = df[c][df[c] > 0]
            if len(positive) > 10:
                q99 = np.exp(np.log(positive).quantile(0.99))
                df[c] = np.where(df[c] > q99, q99, df[c])

    if "cpi_2018" in df.columns:
        deflator = df["cpi_2018"] / 100.0
    else:
        deflator = df["year"].map(
            lambda y: CPI_FALLBACK.get(int(y), 100.0)) / 100.0

    for c in ["totlocsrc", "ira", "totexp", "pubwelf", "educexp",
              "healthexp", "govexp"]:
        if c in df.columns:
            df[f"{c}_real"] = df[c] / deflator

    pre_years = [y - 1 for y in ELECTION_YEARS]
    df["pbc_indicator"] = df["year"].isin(pre_years).astype(int)

    print(f"[*] Fiscal panel: {len(df):,} LGU-years, "
          f"{df['psgc10'].nunique():,} LGUs, "
          f"years {int(df['year'].min())}–{int(df['year'].max())}")
    if "lgutype" in df.columns:
        print(df.groupby("lgutype")["psgc10"].nunique().to_string())
    return df


def load_election_panel(election_path: str, label: str) -> pd.DataFrame:
    elec = pd.read_csv(election_path)
    elec["psgc10"] = (elec["psgc10"].astype(str).str.strip()
                      .str.replace(r"\.0$", "", regex=True).str.zfill(10))
    for c in ["elecyr", "pi", "ENC_lt", "n_candidates"]:
        if c in elec.columns:
            elec[c] = pd.to_numeric(elec[c], errors="coerce")

    elec = elec.dropna(subset=["psgc10", "elecyr"])
    elec = elec.drop_duplicates(subset=["psgc10", "elecyr"])

    # Normalize lgutype string
    if "lgutype" in elec.columns:
        elec["lgutype"] = elec["lgutype"].astype(str).str.strip().str.lower()
    else:
        elec["lgutype"] = "province"

    print(f"\n[*] {label} panel: {len(elec):,} LGU-cycles, "
          f"{elec['psgc10'].nunique():,} LGUs")
    print(f"    by type:\n"
          f"{elec.groupby('lgutype')['psgc10'].nunique().to_string()}")
    print(f"    cycles: {sorted(elec['elecyr'].astype(int).unique())}")
    return elec


def build_regression_panel(fiscal: pd.DataFrame,
                           elec: pd.DataFrame) -> pd.DataFrame:
    fiscal_pre = fiscal.copy()
    fiscal_pre["elecyr"] = fiscal_pre["year"] + 1
    reg = elec.merge(
        fiscal_pre, on=["psgc10", "elecyr"], how="inner",
        suffixes=("", "_fiscal"),
    )
    print(f"\n[*] Regression panel: {len(reg):,} LGU-cycles, "
          f"{reg['psgc10'].nunique()} LGUs, "
          f"{reg['elecyr'].nunique()} cycles")
    return reg


# =============================================================================
# 2. TWO-WAY FE VOTER ELASTICITY MODEL (parameterized by level)
# =============================================================================
def estimate_elasticities(reg: pd.DataFrame, level: str = "province") -> dict:
    """
    Two-way FE (unit + cycle) with unit-clustered SEs.
    level = "province" | "city" | "municipality"
    """
    # Level filter
    if level == "province":
        reg = reg[reg["lgutype"].str.contains("province", case=False, na=False)]
    elif level == "city":
        reg = reg[reg["lgutype"].str.contains("city", case=False, na=False)]
    elif level == "municipality":
        reg = reg[reg["lgutype"].str.contains("municipality",
                                              case=False, na=False)]
    else:
        raise ValueError(f"Unknown level: {level}")

    # Drop uncontested races (keep NaN for pre-2007 COMELEC rows)
    if "n_candidates" in reg.columns:
        reg = reg[reg["n_candidates"].isna() | (reg["n_candidates"] > 1)]

    reg = reg[reg["pi"].notna() & (reg["pi"] > 0) & (reg["pi"] <= 1)].copy()

    needed = ["lgu", "elecyr", "pi", "pubwelf_real", "educexp_real",
              "healthexp_real", "ENC_lt"]
    needed = [c for c in needed if c in reg.columns]
    df = reg[needed].copy()
    df = df.dropna(subset=["pi", "pubwelf_real", "educexp_real",
                           "healthexp_real", "ENC_lt", "lgu", "elecyr"])
    df = df[np.isfinite(df["ENC_lt"]) & (df["ENC_lt"] > 0)]

    if len(df) < 30:
        print(f"\n[!] {level}: only {len(df)} usable rows — skipping.")
        return None

    df["log_pub"] = np.log(df["pubwelf_real"] + 1)
    df["log_edu"] = np.log(df["educexp_real"] + 1)
    df["log_hea"] = np.log(df["healthexp_real"] + 1)
    df["log_enc"] = np.log(df["ENC_lt"])

    means = {k: float(df[k].mean()) for k in
             ["log_pub", "log_edu", "log_hea", "log_enc"]}
    mean_pi = float(df["pi"].mean())

    # Two-way within transformation
    cols = ["pi", "log_pub", "log_edu", "log_hea", "log_enc"]
    lgu_means = df.groupby("lgu")[cols].transform("mean")
    within = df[cols] - lgu_means
    cycle_means = within.groupby(df["elecyr"]).transform("mean")
    within = within - cycle_means

    X = within[["log_pub", "log_edu", "log_hea", "log_enc"]].copy()
    y = within["pi"].values
    X = sm.add_constant(X, has_constant="skip")
    groups = df["lgu"].values

    try:
        model = sm.OLS(y, X).fit(cov_type="cluster",
                                 cov_kwds={"groups": groups})
    except Exception:
        model = sm.OLS(y, X).fit()

    b = {k: float(model.params.get(k, 0.0))
         for k in ["log_pub", "log_edu", "log_hea", "log_enc"]}
    intercept_adj = mean_pi - sum(b[k] * means[k] for k in b)

    ss_res = float(np.sum((y - model.fittedvalues) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    within_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    result = {
        "coefficients": b,
        "std_errors": {k: float(v) for k, v in model.bse.items()},
        "p_values": {k: float(v) for k, v in model.pvalues.items()},
        "n_obs": int(model.nobs),
        "n_lgu": int(df["lgu"].nunique()),
        "n_cycles": int(df["elecyr"].nunique()),
        "within_r2": float(within_r2),
        "f_stat": float(model.fvalue) if model.fvalue is not None else np.nan,
        "resid_std": float(np.std(model.resid, ddof=1)),
        "intercept": float(intercept_adj),
        "mean_pi": mean_pi,
        "mean_predictors": means,
        "level": level,
    }

    print(f"\n[*] Two-way FE ({level}):")
    print(f"    N = {result['n_obs']}, "
          f"LGUs = {result['n_lgu']}, "
          f"cycles = {result['n_cycles']}, "
          f"within-R² = {within_r2:.4f}")
    for k in ["log_pub", "log_edu", "log_hea", "log_enc"]:
        print(f"    {k:10s}: b={b[k]:+.6f}  "
              f"se={result['std_errors'].get(k, np.nan):.6f}  "
              f"p={result['p_values'].get(k, np.nan):.4f}")
    return result


# =============================================================================
# 3. LOGNORMAL CALIBRATION (parameterized by level)
# =============================================================================
def calibrate_lognormal(fiscal: pd.DataFrame, level: str) -> dict:
    if level == "province":
        sub = fiscal[fiscal["lgutype"].str.contains("province",
                                                    case=False, na=False)]
    elif level == "city":
        sub = fiscal[fiscal["lgutype"].str.contains("city",
                                                    case=False, na=False)]
    elif level == "municipality":
        sub = fiscal[fiscal["lgutype"].str.contains("municipality",
                                                    case=False, na=False)]
    else:
        raise ValueError(f"Unknown level: {level}")

    loc_col = "totlocsrc_real" if "totlocsrc_real" in sub.columns \
        else "totlocsrc"
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
    cov = np.array([
        [sig_loc ** 2, rho * sig_loc * sig_ira],
        [rho * sig_loc * sig_ira, sig_ira ** 2],
    ])

    cal = {
        "mean_loc": mean_loc, "std_loc": std_loc,
        "mean_ira": mean_ira, "std_ira": std_ira, "rho": rho,
        "mu": np.array([mu_loc, mu_ira]), "Sigma": cov,
        "mu_loc": mu_loc, "sig_loc": sig_loc,
        "mu_ira": mu_ira, "sig_ira": sig_ira,
        "baseline_budget": mean_loc + mean_ira,
        "level": level,
    }

    print(f"\n[*] Calibration ({level}): "
          f"mean_loc={mean_loc:,.0f}, mean_ira={mean_ira:,.0f}, "
          f"ρ={rho:.3f}, baseline={mean_loc+mean_ira:,.0f} PHP M")
    return cal


# =============================================================================
# 4. MONTE CARLO SIMULATION
# =============================================================================
def run_monte_carlo(elasticities: dict, cal: dict,
                    n_sim: int = N_SIM, n_years: int = N_YEARS,
                    seed: int = RANDOM_SEED,
                    threshold: float = INSOLVENCY_THRESHOLD) -> dict:
    rng = np.random.default_rng(seed)

    mu_loc, sig_loc = cal["mu_loc"], cal["sig_loc"]
    mu_ira, sig_ira = cal["mu_ira"], cal["sig_ira"]
    rho = cal["rho"]
    baseline_budget = cal["baseline_budget"]

    cov = np.array([
        [sig_loc ** 2, rho * sig_loc * sig_ira],
        [rho * sig_loc * sig_ira, sig_ira ** 2],
    ])
    L = np.linalg.cholesky(cov)

    # 1-year income draws
    z = rng.standard_normal(size=(2, n_sim))
    correlated = L @ z
    sim_loc = np.exp(mu_loc + correlated[0, :])
    sim_ira = np.exp(mu_ira + correlated[1, :])
    total_income = sim_loc + sim_ira

    # 10-year income paths (shared across scenarios)
    income_paths = np.zeros((n_sim, n_years))
    for i in range(n_sim):
        z_y = rng.standard_normal(size=(2, n_years))
        corr_y = L @ z_y
        income_paths[i] = (np.exp(mu_loc + corr_y[0, :])
                           + np.exp(mu_ira + corr_y[1, :]))

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

    b_pub = elasticities["coefficients"]["log_pub"]
    b_edu = elasticities["coefficients"]["log_edu"]
    b_hea = elasticities["coefficients"]["log_hea"]
    b_enc = elasticities["coefficients"]["log_enc"]
    intercept = elasticities["intercept"]
    resid_std = elasticities["resid_std"]
    mean_log_enc = elasticities["mean_predictors"]["log_enc"]

    def simulate_pi(pub, edu, hea):
        pi_hat = (intercept
                  + b_pub * np.log(pub + 1)
                  + b_edu * np.log(edu + 1)
                  + b_hea * np.log(hea + 1)
                  + b_enc * mean_log_enc)
        noise = rng.normal(0, resid_std, size=len(pub))
        return np.clip(pi_hat + noise, 0.0, 1.0)

    pi_A = simulate_pi(pub_A, edu_A, hea_A)
    pi_B = simulate_pi(pub_B, edu_B, hea_B)
    pi_C = simulate_pi(pub_C, edu_C, hea_C)

    def insolv_flag(totexp):
        return ((totexp - total_income) / total_income) > threshold

    insolvent_A_1y = insolv_flag(totexp_A)
    insolvent_B_1y = insolv_flag(totexp_B)
    insolvent_C_1y = insolv_flag(totexp_C)

    deficit_A = totexp_A - total_income
    deficit_B = totexp_B - total_income
    deficit_C = totexp_C - total_income

    def sim_cumulative(totexp_vec, income_paths, label, thresh):
        n = totexp_vec.shape[0]
        cum_def = np.zeros(n)
        cum_ratio = np.zeros(n)
        n_insolv_years = np.zeros(n, dtype=int)
        for i in range(n):
            d_y = totexp_vec[i] - income_paths[i]
            r_y = d_y / income_paths[i]
            cum_def[i] = d_y.sum()
            cum_ratio[i] = d_y.sum() / income_paths[i].sum()
            n_insolv_years[i] = int(np.sum(r_y > thresh))
        return {f"cum_def_{label}": cum_def,
                f"cum_ratio_{label}": cum_ratio,
                f"n_insolv_years_{label}": n_insolv_years}

    paths_A = sim_cumulative(totexp_A, income_paths, "A", threshold)
    paths_B = sim_cumulative(totexp_B, income_paths, "B", threshold)
    paths_C = sim_cumulative(totexp_C, income_paths, "C", threshold)

    def mc_se_prob(p):
        return float(np.sqrt(p * (1 - p) / n_sim))

    def mc_se_mean(x):
        return float(np.std(x, ddof=1) / np.sqrt(n_sim))

    results = {
        "sim_loc": sim_loc, "sim_ira": sim_ira, "total_income": total_income,
        "surge_A": surge_A, "surge_B": surge_B, "surge_C": surge_C,
        "totexp_A": totexp_A, "totexp_B": totexp_B, "totexp_C": totexp_C,
        "deficit_A": deficit_A, "deficit_B": deficit_B, "deficit_C": deficit_C,
        "pi_A": pi_A, "pi_B": pi_B, "pi_C": pi_C,
        "insolvent_A_1y": insolvent_A_1y,
        "insolvent_B_1y": insolvent_B_1y,
        "insolvent_C_1y": insolvent_C_1y,
        "mc_se": {
            "insolv_A_1y": mc_se_prob(insolvent_A_1y.mean()),
            "insolv_B_1y": mc_se_prob(insolvent_B_1y.mean()),
            "insolv_C_1y": mc_se_prob(insolvent_C_1y.mean()),
            "n_insolv_A_10y": mc_se_mean(paths_A["n_insolv_years_A"]),
            "n_insolv_B_10y": mc_se_mean(paths_B["n_insolv_years_B"]),
            "n_insolv_C_10y": mc_se_mean(paths_C["n_insolv_years_C"]),
            "cum_ratio_A_10y": mc_se_mean(paths_A["cum_ratio_A"]),
            "cum_ratio_B_10y": mc_se_mean(paths_B["cum_ratio_B"]),
            "cum_ratio_C_10y": mc_se_mean(paths_C["cum_ratio_C"]),
            "mean_deficit_A": mc_se_mean(deficit_A),
            "mean_deficit_B": mc_se_mean(deficit_B),
            "mean_deficit_C": mc_se_mean(deficit_C),
            "mean_pi_A": mc_se_mean(pi_A),
            "mean_pi_B": mc_se_mean(pi_B),
            "mean_pi_C": mc_se_mean(pi_C),
        },
    }
    results.update(paths_A)
    results.update(paths_B)
    results.update(paths_C)
    return results


# =============================================================================
# 5. THREE-LEVEL TABLES
# =============================================================================
def build_three_level_elasticity_table(results: dict) -> pd.DataFrame:
    rows = []
    for label, key in [
        ("ln(pub welfare + 1)", "log_pub"),
        ("ln(education + 1)",   "log_edu"),
        ("ln(health + 1)",      "log_hea"),
        ("ln(ENC)",             "log_enc"),
    ]:
        row = {"Predictor": label}
        for lvl in ["province", "city", "municipality"]:
            e = results.get(lvl)
            if e is None:
                row[f"{lvl.title()} β"] = "—"
                row[f"{lvl.title()} p"] = "—"
                continue
            b = e["coefficients"][key]
            p = e["p_values"].get(key, np.nan)
            row[f"{lvl.title()} β"] = f"{b:+.4f}"
            row[f"{lvl.title()} p"] = f"{p:.3f}"
        rows.append(row)

    for summary_label, summary_key, formatter in [
        ("Within-R²", "within_r2", lambda x: f"{x:.3f}"),
        ("N (LGU-cycle)", "n_obs", lambda x: f"{x}"),
        ("N (LGUs)", "n_lgu", lambda x: f"{x}"),
        ("N (cycles)", "n_cycles", lambda x: f"{x}"),
    ]:
        row = {"Predictor": summary_label}
        for lvl in ["province", "city", "municipality"]:
            e = results.get(lvl)
            row[f"{lvl.title()} β"] = (
                formatter(e[summary_key]) if e else "—")
            row[f"{lvl.title()} p"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def build_three_level_summary_table(sims: dict) -> pd.DataFrame:
    def pct(x):
        return 100 * float(np.mean(x))

    rows = []
    for label, key in [
        ("Expected total expenditure (PHP M)", "totexp"),
        ("Expected deficit (PHP M)",           "deficit"),
        ("1-year insolvency (%)",              "insolvent_1y"),
        ("Mean insolvent years (of 10)",       "n_insolv_years"),
        ("95% VaR 10-yr cum. deficit/income",  "cum_ratio_VaR"),
        ("Mean vote share",                    "pi"),
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
                    row[col] = f"{sim[f'totexp_{scen}'].mean():,.0f}"
                elif key == "deficit":
                    row[col] = f"{sim[f'deficit_{scen}'].mean():,.0f}"
                elif key == "insolvent_1y":
                    row[col] = f"{pct(sim[f'insolvent_{scen}_1y']):.1f}"
                elif key == "n_insolv_years":
                    row[col] = f"{sim[f'n_insolv_years_{scen}'].mean():.2f}"
                elif key == "cum_ratio_VaR":
                    row[col] = f"{np.percentile(sim[f'cum_ratio_{scen}'], 95):.3f}"
                elif key == "pi":
                    row[col] = f"{sim[f'pi_{scen}'].mean():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 6. FIGURES
# =============================================================================
def fig1_empirical_pbc(fiscal: pd.DataFrame, outdir: str) -> None:
    spending_map = {"pubwelf": "Public welfare", "govexp": "General admin.",
                    "educexp": "Education", "healthexp": "Health"}
    cols = [c for c in spending_map if c in fiscal.columns]
    types = [("Province", "Provinces"), ("City", "Cities"),
             ("Municipality", "Municipalities")]
    colors = ["#d9534f", "#f0ad4e", "#0275d8", "#5cb85c"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (t, label) in zip(axes, types):
        sub = fiscal[fiscal["lgutype"].str.contains(t, case=False, na=False)]
        if sub.empty:
            ax.set_title(f"{label}: no data"); ax.set_xticks([]); continue
        plot_cols = []
        for c in cols:
            c_real = f"{c}_real"
            sub = sub.copy()
            sub[f"{c}_plot"] = sub[c_real] / 1e6 if c_real in sub.columns \
                else sub[c]
            plot_cols.append(f"{c}_plot")
        summ = sub.groupby("pbc_indicator")[plot_cols].mean()
        summ.index = summ.index.map({0: "Off-election", 1: "Pre-election"})
        summ.plot(kind="bar", ax=ax, width=0.7, edgecolor="black",
                  linewidth=0.4, color=colors[:len(plot_cols)])
        ax.set_title(f"{label} (N = {sub['psgc10'].nunique()})")
        ax.set_xlabel("Electoral cycle phase")
        ax.set_ylabel("Mean expenditure (PHP M)")
        ax.set_xticklabels(summ.index, rotation=0)
        ax.legend([spending_map[c] for c in cols], frameon=False, fontsize=8)
    fig.suptitle("Figure 1. Empirical political budget cycle by LGU level")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_empirical_pbc.png"))
    plt.close(fig)


def fig2_tradeoff(sim, elasticities, cal, outdir):
    base_budget = cal["baseline_budget"]
    b = elasticities["coefficients"]
    intercept = elasticities["intercept"]
    mean_log_enc = elasticities["mean_predictors"]["log_enc"]

    surges = np.linspace(1.00, 1.45, 25)
    pi_vals, insolv_vals = [], []
    for s in surges:
        totexp = base_budget * s
        pub, edu, hea = totexp * 0.35, totexp * 0.15, totexp * 0.15
        pi_hat = (intercept + b["log_pub"] * np.log(pub + 1)
                  + b["log_edu"] * np.log(edu + 1)
                  + b["log_hea"] * np.log(hea + 1)
                  + b["log_enc"] * mean_log_enc)
        pi_vals.append(float(np.clip(pi_hat, 0, 1)))
        deficit = totexp - sim["total_income"]
        insolv_vals.append(100 * float(np.mean(
            (deficit / sim["total_income"]) > INSOLVENCY_THRESHOLD)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(surges, pi_vals, "b-o", markersize=4)
    axes[0].set_xlabel("Surge multiplier")
    axes[0].set_ylabel("Simulated vote share")
    axes[0].set_title("(a) Vote share vs surge")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].plot(surges, insolv_vals, "r--s", markersize=4)
    axes[1].set_xlabel("Surge multiplier")
    axes[1].set_ylabel("Insolvency probability (%)")
    axes[1].set_title("(b) Insolvency risk vs surge")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    fig.suptitle(f"Figure 2. Vote-share and insolvency trade-off "
                 f"({cal['level'].title()})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig2_tradeoff_{cal['level']}.png"))
    plt.close(fig)


def fig3_heatmap(sim, cal, outdir):
    base_budget = cal["baseline_budget"]
    enc_grid = np.linspace(1.0, 4.0, 5)
    surge_grid = np.linspace(1.00, 1.35, 5)
    M = np.zeros((len(enc_grid), len(surge_grid)))
    for i, _ in enumerate(enc_grid):
        for j, s in enumerate(surge_grid):
            deficit = base_budget * s - sim["total_income"]
            M[i, j] = 100 * float(np.mean(
                (deficit / sim["total_income"]) > INSOLVENCY_THRESHOLD))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(M, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=[f"{s:.2f}×" for s in surge_grid],
                yticklabels=[f"{e:.1f}" for e in enc_grid],
                ax=ax, cbar_kws={"label": "Insolvency probability (%)"})
    ax.set_title(f"Figure 3. Insolvency risk (%) — {cal['level'].title()}")
    ax.set_xlabel("Surge multiplier")
    ax.set_ylabel("Effective number of candidates (ENC)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig3_heatmap_{cal['level']}.png"))
    plt.close(fig)


def fig4_deficit_density(sim, cal, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(sim["deficit_A"], fill=True, color="crimson",
                label="Scenario A", alpha=0.35, ax=ax)
    sns.kdeplot(sim["deficit_B"], fill=True, color="teal",
                label="Scenario B", alpha=0.35, ax=ax)
    varA = float(np.percentile(sim["deficit_A"], 95))
    varB = float(np.percentile(sim["deficit_B"], 95))
    ax.axvline(varA, color="darkred", linestyle="--", linewidth=1.6,
               label=f"95% VaR A: {varA:,.1f} M")
    ax.axvline(varB, color="darkgreen", linestyle="--", linewidth=1.6,
               label=f"95% VaR B: {varB:,.1f} M")
    ax.axvline(0, color="black", linestyle=":", linewidth=1.0,
               label="Balanced budget")
    ax.set_title(f"Figure 4. Deficit distributions — {cal['level'].title()}")
    ax.set_xlabel("Fiscal deficit (PHP M, constant 2018)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig4_deficit_var_{cal['level']}.png"))
    plt.close(fig)


def fig5_revenue_fits(fiscal_level, cal, outdir):
    loc_col = "totlocsrc_raw" if "totlocsrc_raw" in fiscal_level.columns \
        else "totlocsrc"
    ira_col = "ira_raw" if "ira_raw" in fiscal_level.columns else "ira"
    loc = fiscal_level[loc_col].dropna().values
    ira = fiscal_level[ira_col].dropna().values
    if len(loc) < 5 or len(ira) < 5:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for data, mu, sig, color, label in [
        (loc, cal["mu_loc"], cal["sig_loc"], "steelblue", "Local revenue"),
        (ira, cal["mu_ira"], cal["sig_ira"], "seagreen", "IRA"),
    ]:
        ks = stats.kstest(data, "lognorm", args=(sig, 0, np.exp(mu)))
        sns.histplot(data, bins=40, stat="density", alpha=0.30,
                     color=color, ax=ax, label=f"{label} (KS p={ks.pvalue:.2f})")
        xs = np.linspace(0.5, float(np.percentile(data, 99.5)), 400)
        ax.plot(xs, stats.lognorm.pdf(xs, sig, 0, np.exp(mu)),
                color=color, linewidth=2)
    ax.set_title(f"Figure 5. Revenue fits — {cal['level'].title()}")
    ax.set_xlabel("Revenue (PHP M, constant 2018)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig5_revenue_{cal['level']}.png"))
    plt.close(fig)


def fig6_diagnostics(sim, outdir):
    ratio_A = sim["deficit_A"] / sim["total_income"]
    rho_pi_def = float(np.corrcoef(sim["pi_A"], ratio_A)[0, 1])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(sim["total_income"], bins=50, color="steelblue",
                    alpha=0.75, edgecolor="black", linewidth=0.3)
    axes[0, 0].set_title("(a) Simulated total income")
    axes[0, 1].hist(ratio_A, bins=60, color="crimson", alpha=0.75,
                    edgecolor="black", linewidth=0.3)
    axes[0, 1].axvline(INSOLVENCY_THRESHOLD, color="red", linestyle="--",
                       label=f"{INSOLVENCY_THRESHOLD:.2f} threshold")
    axes[0, 1].set_title("(b) Deficit/income ratio (Scenario A)")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].scatter(sim["pi_A"], ratio_A, alpha=0.25, s=4, color="purple")
    axes[1, 0].set_title(f"(c) Vote share vs fiscal risk (ρ={rho_pi_def:+.3f})")
    axes[1, 1].scatter(sim["total_income"], sim["deficit_A"],
                       alpha=0.25, s=4, color="darkgreen")
    axes[1, 1].axhline(0, color="black", linestyle="--")
    axes[1, 1].set_title("(d) Deficit vs income")
    fig.suptitle("Figure 6. Diagnostic plots (Scenario A, provinces)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig6_diagnostics.png"))
    plt.close(fig)


def fig7_trajectories(sim, cal, outdir, n_paths=60, seed=RANDOM_SEED + 7):
    rng = np.random.default_rng(seed)
    mu_loc, sig_loc = cal["mu_loc"], cal["sig_loc"]
    mu_ira, sig_ira = cal["mu_ira"], cal["sig_ira"]
    paths = np.zeros((n_paths, N_YEARS))
    for i in range(n_paths):
        totexp_y = float(sim["totexp_A"][i]) * np.ones(N_YEARS)
        inc_y = (rng.lognormal(mu_loc, sig_loc, N_YEARS)
                 + rng.lognormal(mu_ira, sig_ira, N_YEARS))
        paths[i] = np.cumsum(totexp_y - inc_y)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i in range(n_paths):
        ax.plot(range(1, N_YEARS + 1), paths[i],
                color="crimson", alpha=0.20, linewidth=0.9)
    ax.plot(range(1, N_YEARS + 1), paths.mean(axis=0),
            "k--", linewidth=2.4, label="Mean cumulative deficit")
    ax.axhline(0, color="navy", linestyle=":", label="Zero")
    ax.set_title(f"Figure 7. 10-year trajectories — {cal['level'].title()}")
    ax.set_xlabel("Years since baseline")
    ax.set_ylabel("Cumulative deficit (PHP M)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig7_trajectories_{cal['level']}.png"))
    plt.close(fig)


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    panel_path = os.path.join(_PROJECT_ROOT, "data", "panel.csv")
    elec_prov_path = os.path.join(_PROJECT_ROOT, "data",
                                  "election_panel_hybrid.csv")
    elec_mayor_path = os.path.join(_PROJECT_ROOT, "data",
                                   "election_panel_mayor.csv")

    print(f"[*] Loading fiscal panel: {panel_path}")
    fiscal = load_fiscal_panel(panel_path)

    # ---------- Province ----------
    print(f"\n[*] Loading province electoral panel: {elec_prov_path}")
    elec_prov = load_election_panel(elec_prov_path, "Province")
    reg_prov = build_regression_panel(fiscal, elec_prov)

    # ---------- City + Municipality ----------
    print(f"\n[*] Loading mayor electoral panel: {elec_mayor_path}")
    elec_mayor = load_election_panel(elec_mayor_path, "City+Municipality")
    reg_mayor = build_regression_panel(fiscal, elec_mayor)

    # ---------- Elasticities by level ----------
    e_prov = estimate_elasticities(reg_prov, "province")
    e_city = estimate_elasticities(reg_mayor, "city")
    e_muni = estimate_elasticities(reg_mayor, "municipality")

    results = {"province": e_prov, "city": e_city, "municipality": e_muni}

    table2 = build_three_level_elasticity_table(results)
    table2.to_csv(os.path.join(OUTPUT_DIR, "table2_three_level.csv"),
                  index=False)
    print("\n--- Table 2 (three-level elasticities) ---")
    print(table2.to_string(index=False))

    # ---------- Calibrations ----------
    cal_prov = calibrate_lognormal(fiscal, "province")
    cal_city = calibrate_lognormal(fiscal, "city")
    cal_muni = calibrate_lognormal(fiscal, "municipality")

    # ---------- Monte Carlo ----------
    print("\n[*] Running Monte Carlo (10,000 draws each) …")
    sims = {}
    if e_prov: sims["province"]     = run_monte_carlo(e_prov, cal_prov)
    if e_city: sims["city"]         = run_monte_carlo(e_city, cal_city)
    if e_muni: sims["municipality"] = run_monte_carlo(e_muni, cal_muni)

    table3 = build_three_level_summary_table(sims)
    table3.to_csv(os.path.join(OUTPUT_DIR, "table3_three_level.csv"),
                  index=False)
    print("\n--- Table 3 (three-level Monte Carlo) ---")
    print(table3.to_string(index=False))

    # ---------- Figures ----------
    print("\n[*] Generating figures …")
    fig1_empirical_pbc(fiscal, OUTPUT_DIR)
    if "province" in sims:
        fig2_tradeoff(sims["province"], e_prov, cal_prov, OUTPUT_DIR)
        fig3_heatmap(sims["province"], cal_prov, OUTPUT_DIR)
        fig4_deficit_density(sims["province"], cal_prov, OUTPUT_DIR)
        fig5_revenue_fits(
            fiscal[fiscal["lgutype"].str.contains("Province",
                                                  case=False, na=False)],
            cal_prov, OUTPUT_DIR)
        fig6_diagnostics(sims["province"], OUTPUT_DIR)
        fig7_trajectories(sims["province"], cal_prov, OUTPUT_DIR)

    # ---------- Machine-readable summary ----------
    def level_summary(e, s):
        if e is None or s is None:
            return None
        return {
            "n_obs": e["n_obs"], "n_lgu": e["n_lgu"],
            "n_cycles": e["n_cycles"], "within_r2": e["within_r2"],
            "coefficients": e["coefficients"],
            "p_values": e["p_values"],
            "A_1y_insolv": float(s["insolvent_A_1y"].mean()),
            "B_1y_insolv": float(s["insolvent_B_1y"].mean()),
            "A_mean_insolv_years": float(s["n_insolv_years_A"].mean()),
            "B_mean_insolv_years": float(s["n_insolv_years_B"].mean()),
        }

    summary = {
        "province":     level_summary(e_prov, sims.get("province")),
        "city":         level_summary(e_city, sims.get("city")),
        "municipality": level_summary(e_muni, sims.get("municipality")),
    }
    with open(os.path.join(OUTPUT_DIR, "results_summary_three_level.json"),
              "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n[✓] All outputs saved to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()