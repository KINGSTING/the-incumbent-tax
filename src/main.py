import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm, norm
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# GLOBAL SETTINGS & OUTPUT DIRECTORY
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

OUTPUT_DIR = os.path.join("..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. DATA CLEANING
# -----------------------------------------------------------------------------
def load_and_clean_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    
    numeric_cols = [
        'totlocsrc', 'ira', 'govexp', 'pubwelf', 'educexp', 'healthexp', 
        'totexp', 'lgusincome', 'tottax', 'votes', 'totvot', 'pi', 'ENC_lt',
        'laborexp', 'housingexp', 'socservexp', 'econdevexp'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    sectoral_cols = ['educexp', 'healthexp', 'pubwelf', 'govexp', 'laborexp', 
                     'housingexp', 'socservexp', 'econdevexp']
    for col in sectoral_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            
    for col in ['totlocsrc', 'ira', 'totexp']:
        if col in df.columns:
            q99 = df[col].quantile(0.99)
            df[col] = np.where(df[col] > q99, q99, df[col])
            
    return df

# -----------------------------------------------------------------------------
# 2. ESTIMATE ELASTICITIES (with adjusted intercept)
# -----------------------------------------------------------------------------
def estimate_elasticities(df):
    """
    Estimate coefficients using LGU fixed effects and compute an adjusted intercept
    such that prediction at mean predictors equals mean_pi.
    """
    model_df = df[['lgu', 'pi', 'pubwelf', 'educexp', 'healthexp', 'ENC_lt']].copy()
    model_df = model_df.dropna()
    if model_df.empty:
        raise ValueError("Not enough data to estimate elasticities.")
    
    # Overall mean pi
    mean_pi = model_df['pi'].mean()
    
    # Log transform
    model_df['log_pub'] = np.log(model_df['pubwelf'] + 1)
    model_df['log_edu'] = np.log(model_df['educexp'] + 1)
    model_df['log_hea'] = np.log(model_df['healthexp'] + 1)
    
    # Compute means of predictors (for intercept adjustment)
    mean_log_pub = model_df['log_pub'].mean()
    mean_log_edu = model_df['log_edu'].mean()
    mean_log_hea = model_df['log_hea'].mean()
    mean_enc = model_df['ENC_lt'].mean()
    
    # Demean within LGU (fixed effects)
    lgu_means = model_df.groupby('lgu').mean()
    model_demeaned = model_df.set_index('lgu') - lgu_means
    model_demeaned = model_demeaned.reset_index(drop=True)
    
    # OLS on demeaned data (no intercept)
    X = model_demeaned[['log_pub', 'log_edu', 'log_hea', 'ENC_lt']]
    y = model_demeaned['pi']
    X = sm.add_constant(X)  # constant will be zero
    model = sm.OLS(y, X, missing='drop').fit()
    
    b_pub = model.params.get('log_pub', 0.0)
    b_edu = model.params.get('log_edu', 0.0)
    b_hea = model.params.get('log_hea', 0.0)
    b_enc = model.params.get('ENC_lt', 0.0)
    
    # Adjusted intercept: mean_pi - (b_pub*mean_log_pub + b_edu*mean_log_edu + b_hea*mean_log_hea + b_enc*mean_enc)
    intercept_adj = mean_pi - (b_pub * mean_log_pub + b_edu * mean_log_edu + b_hea * mean_log_hea + b_enc * mean_enc)
    
    coefs = {
        'b_pub': b_pub,
        'b_edu': b_edu,
        'b_hea': b_hea,
        'b_enc': b_enc,
        'resid_std': model.resid.std(),
        'intercept': intercept_adj
    }
    print("[*] Estimated elasticities:", coefs)
    return coefs

# -----------------------------------------------------------------------------
# 3. MONTE CARLO SIMULATION (with adjusted intercept & cumulative tracker)
# -----------------------------------------------------------------------------
def run_empirical_monte_carlo(df, elasticities, n_simulations=10000, seed=42):
    np.random.seed(seed)
    
    # Extract revenue moments
    mean_loc = df['totlocsrc'].mean() if 'totlocsrc' in df.columns else 150.0
    std_loc = df['totlocsrc'].std() if 'totlocsrc' in df.columns else 35.0
    mean_ira = df['ira'].mean() if 'ira' in df.columns else 450.0
    std_ira = df['ira'].std() if 'ira' in df.columns else 40.0
    
    rev_corr = df[['totlocsrc', 'ira']].corr().iloc[0, 1] if {'totlocsrc', 'ira'}.issubset(df.columns) else 0.45
    mean_enc = df['ENC_lt'].mean() if 'ENC_lt' in df.columns and not df['ENC_lt'].isna().all() else 2.5
    
    # Fallback
    mean_loc = 150.0 if np.isnan(mean_loc) else mean_loc
    std_loc = 35.0 if np.isnan(std_loc) else std_loc
    mean_ira = 450.0 if np.isnan(mean_ira) else mean_ira
    std_ira = 40.0 if np.isnan(std_ira) else std_ira
    rev_corr = 0.45 if np.isnan(rev_corr) else rev_corr
    mean_enc = 2.5 if np.isnan(mean_enc) else mean_enc

    # Lognormal parameters
    def norm_to_lognorm(mean, std):
        var = std ** 2
        sig2 = np.log(1 + (var / (mean ** 2)))
        mu = np.log(mean) - (sig2 / 2)
        return mu, np.sqrt(sig2)

    mu_loc, sig_loc = norm_to_lognorm(mean_loc, std_loc)
    mu_ira, sig_ira = norm_to_lognorm(mean_ira, std_ira)

    # Cholesky for correlation
    cov_matrix = np.array([
        [sig_loc**2, rev_corr * sig_loc * sig_ira],
        [rev_corr * sig_loc * sig_ira, sig_ira**2]
    ])
    L = np.linalg.cholesky(cov_matrix)
    uncorrelated = np.random.normal(size=(2, n_simulations))
    correlated = np.dot(L, uncorrelated)

    sim_loc = np.exp(mu_loc + correlated[0, :])
    sim_ira = np.exp(mu_ira + correlated[1, :])
    total_income = sim_loc + sim_ira
    baseline_budget = mean_loc + mean_ira

    # Unpack elasticities
    b_pub = elasticities['b_pub']
    b_edu = elasticities['b_edu']
    b_hea = elasticities['b_hea']
    b_enc = elasticities['b_enc']
    resid_std = elasticities['resid_std']
    intercept = elasticities['intercept']   # adjusted intercept

    # Scenario A: Clientelistic Surge
    surge_A = np.random.uniform(1.15, 1.30, n_simulations)
    totexp_A = baseline_budget * surge_A
    pubwelf_A = totexp_A * 0.35
    govexp_A = totexp_A * 0.35
    educexp_A = totexp_A * 0.15
    healthexp_A = totexp_A * 0.15

    # Scenario B: Balanced Portfolio
    surge_B = np.random.uniform(0.95, 1.02, n_simulations)
    totexp_B = baseline_budget * surge_B
    pubwelf_B = totexp_B * 0.10
    govexp_B = totexp_B * 0.25
    educexp_B = totexp_B * 0.35
    healthexp_B = totexp_B * 0.30

    # Voter model using adjusted intercept + residual noise
    def calc_pi(pub, edu, hea, enc):
        log_pub = np.log(pub + 1)
        log_edu = np.log(edu + 1)
        log_hea = np.log(hea + 1)
        pi_hat = (intercept
                  + b_pub * log_pub
                  + b_edu * log_edu
                  + b_hea * log_hea
                  + b_enc * enc)
        noise = np.random.normal(0, resid_std, len(pub))
        return np.clip(pi_hat + noise, 0.0, 1.0)

    pi_A = calc_pi(pubwelf_A, educexp_A, healthexp_A, mean_enc)
    pi_B = calc_pi(pubwelf_B, educexp_B, healthexp_B, mean_enc)

    # Single-period deficits
    deficit_A = totexp_A - total_income
    deficit_B = totexp_B - total_income
    insolvent_A = (deficit_A / total_income) > 0.05
    insolvent_B = (deficit_B / total_income) > 0.05

    # ---- CUMULATIVE DEFICIT TRACKER (10 years) for both scenarios ----
    n_years = 10
    cum_def_A = np.zeros(n_simulations)
    cum_inc_A = np.zeros(n_simulations)
    cum_def_B = np.zeros(n_simulations)
    cum_inc_B = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Simulate 10 years of income (same for both)
        inc_years = np.exp(mu_loc + np.random.normal(0, sig_loc, n_years)) + \
                    np.exp(mu_ira + np.random.normal(0, sig_ira, n_years))
        # Scenario A: constant surge_A[i]
        exp_A = baseline_budget * surge_A[i]
        cum_def_A[i] = np.sum(exp_A - inc_years)
        cum_inc_A[i] = np.sum(inc_years)
        # Scenario B: constant surge_B[i]
        exp_B = baseline_budget * surge_B[i]
        cum_def_B[i] = np.sum(exp_B - inc_years)
        cum_inc_B[i] = np.sum(inc_years)

    cum_insolvent_A = (cum_def_A / cum_inc_A) > 0.05
    cum_insolvent_B = (cum_def_B / cum_inc_B) > 0.05

    sim_data = {
        'totexp_A': totexp_A, 'totexp_B': totexp_B,
        'deficit_A': deficit_A, 'deficit_B': deficit_B,
        'insolvent_A': insolvent_A, 'insolvent_B': insolvent_B,
        'pi_A': pi_A, 'pi_B': pi_B,
        'sim_loc': sim_loc, 'sim_ira': sim_ira,
        'total_income': total_income,
        'surge_A': surge_A,
        'surge_B': surge_B,
        'cum_def_A': cum_def_A, 'cum_inc_A': cum_inc_A,
        'cum_insolvent_A': cum_insolvent_A,
        'cum_def_B': cum_def_B, 'cum_inc_B': cum_inc_B,
        'cum_insolvent_B': cum_insolvent_B,
    }
    return sim_data

# -----------------------------------------------------------------------------
# 4. VISUALIZATION GENERATOR (7 figures)
# -----------------------------------------------------------------------------
def generate_visualizations(df, sim, elasticities):
    # ---- Figure 1: Empirical PBC ----
    generate_figure1(df, output_dir=OUTPUT_DIR)

    # ---- Figure 2: Deficit distribution & VaR ----
    plt.figure(figsize=(9, 5))
    sns.kdeplot(sim['deficit_A'], fill=True, color='crimson', label='Scenario A', alpha=0.4)
    sns.kdeplot(sim['deficit_B'], fill=True, color='teal', label='Scenario B', alpha=0.4)
    var_95_A = np.percentile(sim['deficit_A'], 95)
    var_95_B = np.percentile(sim['deficit_B'], 95)
    plt.axvline(var_95_A, color='darkred', linestyle='--', linewidth=2, label=f'95% VaR A: {var_95_A:.1f}M')
    plt.axvline(var_95_B, color='darkgreen', linestyle='--', linewidth=2, label=f'95% VaR B: {var_95_B:.1f}M')
    plt.axvline(0, color='black', linestyle=':', label='Balanced Budget')
    plt.title("Deficit Distributions & Tail Risk")
    plt.xlabel("Fiscal Deficit (PHP M)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_deficit_var_density.png"), dpi=300)
    plt.close()

    # ---- Figure 3: Trade-off frontier (using empirical elasticities) ----
    surges = np.linspace(1.0, 1.45, 20)
    vote_shares = []
    insolvency_rates = []
    base_budget = sim['total_income'].mean()
    # Use empirical coefficients from elasticities (including adjusted intercept)
    b_pub = elasticities['b_pub']
    b_edu = elasticities['b_edu']
    b_hea = elasticities['b_hea']
    b_enc = elasticities['b_enc']
    intercept = elasticities['intercept']

    for s in surges:
        tot_exp = base_budget * s
        pub = tot_exp * 0.35
        edu = tot_exp * 0.15
        hea = tot_exp * 0.15
        # Compute vote share using the same model
        pi_hat = intercept + b_pub * np.log(pub + 1) + b_edu * np.log(edu + 1) + b_hea * np.log(hea + 1) + b_enc * 2.5
        pi = np.clip(pi_hat, 0, 1)
        # Insolvency risk
        defic = tot_exp - sim['total_income']
        insolv = np.mean((defic / sim['total_income']) > 0.05)
        vote_shares.append(pi)
        insolvency_rates.append(insolv * 100)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(surges, vote_shares, 'b-o', label='Vote Share')
    ax1.set_xlabel('Surge Multiplier')
    ax1.set_ylabel('Vote Share', color='b')
    ax2 = ax1.twinx()
    ax2.plot(surges, insolvency_rates, 'r--s', label='Insolvency Risk')
    ax2.set_ylabel('Insolvency Probability (%)', color='r')
    plt.title("Trade-off: Vote Share vs Insolvency Risk (Empirical Elasticities)")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_patronage_tradeoff_frontier.png"), dpi=300)
    plt.close()

    # ---- Figure 4: Heatmap ----
    enc_grid = np.linspace(1.0, 4.0, 5)
    surge_grid = np.linspace(1.0, 1.35, 5)
    heatmap_matrix = np.zeros((len(enc_grid), len(surge_grid)))
    for i, enc in enumerate(enc_grid):
        for j, s in enumerate(surge_grid):
            exp = base_budget * s
            defic = exp - sim['total_income']
            insolv_rate = np.mean((defic / sim['total_income']) > 0.05)
            heatmap_matrix[i, j] = insolv_rate * 100
    plt.figure(figsize=(7, 5))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=[f"{s:.2f}x" for s in surge_grid],
                yticklabels=[f"{e:.1f}" for e in enc_grid])
    plt.title("Insolvency Risk (%) by ENC & Surge")
    plt.xlabel("Surge Multiplier")
    plt.ylabel("ENC_lt")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_competition_risk_heatmap.png"), dpi=300)
    plt.close()

    # ---- Figure 5: Revenue distributions ----
    plt.figure(figsize=(8, 5))
    sns.histplot(sim['sim_loc'], color='blue', label='Local Revenue', kde=True, stat="density", bins=30, alpha=0.3)
    sns.histplot(sim['sim_ira'], color='green', label='IRA', kde=True, stat="density", bins=30, alpha=0.3)
    plt.title("Lognormal Revenue Fits")
    plt.xlabel("Revenue (PHP M)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_revenue_distributions.png"), dpi=300)
    plt.close()

    # ---- Figure 6: 10-year stochastic trajectories ----
    n_years = 10
    n_paths = 50
    stochastic_paths_A = np.zeros((n_paths, n_years))
    for i in range(n_paths):
        path_inc = np.random.normal(base_budget, sim['total_income'].std(), n_years)
        path_exp = base_budget * np.random.uniform(1.15, 1.25, n_years)
        stochastic_paths_A[i, :] = np.cumsum(path_exp - path_inc)
    plt.figure(figsize=(8, 5))
    for i in range(n_paths):
        plt.plot(range(1, n_years+1), stochastic_paths_A[i, :], color='crimson', alpha=0.25)
    plt.plot(range(1, n_years+1), np.mean(stochastic_paths_A, axis=0), 'k--', linewidth=3, label='Mean Deficit')
    plt.axhline(0, color='blue', linestyle=':', label='Zero')
    plt.title("10-Year Cumulative Deficit (Clientelistic Path)")
    plt.xlabel("Years")
    plt.ylabel("Cumulative Deficit (PHP M)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig6_10yr_stochastic_trajectories.png"), dpi=300)
    plt.close()

    # ---- Figure 7: DIAGNOSTIC PLOTS ----
    fig7, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].hist(sim['total_income'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0,0].set_title("Simulated Total Income Distribution")
    axes[0,0].set_xlabel("Income (PHP M)")
    axes[0,0].set_ylabel("Frequency")

    ratio_A = sim['deficit_A'] / sim['total_income']
    axes[0,1].hist(ratio_A, bins=50, color='crimson', alpha=0.7, edgecolor='black')
    axes[0,1].axvline(0, color='black', linestyle='--', label='Balanced')
    axes[0,1].axvline(0.05, color='red', linestyle='--', label='5% Threshold')
    axes[0,1].set_title("Deficit / Income Ratio (Scenario A)")
    axes[0,1].set_xlabel("Ratio")
    axes[0,1].set_ylabel("Frequency")
    axes[0,1].legend()

    axes[1,0].scatter(sim['pi_A'], sim['deficit_A'] / sim['total_income'], alpha=0.3, color='purple')
    axes[1,0].set_xlabel("Vote Share (π)")
    axes[1,0].set_ylabel("Deficit / Income Ratio")
    axes[1,0].set_title("Vote Share vs Fiscal Risk (Scenario A)")

    axes[1,1].scatter(sim['total_income'], sim['deficit_A'], alpha=0.3, color='darkgreen')
    axes[1,1].set_xlabel("Total Income (PHP M)")
    axes[1,1].set_ylabel("Deficit (PHP M)")
    axes[1,1].axhline(0, color='black', linestyle='--')
    axes[1,1].set_title("Deficit vs Income")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig7_diagnostics.png"), dpi=300)
    plt.close()
    print("[✓] Diagnostic plots saved as fig7_diagnostics.png")

# -----------------------------------------------------------------------------
# 5. Figure 1 generator
# -----------------------------------------------------------------------------
def generate_figure1(df, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    if 'yearb4elec' in df.columns:
        pbc_col = 'yearb4elec'
    elif 'yrb4' in df.columns:
        pbc_col = 'yrb4'
    elif 'elecyr' in df.columns and 'year' in df.columns:
        df['pbc_indicator'] = ((df['year'] + 1) == df['elecyr']).astype(int)
        pbc_col = 'pbc_indicator'
    elif 'year' in df.columns:
        election_years = [1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016, 2019, 2022]
        df['pbc_indicator'] = df['year'].isin([y - 1 for y in election_years]).astype(int)
        pbc_col = 'pbc_indicator'
    else:
        df['pbc_indicator'] = 0
        pbc_col = 'pbc_indicator'
        print("[!] No year column; assuming no pre-election years.")
    
    spending_map = {
        'pubwelf': 'Public Welfare',
        'govexp': 'General Admin',
        'educexp': 'Education',
        'healthexp': 'Health'
    }
    available_cols = [c for c in spending_map.keys() if c in df.columns]
    if not available_cols:
        print("[!] No expenditure columns; creating placeholder.")
        fig, ax = plt.subplots(figsize=(9,5.5), dpi=300)
        ax.text(0.5, 0.5, "No expenditure data", ha='center', va='center')
        ax.set_title("Empirical PBC – Data Missing")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig1_empirical_pbc.png"))
        plt.close(fig)
        return
    
    for col in available_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    pbc_summary = df.groupby(pbc_col)[available_cols].mean().reset_index()
    pbc_summary[pbc_col] = pbc_summary[pbc_col].map({0: 'Off-Election', 1: 'Pre-Election'})
    pbc_summary = pbc_summary.set_index(pbc_col)
    
    if pbc_summary.max().max() > 1000:
        pbc_summary_plot = pbc_summary / 1e6
        y_label = "Mean Expenditure (PHP M)"
    else:
        pbc_summary_plot = pbc_summary
        y_label = "Mean Expenditure (PHP)"
    
    fig, ax = plt.subplots(figsize=(9,5.5), dpi=300)
    pbc_summary_plot.plot(kind='bar', ax=ax, width=0.6, color=['#d9534f','#f0ad4e','#0275d8','#5cb85c'][:len(available_cols)])
    ax.set_title("Empirical Political Budget Cycle")
    ax.set_ylabel(y_label)
    ax.set_xlabel("Electoral Cycle Phase")
    ax.set_xticklabels(pbc_summary_plot.index, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend([spending_map[c] for c in available_cols])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_empirical_pbc.png"))
    plt.close()
    print("[✓] Figure 1 saved.")

# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    data_path = os.path.join("..", "data", "COMELEC_fiscal+electoral_data_July 2025.xlsx")
    print(f"[*] Loading data from: {data_path}")
    df = load_and_clean_data(data_path)
    
    print("[*] Estimating voter elasticities from data...")
    elasticities = estimate_elasticities(df)
    
    print("[*] Running 10,000 Monte Carlo simulations...")
    sim_results = run_empirical_monte_carlo(df, elasticities, n_simulations=10000)
    
    print("[*] Generating 7 publication-ready figures...")
    generate_visualizations(df, sim_results, elasticities)
    
    # Summary table
    print("\n" + "="*70)
    print("MONTE CARLO SIMULATION SUMMARY RESULTS")
    print("="*70)
    results_summary = pd.DataFrame({
        'Metric': [
            'Expected Total Expenditure (PHP M)',
            'Expected Deficit / Surplus (PHP M)',
            'Probability of Fiscal Insolvency (>5% Deficit)',
            'Value at Risk (95% VaR Deficit, PHP M)',
            'Mean Expected Vote Share (pi)',
            'Vote Share Volatility (Std Dev)',
            'Probability of 10-Year Cumulative Insolvency'
        ],
        'Clientelistic Surge (A)': [
            np.mean(sim_results['totexp_A']),
            np.mean(sim_results['deficit_A']),
            np.mean(sim_results['insolvent_A']),
            np.percentile(sim_results['deficit_A'], 95),
            np.mean(sim_results['pi_A']),
            np.std(sim_results['pi_A']),
            np.mean(sim_results['cum_insolvent_A'])
        ],
        'Balanced Portfolio (B)': [
            np.mean(sim_results['totexp_B']),
            np.mean(sim_results['deficit_B']),
            np.mean(sim_results['insolvent_B']),
            np.percentile(sim_results['deficit_B'], 95),
            np.mean(sim_results['pi_B']),
            np.std(sim_results['pi_B']),
            np.mean(sim_results['cum_insolvent_B'])
        ]
    })
    print(results_summary.to_string(index=False))
    print("="*70)
    print(f"[✓] All outputs saved to '{OUTPUT_DIR}'.")