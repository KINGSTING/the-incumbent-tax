"""
Build a cpi_2018 column (base 2018 = 100) for the Philippines from the
World Bank CPI index (FP.CPI.TOTL, base 2010 = 100) and merge it into
data/panel.csv.

Source: World Development Indicators, indicator FP.CPI.TOTL
        ("Consumer price index (2010 = 100)").

Robust to: running from any cwd, minor variations in the World Bank CSV
filename, and non-numeric cells in the year columns.
"""
import os
import glob
import pandas as pd

# -----------------------------------------------------------------------------
# Paths — anchored to the project root, not the cwd
# -----------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA = os.path.join(_ROOT, "data")

PANEL_PATH = os.path.join(DATA, "panel.csv")
OUT_CPI = os.path.join(DATA, "cpi_2018.csv")

BASE_YEAR_RAW = 2010
BASE_YEAR_NEW = 2018
COUNTRY = "PHL"

# -----------------------------------------------------------------------------
# Locate the World Bank CPI index file (tolerant to whitespace/variants)
# -----------------------------------------------------------------------------
candidates = sorted(glob.glob(os.path.join(DATA, "*FP.CPI.TOTL_DS2*.csv")))
# Exclude the metadata files
candidates = [c for c in candidates if "Metadata" not in os.path.basename(c)
              and not os.path.basename(c).startswith(".~")]

if not candidates:
    raise SystemExit(
        f"No World Bank CPI index file found in {DATA}.\n"
        f"Expected a file matching '*FP.CPI.TOTL_DS2*.csv' "
        f"(e.g. 'API_FP.CPI.TOTL_DS2_en_csv_v2_253.csv').\n"
        f"Files in {DATA}:\n" + "\n".join(sorted(os.listdir(DATA)))
    )

WB_PATH = candidates[0]
print(f"[*] Using World Bank CPI file: {os.path.basename(WB_PATH)}")

# -----------------------------------------------------------------------------
# 1. Load the World Bank file (4 metadata rows, then a header)
# -----------------------------------------------------------------------------
wb = pd.read_csv(WB_PATH, skiprows=4, low_memory=False)

if "Country Code" not in wb.columns:
    raise SystemExit(
        f"Unexpected header in {WB_PATH}.\n"
        f"Columns found: {wb.columns.tolist()[:6]}..."
    )

row = wb[wb["Country Code"] == COUNTRY]
if row.empty:
    raise SystemExit(f"{COUNTRY} not found in {WB_PATH}")

row = row.iloc[0]
years = [int(c) for c in wb.columns if str(c).strip().isdigit()]

raw = pd.Series({y: row[str(y)] for y in years}, name="cpi_2010")
raw = pd.to_numeric(raw, errors="coerce").dropna()
raw.index.name = "fiscal_year"

print(f"[*] Loaded {COUNTRY} CPI ({BASE_YEAR_RAW}=100), "
      f"{raw.index.min()}–{raw.index.max()}, N = {len(raw)}")

# -----------------------------------------------------------------------------
# 2. Rebase to 2018 = 100
# -----------------------------------------------------------------------------
if BASE_YEAR_NEW not in raw.index:
    raise SystemExit(f"Base year {BASE_YEAR_NEW} missing from source series.")

base_value = raw.loc[BASE_YEAR_NEW]
cpi = (raw / base_value * 100).rename("cpi_2018")

print(f"[*] Rebased to {BASE_YEAR_NEW}=100. "
      f"Anchor value (raw {BASE_YEAR_NEW}): {base_value:.6f}")

# Print a compact preview so the user can eyeball the splice
preview = cpi.loc[[y for y in (1992, 2000, 2010, 2015, 2018, 2022, 2024)
                   if y in cpi.index]]
print("[*] Preview (cpi_2018):")
for y, v in preview.items():
    print(f"      {y}: {v:8.4f}")

# -----------------------------------------------------------------------------
# 3. Sanity check the reconstruction
# -----------------------------------------------------------------------------
implied = cpi.pct_change() * 100
raw_implied = raw.pct_change() * 100
err = (implied - raw_implied).abs().max()
print(f"[*] Rebase consistency — max |Δ inflation| = {err:.8f} pp "
      f"(should be ~0)")

# -----------------------------------------------------------------------------
# 4. Save cpi_2018.csv
# -----------------------------------------------------------------------------
out = cpi.reset_index()[["fiscal_year", "cpi_2018"]]
out = out[(out["fiscal_year"] >= 1991) & (out["fiscal_year"] <= 2025)]
out.to_csv(OUT_CPI, index=False)
print(f"[✓] Wrote {OUT_CPI} ({len(out)} rows, "
      f"{int(out['fiscal_year'].min())}–{int(out['fiscal_year'].max())})")

# -----------------------------------------------------------------------------
# 5. Merge into panel.csv
# -----------------------------------------------------------------------------
panel = pd.read_csv(PANEL_PATH, low_memory=False)
if "cpi_2018" in panel.columns:
    panel = panel.drop(columns=["cpi_2018"])

panel = panel.merge(out, on="fiscal_year", how="left")

miss = panel["cpi_2018"].isna().sum()
if miss:
    missing_years = sorted(
        panel.loc[panel["cpi_2018"].isna(), "fiscal_year"]
             .dropna().astype(int).unique().tolist())
    print(f"[!] {miss} rows in panel.csv have no CPI. "
          f"Missing fiscal years: {missing_years}")
else:
    print("[✓] All panel rows now carry a cpi_2018 value.")

fy_min = int(panel["fiscal_year"].min())
fy_max = int(panel["fiscal_year"].max())
cpi_min = panel["cpi_2018"].min()
cpi_max = panel["cpi_2018"].max()
print(f"[*] panel.csv cpi_2018 range: {cpi_min:.3f}–{cpi_max:.3f} "
      f"(fiscal years {fy_min}–{fy_max})")

panel.to_csv(PANEL_PATH, index=False)
print(f"[✓] Updated {PANEL_PATH}")
print(f"\nNext: run 'python src/main.py' from the project root. "
      f"You should see '[✓] Deflation: cpi_2018 column present — "
      f"using real 2018 PHP.'")