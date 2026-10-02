"""
Build a hybrid province-cycle electoral panel.

  • Cycles 1992–2004: from the COMELEC extract (original submission source).
  • Cycles 2007–2025: from OpenHalalan (cleaner, publicly verifiable).
  • Overlapping cycles (2007–2022): both sources available — we use OpenHalalan
    for the regression and report the cross-source correlation as a validation.

Output: data/election_panel_hybrid.csv with columns
    psgc10 | lgu_name | elecyr | pi | ENC_lt | n_candidates | source
"""
import os
import re
import unicodedata
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_ROOT, "data")

RENAME_MAP = {
    "COMPOSTELA VALLEY": "DAVAO DE ORO",
    "MT PROVINCE": "MOUNTAIN PROVINCE",
    "MT. PROVINCE": "MOUNTAIN PROVINCE",
    "NORTH COTABATO": "COTABATO",
    "SHARIFF KABUNSUAN": "MAGUINDANAO",
    "TAWI TAWI": "TAWI-TAWI",
}

def norm(name) -> str:
    if pd.isna(name):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.upper().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("MT.", "MT").replace(".", "")
    s = re.sub(r"\s*\(.*?\)\s*", "", s)
    return RENAME_MAP.get(s, s)


# -----------------------------------------------------------------------------
# 1. Pre-2007 panel from the COMELEC extract
# -----------------------------------------------------------------------------
print("[*] Loading COMELEC extract …")
old = pd.read_excel(os.path.join(DATA, "COMELEC_fiscal+electoral_data_July 2025.xlsx"))
old["lgutype_clean"] = old["lgutype"].astype(str).str.strip().str.lower()
old["elecyr"] = pd.to_numeric(old["elecyr"], errors="coerce")
old["pi"] = pd.to_numeric(old["pi"], errors="coerce")
old["ENC_lt"] = pd.to_numeric(old["ENC_lt"], errors="coerce")
old["lgu_key"] = old["lgu"].map(norm)

# Restrict to provinces, cycles ≤ 2004, and drop sentinel rows
prov = old[(old["lgutype_clean"] == "province")
           & (old["elecyr"] <= 2004)
           & old["pi"].notna() & old["ENC_lt"].notna()]
prov = prov[(prov["pi"] > 0) & (prov["pi"] <= 1)
            & (prov["ENC_lt"] >= 1) & (prov["ENC_lt"] <= 20)]

pre = prov[["lgu_key", "elecyr", "pi", "ENC_lt"]].copy()
pre["source"] = "COMELEC"

# n_candidates not available in the old extract — carry NaN
pre["n_candidates"] = np.nan
print(f"[*] Pre-2007 rows kept: {len(pre):,} "
      f"({pre['elecyr'].nunique()} cycles, "
      f"{pre['lgu_key'].nunique()} provinces)")


# -----------------------------------------------------------------------------
# 2. Post-2007 panel from OpenHalalan
# -----------------------------------------------------------------------------
print("\n[*] Loading OpenHalalan panel …")
oh = pd.read_csv(os.path.join(DATA, "election_panel.csv"))
oh["elecyr"] = pd.to_numeric(oh["elecyr"], errors="coerce")
oh["lgu_key"] = oh["province_norm"].map(norm)
post = oh[["lgu_key", "elecyr", "pi", "ENC_lt", "n_candidates"]].copy()
post["source"] = "OpenHalalan"
print(f"[*] Post-2007 rows kept: {len(post):,} "
      f"({post['elecyr'].nunique()} cycles, "
      f"{post['lgu_key'].nunique()} provinces)")


# -----------------------------------------------------------------------------
# 3. Combine and map names → PSGC10
# -----------------------------------------------------------------------------
combined = pd.concat([pre, post], ignore_index=True)
combined = combined.dropna(subset=["lgu_key", "elecyr"])
combined["elecyr"] = combined["elecyr"].astype(int)

panel = pd.read_csv(os.path.join(DATA, "panel.csv"), low_memory=False)
panel["lgu_name_norm"] = panel["lgu_name"].map(norm)

# Manual patch for Maguindanao pre-2019: parent province PSGC
mag_parent = panel[panel["lgu_name"].str.strip().str.upper() == "MAGUINDANAO"]
mag_psgc = mag_parent["psgc10"].iloc[0] if not mag_parent.empty else None

lookup = (panel[panel["lgu_type"].str.strip().str.lower() == "province"]
          [["lgu_name_norm", "psgc10", "lgu_name"]]
          .drop_duplicates("lgu_name_norm"))

combined = combined.merge(lookup, left_on="lgu_key",
                          right_on="lgu_name_norm", how="left")

# Patch missing Maguindanao
if mag_psgc is not None:
    mask = (combined["lgu_key"] == "MAGUINDANAO") & combined["psgc10"].isna()
    combined.loc[mask, "psgc10"] = mag_psgc
    combined.loc[mask, "lgu_name"] = "Maguindanao"

# Drop post-2019 Maguindanao (split into del Norte / del Sur)
combined = combined[~((combined["lgu_key"] == "MAGUINDANAO")
                      & (combined["elecyr"] >= 2019))]

unmapped = sorted(combined.loc[combined["psgc10"].isna(), "lgu_key"].unique())
if unmapped:
    print(f"\n[!] {len(unmapped)} provinces still unmapped: {unmapped}")
else:
    print("\n[✓] All provinces mapped to PSGC10.")

combined = combined.dropna(subset=["psgc10"])
combined["psgc10"] = (combined["psgc10"].astype(str).str.strip()
                      .str.replace(r"\.0$", "", regex=True).str.zfill(10))

# Exactly one row per (province, cycle) — OpenHalalan wins when both present
combined = (combined.sort_values(["psgc10", "elecyr",
                                  "source"],
                                 key=lambda s: s.map({"OpenHalalan": 0,
                                                      "COMELEC": 1})
                                 if s.name == "source" else s)
                     .drop_duplicates(["psgc10", "elecyr"], keep="first"))

out = combined[["psgc10", "lgu_name", "elecyr",
                "pi", "ENC_lt", "n_candidates", "source"]]
out = out.sort_values(["psgc10", "elecyr"]).reset_index(drop=True)

out_path = os.path.join(DATA, "election_panel_hybrid.csv")
out.to_csv(out_path, index=False)

print(f"\n[✓] Wrote {out_path}")
print(f"    rows:                     {len(out):,}")
print(f"    provinces:                {out['psgc10'].nunique()}")
print(f"    cycles:                   {sorted(out['elecyr'].unique())}")
print(f"    rows by source:\n{out['source'].value_counts().to_string()}")


# -----------------------------------------------------------------------------
# 4. Cross-source validation on overlapping cycles
# -----------------------------------------------------------------------------
print("\n[*] Cross-source validation on overlapping cycles (2007–2022) …")
overlap = (pre.merge(post, on=["lgu_key", "elecyr"],
                     suffixes=("_comelec", "_oh"), how="inner"))
if len(overlap) > 10:
    rho = overlap[["pi_comelec", "pi_oh"]].corr().iloc[0, 1]
    mad = (overlap["pi_comelec"] - overlap["pi_oh"]).abs().mean()
    print(f"    N overlapping province-cycles: {len(overlap):,}")
    print(f"    Pearson r (π_COMELEC vs π_OpenHalalan): {rho:.4f}")
    print(f"    Mean absolute difference: {mad:.4f}")
else:
    print("    Insufficient overlapping rows to validate.")

print("\nπ summary (hybrid panel):")
print(out["pi"].describe().to_string())
print("\nENC_lt summary:")
print(out["ENC_lt"].describe().to_string())