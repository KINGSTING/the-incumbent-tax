"""
Build a province-level electoral panel from OpenHalalan (NLE vote counts and
winners, 2007-2025).

The raw file reports one row per (candidate, city) pair. For governor races
each candidate therefore appears once per city in the province. We collapse
to one row per (province, cycle, candidate) by summing votes across cities,
then compute the winner, vote share, and effective number of candidates.

Adds a pi_incumbent column: the previous cycle's winner's share in the current
cycle (NaN when the previous winner did not run).
"""
import os
import re
import unicodedata
import pandas as pd
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_ROOT, "data")


# -----------------------------------------------------------------------------
# Name normalization
# -----------------------------------------------------------------------------
RENAME_MAP = {
    "COMPOSTELA VALLEY": "DAVAO DE ORO",
    "MT PROVINCE": "MOUNTAIN PROVINCE",
    "MT. PROVINCE": "MOUNTAIN PROVINCE",
    "NORTH COTABATO": "COTABATO",
    "SHARIFF KABUNSUAN": "MAGUINDANAO",
    "TAWI TAWI": "TAWI-TAWI",
    "MAGUINDANAO": "MAGUINDANAO",
}

OVERSEAS = {
    "AMERICAS", "ASIA PACIFIC", "EUROPE", "MIDDLE EAST",
    "AFRICA", "OCEANIA", "HONG KONG", "SINGAPORE",
    "SAUDI ARABIA", "UNITED ARAB EMIRATES", "QATAR", "KUWAIT",
}


def norm(name) -> str:
    if pd.isna(name):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.upper().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("MT.", "MT")
    s = s.replace(".", "")
    s = re.sub(r"\s*\(.*?\)\s*", "", s)
    return RENAME_MAP.get(s, s)


# -----------------------------------------------------------------------------
# 1. Load raw files
# -----------------------------------------------------------------------------
votes_path = os.path.join(DATA, "NLE_vote_counts_2007-2025.csv")
panel_path = os.path.join(DATA, "panel.csv")

v = pd.read_csv(votes_path, low_memory=False)
v["province_norm"] = v["province"].map(norm)
v["year"] = pd.to_numeric(v["year"], errors="coerce")
v["votes"] = pd.to_numeric(v["votes"], errors="coerce").fillna(0)

# -----------------------------------------------------------------------------
# 2. Filter to gubernatorial races in provinces
# -----------------------------------------------------------------------------
gov = v[
    (v["position"].str.upper() == "GOVERNOR")
    & (v["province_norm"] != "")
    & (~v["province_norm"].isin(OVERSEAS))
].copy()

print(f"[*] Raw governor rows:      {len(gov):,}")
print(f"[*] Distinct provinces:     {gov['province_norm'].nunique()}")
print(f"[*] Cycles:                 {sorted(gov['year'].dropna().unique().astype(int))}")

pre = gov.groupby(["province_norm", "year"]).size()
print(f"[*] Rows per province-cycle — "
      f"median {int(pre.median())}, max {int(pre.max())}")

# -----------------------------------------------------------------------------
# 3. Collapse city-level rows to candidate level
# -----------------------------------------------------------------------------
gov_cand = (
    gov.groupby(["province_norm", "year", "candidate_name"], as_index=False)
       ["votes"].sum()
)

post = gov_cand.groupby(["province_norm", "year"]).size()
print(f"[*] Candidates per province-cycle — "
      f"median {int(post.median())}, max {int(post.max())}")

# -----------------------------------------------------------------------------
# 4. Per province-cycle: totals, shares, ENC
# -----------------------------------------------------------------------------
totals = (gov_cand.groupby(["province_norm", "year"])["votes"]
                 .sum()
                 .rename("total_votes")
                 .reset_index())

gov_cand = gov_cand.merge(totals, on=["province_norm", "year"], how="left")
gov_cand["share"] = gov_cand["votes"] / gov_cand["total_votes"]
gov_cand["share_sq"] = gov_cand["share"] ** 2

enc = (
    gov_cand.groupby(["province_norm", "year"])["share_sq"]
            .sum()
            .reset_index()
            .assign(ENC_lt=lambda d: 1.0 / d["share_sq"])
            .drop(columns="share_sq")
            .rename(columns={"year": "elecyr"})
)

ncand = (
    gov_cand.groupby(["province_norm", "year"])
            .size()
            .rename("n_candidates")
            .reset_index()
            .rename(columns={"year": "elecyr"})
)

# -----------------------------------------------------------------------------
# 5. Pick the winning candidate (one row per province-cycle)
# -----------------------------------------------------------------------------
gov_cand_sorted = gov_cand.sort_values(
    ["province_norm", "year", "votes"],
    ascending=[True, True, False],
)

winners_now = (
    gov_cand_sorted.groupby(["province_norm", "year"], as_index=False)
                   .head(1)
                   .rename(columns={"candidate_name": "winner_name",
                                    "share": "pi"})
)

elec = (
    winners_now[["province_norm", "year", "winner_name",
                 "votes", "total_votes", "pi"]]
    .rename(columns={"year": "elecyr", "votes": "winner_votes",
                     "total_votes": "province_total_votes"})
    .merge(enc,   on=["province_norm", "elecyr"], how="left")
    .merge(ncand, on=["province_norm", "elecyr"], how="left")
)

# -----------------------------------------------------------------------------
# 6. Chain winners to flag incumbents
# -----------------------------------------------------------------------------
elec = elec.sort_values(["province_norm", "elecyr"]).reset_index(drop=True)
elec["prev_winner"] = elec.groupby("province_norm")["winner_name"].shift(1)
elec["prev_elecyr"] = elec.groupby("province_norm")["elecyr"].shift(1)

elec["is_incumbent"] = (
    (elec["prev_elecyr"] == elec["elecyr"] - 3)
    & (elec["winner_name"] == elec["prev_winner"])
)

# -----------------------------------------------------------------------------
# 6b. Add pi_incumbent: previous cycle's winner's share in the current cycle.
# -----------------------------------------------------------------------------
cand_shares = (gov_cand[["province_norm", "year", "candidate_name", "share"]]
               .copy())
cand_shares = cand_shares.rename(columns={"year": "elecyr",
                                          "candidate_name": "prev_winner",
                                          "share": "pi_incumbent"})

elec = elec.merge(cand_shares,
                  on=["province_norm", "elecyr", "prev_winner"],
                  how="left")

print(f"[*] pi_incumbent populated: {elec['pi_incumbent'].notna().sum():,} "
      f"of {len(elec):,} rows")

# -----------------------------------------------------------------------------
# 7. Map province names to PSGC10
# -----------------------------------------------------------------------------
panel = pd.read_csv(panel_path, low_memory=False)
panel["lgu_name_norm"] = panel["lgu_name"].map(norm)

prov_lookup = (
    panel[panel["lgu_type"].str.strip().str.lower() == "province"]
    [["lgu_name_norm", "psgc10", "lgu_name"]]
    .drop_duplicates("lgu_name_norm")
)

elec = elec.merge(prov_lookup,
                  left_on="province_norm",
                  right_on="lgu_name_norm",
                  how="left")

unmapped = sorted(elec.loc[elec["psgc10"].isna(), "province_norm"].unique())
if unmapped:
    print(f"\n[!] {len(unmapped)} provinces did not map to PSGC10:")
    for p in unmapped:
        print(f"    - {p}")
    print("\n    Panel province names near the miss:")
    for u in unmapped:
        stem = u.split()[0]
        near = prov_lookup[
            prov_lookup["lgu_name_norm"].str.contains(stem, na=False)
        ]["lgu_name_norm"].unique()
        print(f"    '{u}' → candidates: {list(near)}")
else:
    print("\n[✓] All OpenHalalan provinces mapped to PSGC10.")

# -----------------------------------------------------------------------------
# 8. Finalize and save
# -----------------------------------------------------------------------------
out = elec[[
    "psgc10", "lgu_name", "province_norm", "elecyr",
    "pi", "pi_incumbent", "ENC_lt", "n_candidates",
    "winner_name", "prev_winner", "is_incumbent",
]].copy()

out = out.dropna(subset=["psgc10", "elecyr"])
out = out.sort_values(["psgc10", "elecyr"]).reset_index(drop=True)

dups = out.duplicated(subset=["psgc10", "elecyr"]).sum()
if dups:
    print(f"[!] {dups} duplicated province-cycles — keeping first.")
    out = out.drop_duplicates(subset=["psgc10", "elecyr"], keep="first")

out_path = os.path.join(DATA, "election_panel.csv")
out.to_csv(out_path, index=False)

print(f"\n[✓] Wrote {out_path}")
print(f"    rows:                           {len(out):,}")
print(f"    provinces with electoral data:  {out['psgc10'].nunique()}")
print(f"    cycles:                         "
      f"{sorted(out['elecyr'].astype(int).unique())}")

print(f"\nπ summary:")
print(out["pi"].describe().to_string())

print(f"\nπ_incumbent summary:")
print(out["pi_incumbent"].describe().to_string())

print(f"\nENC_lt summary:")
print(out["ENC_lt"].describe().to_string())

print(f"\nIncumbent share of races: {out['is_incumbent'].mean():.3f}")