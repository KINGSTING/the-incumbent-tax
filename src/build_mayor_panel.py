"""
Build a city- and municipality-level electoral panel from OpenHalalan's
mayoral vote counts (2007-2025), keyed to PSGC10.

Matching stages, applied in order:
  A. (province, city) exact on normalized names.
  B. (city) exact, province-agnostic.
  C. Manual override table.
  D. Token-subset, province-restricted (bidirectional).
  E. Token-subset, province-agnostic, cities only (for independent cities).
  F. Drop SGA barangays and any residual unmatched rows.

Adds a pi_incumbent column: previous cycle's winner's share in the current
cycle (NaN when the previous winner did not run).
"""
import os
import re
import unicodedata
import numpy as np
import pandas as pd

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
    "NCR": "METRO MANILA",
    "NATIONAL CAPITAL REGION": "METRO MANILA",
    "NORZAGRAY": "NORZAGARAY",
    "POZZORUBIO": "POZORRUBIO",
    "BROOKES POINT": "BROOKE S POINT",
}

OVERSEAS = {
    "AMERICAS", "ASIA PACIFIC", "EUROPE", "MIDDLE EAST",
    "AFRICA", "OCEANIA", "HONG KONG", "SINGAPORE",
    "SAUDI ARABIA", "UNITED ARAB EMIRATES", "QATAR", "KUWAIT",
}

SGA_DROP = {
    "KADAYANGAN", "KAPALAWAN", "LIGAWASAN", "MALIDEGAO",
    "NABALAWAG", "OLD KAABAKAN", "PAHAMUDDIN", "TUGUNAN",
}


def norm(name) -> str:
    if pd.isna(name):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.upper().strip()
    s = re.sub(r"^CITY OF ", "", s)
    s = re.sub(r"^MUNICIPALITY OF ", "", s)
    s = re.sub(r"^MUN OF ", "", s)
    s = re.sub(r"\s+CITY$", "", s)
    s = re.sub(r"\s+MUNICIPALITY$", "", s)
    s = re.sub(r"\s+MUN$", "", s)
    s = s.replace("MT.", "MT")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return RENAME_MAP.get(s, s)


def tokens(name: str) -> set:
    return set(norm(name).split())


# -----------------------------------------------------------------------------
# Manual overrides (OH name → panel name; both normalized)
# -----------------------------------------------------------------------------
MANUAL_OVERRIDES = {
    "DARAGA LOCSIN": "DARAGA",
    "LICUAN BAAY LICUAN": "LICUAN BAAY",
    "TOBIAS FORNIER DAO": "TOBIAS FORNIER",
    "PRES CARLOS P GARCIA PITOGO": "PRES CARLOS P GARCIA",
    "SAGBAYAN BORJA": "SAGBAYAN",
    "BALAGTAS BIGAA": "BALAGTAS",
    "SASMUAN SEXMOAN": "SASMUAN",
    "SAN LORENZO RUIZ IMELDA": "SAN LORENZO RUIZ",
    "PRESENTACION PARUBCAN": "PRESENTACION",
    "PANGANIBAN PAYO": "PANGANIBAN",
    "SAN ANDRES CALOLBON": "SAN ANDRES",
    "MENDEZ MENDEZ NUNEZ": "MENDEZ",
    "SAN FRANCISCO AURORA": "SAN FRANCISCO",
    "RODRIGUEZ MONTALBAN": "RODRIGUEZ",
    "SANTA MARIA IMELDA": "SANTA MARIA",
    "PARANAS WRIGHT": "PARANAS",
    "SAN JUAN CABALIAN": "SAN JUAN",
    "AMLAN AYUQUITAN": "AMLAN",
    "BINDOY PAYABON": "BINDOY",
    "VALENCIA LUZURRIAGA": "VALENCIA",
    "BAYAWAN CITY TULONG": "BAYAWAN",
    "GABALDON BITULOK AND SABANI": "GABALDON",
    "GENERAL TINIO PAPAYA": "GENERAL TINIO",
    "DELFIN ALBANO MAGSAYSAY": "DELFIN ALBANO",
    "RIZAL LIWAN": "RIZAL",
    "ALFONSO LISTA POTIA": "ALFONSO LISTA",
    "BANNA ESPIRITU": "BANNA",
    "SULTAN NAGA DIMAPORO KAROMATAN": "SULTAN NAGA DIMAPORO",
    "AMAI MANABILANG BUMBARAN": "AMAI MANABILANG",
    "BACOLOD KALAWI BACOLOD GRANDE": "BACOLOD KALAWI",
    "BALINDONG WATU": "BALINDONG",
    "LUMBA BAYABAO MAGUING": "LUMBA BAYABAO",
    "PAGAYAWAN TATARIKAN": "PAGAYAWAN",
    "PICONG SULTAN GUMANDER": "PICONG",
    "POONA BAYABAO GATA": "POONA BAYABAO",
    "JAVIER BUGHO": "JAVIER",
    "DATU ODIN SINSUAT DINAIG": "DATU ODIN SINSUAT",
    "KABUNTALAN TUMBAO": "KABUNTALAN",
    "SHARIFF AGUAK MAGANOY": "SHARIFF AGUAK",
    "SULTAN KUDARAT NULING": "SULTAN KUDARAT",
    "SULTAN SA BARONGIS LAMBAYONG": "SULTAN SA BARONGIS",
    "PIO V CORPUZ LIMBUHAN": "PIO V CORPUZ",
    "DON VICTORIANO CHIONGBIAN DON MARIANO MARCOS": "DON VICTORIANO CHIONGBIAN",
    "ENRIQUE B MAGALONA SARAVIA": "ENRIQUE B MAGALONA",
    "MOISES PADILLA MAGALLON": "MOISES PADILLA",
    "HINOBA AN ASIA": "HINOBA AN",
    "LAMBAYONG MARIANO MARCOS": "LAMBAYONG",
    "HADJI PANGLIMA TAHIL MARUNGGAS": "HADJI PANGLIMA TAHIL",
    "PANGLIMA ESTINO NEW PANAMAO": "PANGLIMA ESTINO",
    "SAN FRANCISCO ANAO AON": "SAN FRANCISCO",
    "SANTA MONICA SAPAO": "SANTA MONICA",
    "MAPUN CAGAYAN DE TAWI TAWI": "MAPUN",
    "PANGLIMA SUGALA BALIMBING": "PANGLIMA SUGALA",
    "JOSE DALMAN PONOT": "JOSE DALMAN",
    "PINAN NEW PINAN": "PINAN",
    "RAMON MAGSAYSAY LIARGO": "RAMON MAGSAYSAY",
    "SOMINOT DON MARIANO MARCOS": "SOMINOT",
    "BACUNGAN LEON T POSTIGO": "LEON B POSTIGO",
    "BULALACAO SAN PEDRO": "BULALACAO",
    "EL NIDO BACUIT": "EL NIDO",
    "RIZAL MARCOS": "RIZAL",
    "LAPU LAPU CITY OPON": "LAPU LAPU",
    "GENERAL SANTOS CITY DADIANGAS": "GENERAL SANTOS",
    "JOSE ABAD SANTOS TRINIDAD": "JOSE ABAD SANTOS",
    "CALANASAN BAYAG": "CALANASAN",
    "SANTO NINO FAIRE": "SANTO NINO",
    "QUIRINO ANGKAKI": "QUIRINO",
    "SALCEDO BAUGEN": "SALCEDO",
    "SAN JUAN LAPOG": "SAN JUAN",
    "GREGORIO DEL PILAR CONCEPCION": "GREGORIO DEL PILAR",
    "ASUNCION SAUG": "ASUNCION",
    "MAGSAYSAY LINUGOS": "MAGSAYSAY",
    "STA CRUZ": "SANTA CRUZ",
    "M LANG": "MLANG",
    "T BOLI": "TBOLI",
    "T`BOLI": "TBOLI",
    "JETAFE": "GETAFE",
    "PINAMUNGAHAN": "PINAMUNGAJAN",
    "OZAMIS": "OZAMIZ",
    "BACUNGAN": "LEON T POSTIGO",
    "LEON B POSTIGO": "LEON T POSTIGO",
    "PRES CARLOS P GARCIA": "PRESIDENT CARLOS P GARCIA",
    "PRES CARLOS P GARCIA PITOGO": "PRESIDENT CARLOS P GARCIA",
    "BALIUAG": "BALIWAG",
    "BULAKAN": "BULACAN",
    "GENERAL MARIANO ALVAREZ": "GEN MARIANO ALVAREZ",
    "CORDOBA": "CORDOVA",
    "PAGAGAWAN": "DATU PAGLAS",
    "PIO V CORPUZ": "PIO V CORPUS",
    "PIO V CORPUZ LIMBUHAN": "PIO V CORPUS",
    "MUNOZ": "MUNOZ",
    "MATAAS NA KAHOY": "MATAASNAKAHOY",
}


# -----------------------------------------------------------------------------
# 1. Load and filter to MAYOR races
# -----------------------------------------------------------------------------
print("[*] Loading OpenHalalan vote counts …")
v = pd.read_csv(os.path.join(DATA, "NLE_vote_counts_2007-2025.csv"),
                low_memory=False)

v["year"] = pd.to_numeric(v["year"], errors="coerce")
v["votes"] = pd.to_numeric(v["votes"], errors="coerce").fillna(0)
v["province_norm"] = v["province"].map(norm)
v["city_norm"] = v["city"].map(norm)

mayor = v[(v["position"].str.upper() == "MAYOR")
          & (v["city_norm"] != "")
          & (~v["province_norm"].isin(OVERSEAS))].copy()

print(f"[*] Mayor rows: {len(mayor):,}")
print(f"[*] Cycles: {sorted(mayor['year'].dropna().unique().astype(int))}")
print(f"[*] Distinct (province, city) pairs: "
      f"{mayor.groupby(['province_norm','city_norm']).ngroups}")


# -----------------------------------------------------------------------------
# 2. Collapse to candidate level
# -----------------------------------------------------------------------------
cand = (mayor.groupby(["province_norm", "city_norm", "year", "candidate_name"],
                     as_index=False)["votes"].sum())
totals = (cand.groupby(["province_norm", "city_norm", "year"])["votes"]
              .sum().rename("total_votes").reset_index())
cand = cand.merge(totals, on=["province_norm", "city_norm", "year"])

n_before = cand.groupby(["province_norm", "city_norm", "year"]).ngroups
cand = cand[cand["total_votes"] > 0].copy()
n_after = cand.groupby(["province_norm", "city_norm", "year"]).ngroups
if n_before != n_after:
    print(f"[*] Dropped {n_before - n_after} zero-vote races.")

cand["share"] = cand["votes"] / cand["total_votes"]
cand["share_sq"] = cand["share"] ** 2

enc = (cand.groupby(["province_norm", "city_norm", "year"])["share_sq"]
           .sum().reset_index())
enc["ENC_lt"] = np.where(enc["share_sq"] > 0, 1.0 / enc["share_sq"], np.nan)
enc = (enc.drop(columns="share_sq")
          .rename(columns={"year": "elecyr"})
          .dropna(subset=["ENC_lt"]))

ncand = (cand.groupby(["province_norm", "city_norm", "year"])
             .size().rename("n_candidates").reset_index()
             .rename(columns={"year": "elecyr"}))

cand_sorted = cand.sort_values(
    ["province_norm", "city_norm", "year", "votes"],
    ascending=[True, True, True, False])
winners = (cand_sorted.groupby(["province_norm", "city_norm", "year"],
                               as_index=False).head(1)
                     .rename(columns={"candidate_name": "winner_name",
                                      "share": "pi"}))

elec = (winners[["province_norm", "city_norm", "year", "winner_name",
                 "pi", "total_votes"]]
        .rename(columns={"year": "elecyr"})
        .merge(enc, on=["province_norm", "city_norm", "elecyr"], how="inner")
        .merge(ncand, on=["province_norm", "city_norm", "elecyr"], how="left"))


# -----------------------------------------------------------------------------
# 3. Chain winners
# -----------------------------------------------------------------------------
elec = elec.sort_values(["province_norm", "city_norm", "elecyr"]).reset_index(drop=True)
elec["prev_winner"] = elec.groupby(["province_norm", "city_norm"])["winner_name"].shift(1)
elec["prev_elecyr"] = elec.groupby(["province_norm", "city_norm"])["elecyr"].shift(1)
elec["is_incumbent"] = ((elec["prev_elecyr"] == elec["elecyr"] - 3)
                        & (elec["winner_name"] == elec["prev_winner"]))


# -----------------------------------------------------------------------------
# 3b. Add pi_incumbent: previous cycle's winner's share in current cycle.
# -----------------------------------------------------------------------------
cand_shares = (cand[["province_norm", "city_norm", "year",
                     "candidate_name", "share"]]
               .copy())
cand_shares = cand_shares.rename(columns={"year": "elecyr",
                                          "candidate_name": "prev_winner",
                                          "share": "pi_incumbent"})

elec = elec.merge(cand_shares,
                  on=["province_norm", "city_norm", "elecyr", "prev_winner"],
                  how="left")

print(f"[*] pi_incumbent populated: {elec['pi_incumbent'].notna().sum():,} "
      f"of {len(elec):,} rows")


# -----------------------------------------------------------------------------
# 4. Match (province, city) → PSGC10
# -----------------------------------------------------------------------------
panel = pd.read_csv(os.path.join(DATA, "panel.csv"), low_memory=False)
panel["lgu_norm"] = panel["lgu_name"].map(norm)
panel["prov_norm"] = panel["province"].map(norm)
panel["lgu_type_clean"] = panel["lgu_type"].astype(str).str.strip().str.lower()
sub = panel[panel["lgu_type_clean"].isin(["city", "municipality"])].copy()

# Stage A: (province, city) exact
lookup_two = (sub[["prov_norm", "lgu_norm", "psgc10", "lgu_name",
                   "lgu_type_clean"]]
              .drop_duplicates(["prov_norm", "lgu_norm"]))
elec = elec.merge(
    lookup_two.rename(columns={"prov_norm": "province_norm",
                               "lgu_norm": "city_norm"}),
    on=["province_norm", "city_norm"], how="left")
print(f"[*] Stage A: {elec['psgc10'].notna().sum():,} / {len(elec):,}")

# Stage B: (city) exact, province-agnostic
missing = elec["psgc10"].isna()
lookup_one = (sub.drop_duplicates("lgu_norm")
                 .set_index("lgu_norm")[["psgc10", "lgu_name",
                                         "lgu_type_clean"]])
m = elec.loc[missing, "city_norm"]
elec.loc[missing, "psgc10"] = m.map(lookup_one["psgc10"]).values
elec.loc[missing, "lgu_name"] = m.map(lookup_one["lgu_name"]).values
elec.loc[missing, "lgu_type_clean"] = m.map(lookup_one["lgu_type_clean"]).values
print(f"[*] Stage B: {elec['psgc10'].notna().sum():,} / {len(elec):,}")

# Stage C: manual overrides
missing = elec["psgc10"].isna()
panel_by_norm = (sub.drop_duplicates("lgu_norm")
                    .set_index("lgu_norm")[["psgc10", "lgu_name",
                                            "lgu_type_clean"]])
overrides = pd.Series(MANUAL_OVERRIDES)
m = elec.loc[missing, "city_norm"]
mapped = m.map(overrides)
elec.loc[missing, "psgc10"] = mapped.map(panel_by_norm["psgc10"]).values
elec.loc[missing, "lgu_name"] = mapped.map(panel_by_norm["lgu_name"]).values
elec.loc[missing, "lgu_type_clean"] = (
    mapped.map(panel_by_norm["lgu_type_clean"]).values)
print(f"[*] Stage C: {elec['psgc10'].notna().sum():,} / {len(elec):,}")

# Stage D: token subset, province-restricted
missing = elec["psgc10"].isna()
panel_by_prov = {}
for prov, grp in sub.groupby("prov_norm"):
    panel_by_prov[prov] = list(zip(grp["lgu_norm"], grp["psgc10"],
                                   grp["lgu_name"], grp["lgu_type_clean"]))


def token_subset(city_norm, candidate_rows):
    oh_tokens = set(city_norm.split())
    hits = []
    for lgu_norm, psgc, lgu_name, lgu_type in candidate_rows:
        panel_tokens = set(lgu_norm.split())
        if not panel_tokens:
            continue
        if panel_tokens.issubset(oh_tokens) or oh_tokens.issubset(panel_tokens):
            score = len(panel_tokens & oh_tokens) - 0.1 * len(panel_tokens ^ oh_tokens)
            hits.append((score, psgc, lgu_name, lgu_type))
    if not hits:
        return (np.nan, np.nan, np.nan)
    hits.sort(reverse=True)
    _, psgc, lgu_name, lgu_type = hits[0]
    return (psgc, lgu_name, lgu_type)


if missing.any():
    for idx in elec.index[missing]:
        prov = elec.at[idx, "province_norm"]
        city = elec.at[idx, "city_norm"]
        rows = panel_by_prov.get(prov, [])
        psgc, lgu_name, lgu_type = token_subset(city, rows)
        elec.at[idx, "psgc10"] = psgc
        elec.at[idx, "lgu_name"] = lgu_name
        elec.at[idx, "lgu_type_clean"] = lgu_type
print(f"[*] Stage D: {elec['psgc10'].notna().sum():,} / {len(elec):,}")

# Stage E: token subset, province-agnostic, cities only
missing = elec["psgc10"].isna()
city_panel_rows = list(zip(sub[sub["lgu_type_clean"] == "city"]["lgu_norm"],
                           sub[sub["lgu_type_clean"] == "city"]["psgc10"],
                           sub[sub["lgu_type_clean"] == "city"]["lgu_name"],
                           sub[sub["lgu_type_clean"] == "city"]["lgu_type_clean"]))
if missing.any():
    for idx in elec.index[missing]:
        city = elec.at[idx, "city_norm"]
        psgc, lgu_name, lgu_type = token_subset(city, city_panel_rows)
        elec.at[idx, "psgc10"] = psgc
        elec.at[idx, "lgu_name"] = lgu_name
        elec.at[idx, "lgu_type_clean"] = lgu_type
print(f"[*] Stage E: {elec['psgc10'].notna().sum():,} / {len(elec):,}")

# Stage F: drop SGA rows, then any residual
before_sga = len(elec)
elec = elec[~elec["city_norm"].isin(SGA_DROP)].copy()
if len(elec) < before_sga:
    print(f"[*] Dropped {before_sga - len(elec)} rows in SGA_DROP "
          f"(BARMM barangays, not standalone LGUs).")

still_missing = elec["psgc10"].isna()
if still_missing.any():
    unmatched = (elec.loc[still_missing, ["province_norm", "city_norm"]]
                     .drop_duplicates()
                     .sort_values(["province_norm", "city_norm"]))
    print(f"\n[!] {len(unmatched)} pairs still unmapped:")
    print(unmatched.to_string(index=False))
    elec = elec.dropna(subset=["psgc10"])
else:
    print("\n[✓] All pairs mapped.")

# Finalize
elec["psgc10"] = (elec["psgc10"].astype(str).str.strip()
                  .str.replace(r"\.0$", "", regex=True).str.zfill(10))
elec = elec.rename(columns={"lgu_type_clean": "lgutype"})
elec = elec[elec["n_candidates"] > 1]

out = elec[["psgc10", "lgu_name", "lgutype", "elecyr",
            "pi", "pi_incumbent", "ENC_lt", "n_candidates",
            "winner_name", "prev_winner", "is_incumbent"]]
out = out.sort_values(["psgc10", "elecyr"]).reset_index(drop=True)
out.to_csv(os.path.join(DATA, "election_panel_mayor.csv"), index=False)

print(f"\n[✓] Wrote election_panel_mayor.csv")
print(f"    rows:       {len(out):,}")
print(f"    LGUs:       {out['psgc10'].nunique()}")
print(f"    cities:     {out[out['lgutype']=='city']['psgc10'].nunique()} "
      f"(panel has 149)")
print(f"    municips:   "
      f"{out[out['lgutype']=='municipality']['psgc10'].nunique()} "
      f"(panel has 1,483)")
print(f"    cycles:     {sorted(out['elecyr'].astype(int).unique())}")
print(f"\nπ:            mean={out['pi'].mean():.3f}  sd={out['pi'].std():.3f}")
print(f"π_incumbent:  mean={out['pi_incumbent'].mean():.3f}  "
      f"sd={out['pi_incumbent'].std():.3f}  "
      f"N={out['pi_incumbent'].notna().sum():,}")
print(f"ENC:          mean={out['ENC_lt'].mean():.3f}  "
      f"sd={out['ENC_lt'].std():.3f}")
print(f"Incumbent share: {out['is_incumbent'].mean():.3f}")
print(f"\nRows per cycle:\n{out.groupby('elecyr').size().to_string()}")