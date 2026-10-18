"""
QUEFTS — Quantitative Evaluation of the Fertility of Tropical Soils (Janssen et al., 1990, Geoderma 46: 299-318).

The standard crop-nutrient model behind fertilizer advice in Africa. Given
  - what the SOIL supplies (N, P, K in kg/ha, from pH, organic carbon, P and K),
  - what FERTILIZER adds (kg/ha of N, P, K, times a recovery fraction),
  - the yield that WATER allows (from the rainfall water balance),
it computes the yield the crop can actually reach. It captures the key agronomy:
a nutrient only helps if the others are not limiting (balanced nutrition).

Line-by-line port of the reference implementation: Rquefts (R. Hijmans, quefts.cpp, GPL >= 3),
including its crop parameters (inst/extdata/quefts_crop_pars.csv) and soil supply functions (nutSupply.R).
"""
import numpy as np

# Crop nutrient concentration parameters (kg/kg), Yzero (kg/ha) and N fixation share, from Rquefts.
CROP_PARS = {
    #              NminSt  NminVeg NmaxSt  NmaxVeg PminSt  PminVeg PmaxSt  PmaxVeg KminSt  KminVeg KmaxSt  KmaxVeg Yzero Nfix
    "Maize":      (0.0095, 0.004,  0.022,  0.0125, 0.0017, 0.0004, 0.0075, 0.003,  0.002,  0.005,  0.006,  0.02,  400, 0),
    "Sorghum":    (0.01,   0.0035, 0.032,  0.012,  0.0014, 0.0005, 0.006,  0.0025, 0.0025, 0.007,  0.0075, 0.028, 200, 0),
    "Millet":     (0.01,   0.0032, 0.03,   0.0105, 0.0014, 0.0005, 0.008,  0.0025, 0.003,  0.007,  0.008,  0.028, 200, 0),
    "Rice":       (0.009,  0.0032, 0.027,  0.011,  0.0011, 0.0003, 0.004,  0.0016, 0.0025, 0.007,  0.0075, 0.028, 200, 0),
    "Wheat":      (0.011,  0.003,  0.031,  0.0105, 0.0016, 0.0004, 0.006,  0.002,  0.003,  0.007,  0.008,  0.028, 200, 0),
    "Field bean": (0.03,   0.008,  0.049,  0.025,  0.0026, 0.0008, 0.006,  0.004,  0.008,  0.01,   0.02,   0.03,  100, 0.75),
    "Cowpea":     (0.03,   0.008,  0.047,  0.027,  0.002,  0.001,  0.006,  0.004,  0.009,  0.01,   0.018,  0.03,  100, 0.75),
    "Groundnut":  (0.022,  0.013,  0.045,  0.019,  0.0012, 0.001,  0.0048, 0.0019, 0.006,  0.0105, 0.015,  0.033, 100, 0.75),
    "Cassava":    (0.0022, 0.008,  0.009,  0.017,  0.0007, 0.001,  0.0024, 0.005,  0.0035, 0.007,  0.014,  0.017, 100, 0),
}
KEYS = ["NminStore", "NminVeg", "NmaxStore", "NmaxVeg", "PminStore", "PminVeg", "PmaxStore", "PmaxVeg",
        "KminStore", "KminVeg", "KmaxStore", "KmaxVeg", "Yzero", "Nfix"]

RECOVERY = {"N": 0.5, "P": 0.1, "K": 0.5}          # share of applied fertilizer the crop takes up (Rquefts default)
UPTAKE_ADJUST = ([0, 40, 80, 120, 240, 360, 1000], [0, 0.4, 0.7, 1, 1.6, 2, 2])   # season length (days) -> soil supply factor


# ---------------- soil supply (Rquefts nutSupply.R; Janssen et al. 1990 Table 2, bounded pH factors) ----------------
def soil_supply(pH, soc_gkg, k_mmolkg, p_olsen_mgkg):
    """Indigenous soil supply (kg/ha per standard 120-day season) of N, P and K."""
    fN = np.interp(pH, [4.7, 7], [0.4, 1])
    fP = np.interp(pH, [4.7, 5, 5.5, 6, 6.7, 7, 7.5, 8], [0.02, .5, 0.875, 1, 1, 0.9775, 0.84, 0.02])
    fK = np.interp(pH, [4.5, 5, 6, 6.8], [1, 0.8842312, 0.71, 0.6])
    N = fN * 6.8 * soc_gkg
    P = fP * 0.35 * soc_gkg + 0.5 * p_olsen_mgkg
    K = (fK * 400 * k_mmolkg) / (2 + 0.9 * soc_gkg)
    return max(N, 0.0), max(P, 0.0), max(K, 0.0)


# ---------------- model (quefts.cpp) ----------------
def _uptake(Si, iZero, iYratA, iYratD, Sj, jZero, jYratA, jYratD):
    if Si <= iZero + jYratA * (Sj - jZero) / iYratD:
        return Si
    if Si < iZero + (Sj - jZero) * (2 * jYratD / iYratA - jYratA / iYratD):
        return Si - (0.25 * (Si - iZero - jYratA * (Sj - jZero) / iYratD) ** 2) / \
            ((jYratD / iYratA - jYratA / iYratD) * (Sj - jZero))
    return iZero + jYratD * (Sj - jZero) / iYratA


def _yield(iYA, jYA, iUPT, izero, iYratD, iYratA, iYD, jYD, zYD, YldO):
    LYD = min(iYD, jYD, zYD, YldO)
    if iYD <= jYA:
        return LYD if iYD > LYD else iYD
    elif (iYA >= LYD) or (iYD <= LYD):
        if (iYA < LYD) and (iYD == LYD):
            LYD = min(jYD, zYD, YldO)
        else:
            return LYD
    EY = jYA + 2 * (LYD - jYA) * (iUPT - izero - jYA / iYratD) / (LYD / iYratA - jYA / iYratD) \
        - (LYD - jYA) * (iUPT - izero - jYA / iYratD) ** 2 / (LYD / iYratA - jYA / iYratD) ** 2
    if jYA > LYD:
        EY = LYD
    return EY


def _requirements(minVeg, maxVeg, minStore, maxStore, supply, recovery, leaf, stem, store, fix):
    limVeg = (minVeg + maxVeg) / 2; limStore = (maxStore + minStore) / 2
    req = 0 if fix else (limVeg * (leaf + stem) + limStore * store)
    reqFert = max(0.0, (req - supply) / recovery)
    reqAcc = (maxVeg * (leaf + stem) + maxStore * store) * (1 - fix)
    reqDil = (minVeg * (leaf + stem) + minStore * store) * (1 - fix)
    return reqAcc, reqDil, reqFert


def quefts(crop, soilN, soilP, soilK, fertN, fertP, fertK, store_att, season_days=120, leaf_frac=0.45, stem_frac=0.55):
    """Nutrient-limited yield of the storage organ (grain/roots, kg dry matter/ha).
    store_att = water-limited attainable yield (kg DM/ha). Returns (yield, details)."""
    c = dict(zip(KEYS, CROP_PARS[crop]))
    leaf, stem, store = leaf_frac * store_att, stem_frac * store_att, store_att
    Nzero = Pzero = Kzero = 0.0
    if store > 200:
        Yo = store
        Nzero = c["Yzero"] * (1 - c["Nfix"]) * (c["NmaxVeg"] + 2 * c["NminVeg"]) / 3
        Pzero = c["Yzero"] * (c["PmaxVeg"] + 2 * c["PminVeg"]) / 3
        Kzero = c["Yzero"] * (c["KmaxVeg"] + 2 * c["KminVeg"]) / 3
    else:
        return 0.0, {}
    rel = float(np.interp(season_days, *UPTAKE_ADJUST))
    NS = soilN * rel + fertN * RECOVERY["N"]
    PS = soilP * rel + fertP * RECOVERY["P"]
    KS = soilK * rel + fertK * RECOVERY["K"]
    Nr = _requirements(c["NminVeg"], c["NmaxVeg"], c["NminStore"], c["NmaxStore"], NS, RECOVERY["N"], leaf, stem, store, c["Nfix"])
    Pr = _requirements(c["PminVeg"], c["PmaxVeg"], c["PminStore"], c["PmaxStore"], PS, RECOVERY["P"], leaf, stem, store, 0.)
    Kr = _requirements(c["KminVeg"], c["KmaxVeg"], c["KminStore"], c["KmaxStore"], KS, RECOVERY["K"], leaf, stem, store, 0.)
    NA, ND = Yo / max(0.1, Nr[0] - Nzero), Yo / max(0.1, Nr[1] - Nzero)
    PA, PD = Yo / max(0.1, Pr[0] - Pzero), Yo / max(0.1, Pr[1] - Pzero)
    KA, KD = Yo / max(0.1, Kr[0] - Kzero), Yo / max(0.1, Kr[1] - Kzero)
    UN = max(0.0, min(_uptake(NS, Nzero, NA, ND, PS, Pzero, PA, PD), _uptake(NS, Nzero, NA, ND, KS, Kzero, KA, KD)))
    UP = max(0.0, min(_uptake(PS, Pzero, PA, PD, NS, Nzero, NA, ND), _uptake(PS, Pzero, PA, PD, KS, Kzero, KA, KD)))
    UK = max(0.0, min(_uptake(KS, Kzero, KA, KD, NS, Nzero, NA, ND), _uptake(KS, Kzero, KA, KD, PS, Pzero, PA, PD)))
    YNA, YND = NA * max(0.0, UN - Nzero), ND * max(0.0, UN - Nzero)
    YPA, YPD = PA * max(0.0, UP - Pzero), PD * max(0.0, UP - Pzero)
    YKA, YKD = KA * max(0.0, UK - Kzero), KD * max(0.0, UK - Kzero)
    E = [_yield(YNA, YPA, UN, Nzero, ND, NA, YND, YPD, YKD, Yo), _yield(YPA, YNA, UP, Pzero, PD, PA, YPD, YND, YKD, Yo),
         _yield(YNA, YKA, UN, Nzero, ND, NA, YND, YKD, YPD, Yo), _yield(YKA, YNA, UK, Kzero, KD, KA, YKD, YND, YPD, Yo),
         _yield(YPA, YKA, UP, Pzero, PD, PA, YPD, YKD, YND, Yo), _yield(YKA, YPA, UK, Kzero, KD, KA, YKD, YPD, YND, Yo)]
    y = min(min(YND, YPD, YKD, Yo), sum(E) / 6)
    limiting = min((("N", YND), ("P", YPD), ("K", YKD)), key=lambda t: t[1])[0]
    return max(0.0, y), {"supply": {"N": NS, "P": PS, "K": KS}, "uptake": {"N": UN, "P": UP, "K": UK},
                         "limiting": limiting, "gap_kg": {"N": Nr[2], "P": Pr[2], "K": Kr[2]}}
