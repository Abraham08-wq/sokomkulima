"""
INSURANCE STEP 2 — what the weather did in each district's season.

For one season (planting month -> harvest month) the daily NASA POWER weather is turned into 18 DEKADS (10-day
steps, 180 days from planting). Per dekad (the deep model reads this sequence):
    0 rain (mm/100)          1 rain minus the 35-year normal for these dates (mm/100)
    2 crop water need ETc (mm/100)                 3 water supply ratio ETa/ETc (soil bucket, FAO)
    4 mean Tmax minus 30 C (/10)                   5 heavy-rain days >= 30 mm (count/5)
Static features (also used by the simple models):
    season WRSI, season rain / normal, early-season (first 40 days) rain / normal, dry-spell max (days/30),
    latitude, longitude, crop (one-hot), season length (/180)
"""
import numpy as np
from agro.crops import CROPS
from agro.weather import et0_hargreaves, kc_curve

N_DEKADS, DEKAD = 18, 10
SEQ = ["rain", "rain_anom", "etc", "supply", "tmax", "heavy"]
STATIC = ["wrsi", "rain_ratio", "early_ratio", "dry_spell", "lat", "lon", "maize", "sorghum", "millet", "length"]
KY = {"maize": 1.25, "sorghum": 0.9, "millet": 0.9}
AWC = 100.0                      # mm; one value for all districts (district soils vary, the model learns the rest)


def month_doy(month, end=False):
    cum = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    return int(cum[month] - 1) if end else int(cum[month - 1])


def prepare_cell(w, lat):
    """Flatten a get_daily() result into continuous daily series + day-of-year normals."""
    et0 = et0_hargreaves(w["tmax"], w["tmin"], lat)
    return {"rain": w["rain"].reshape(-1), "et0": et0.reshape(-1), "tmax": w["tmax"].reshape(-1),
            "rain_clim": w["rain"].mean(0), "first_year": w["years"][0], "n_years": len(w["years"])}


def season_window(cell, planting_year, planting_month, harvest_year, harvest_month):
    s = (planting_year - cell["first_year"]) * 365 + month_doy(planting_month)
    e = (harvest_year - cell["first_year"]) * 365 + month_doy(harvest_month, end=True) + 1
    if e <= s:                                    # harvest month earlier in the year than planting -> next year
        e += 365
    return s, min(e, s + N_DEKADS * DEKAD)


def season_features(rain, et0, tmax, rain_clim_seq, crop, lat, lon):
    """rain/et0/tmax/rain_clim_seq: 1-D arrays covering the season from planting (length L <= 180).
    -> (seq [18 x 6], mask [18], static [10])"""
    L = len(rain)
    c = dict(CROPS[crop]); c["days"] = L
    kc = kc_curve(c)
    S = 0.0; etc = kc * et0; eta = np.zeros(L)
    for d in range(L):                            # FAO soil-water bucket
        avail = S + rain[d]
        eta[d] = min(etc[d], avail)
        S = min(AWC, avail - eta[d])
    seq = np.zeros((N_DEKADS, len(SEQ)), dtype=np.float32); mask = np.zeros(N_DEKADS, dtype=np.float32)
    for k in range(N_DEKADS):
        a, b = k * DEKAD, min(L, (k + 1) * DEKAD)
        if a >= L:
            break
        mask[k] = 1
        r = rain[a:b]
        seq[k] = [r.sum() / 100, (r.sum() - rain_clim_seq[a:b].sum()) / 100, etc[a:b].sum() / 100,
                  eta[a:b].sum() / max(etc[a:b].sum(), 1e-6), (tmax[a:b].mean() - 30) / 10, (r >= 30).sum() / 5]
    wrsi = eta.sum() / max(etc.sum(), 1e-6)
    clim = max(rain_clim_seq[:L].sum(), 1.0)
    early = min(40, L)
    dry = run = 0
    for v in rain:
        run = run + 1 if v < 1 else 0
        dry = max(dry, run)
    static = np.array([wrsi, rain.sum() / clim, rain[:early].sum() / max(rain_clim_seq[:early].sum(), 1.0), dry / 30,
                       lat / 90, lon / 90, crop == "maize", crop == "sorghum", crop == "millet", L / 180], dtype=np.float32)
    return seq, mask, static


def from_cell(cell, crop, lat, lon, py, pm, hy, hm):
    s, e = season_window(cell, py, pm, hy, hm)
    if s < 0 or e > len(cell["rain"]) or e - s < 45:
        return None
    doy0 = s % 365
    clim = np.array([cell["rain_clim"][(doy0 + i) % 365] for i in range(e - s)])
    return season_features(cell["rain"][s:e], cell["et0"][s:e], cell["tmax"][s:e], clim, crop, lat, lon)
