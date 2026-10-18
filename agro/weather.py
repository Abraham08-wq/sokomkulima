"""
RAIN, SEASONS AND THE WATER-LIMITED YIELD.

Data: NASA POWER (power.larc.nasa.gov), free, no key, every point on Earth: daily rainfall (PRECTOTCORR) and
      daily max/min temperature since 1981. We use every full year since 1991.

For every possible planting date (every 5 days of the year) and every one of the 35 years, a daily soil-water
bucket is simulated for the crop's whole season:
    crop water need  ETc = Kc(day) x ET0         ET0 = Hargreaves (FAO-56 eq. 52), from temperature + latitude
    soil bucket      S   = min(AWC, S + rain - ETa)   (rain beyond a full bucket runs off)
    water used       ETa = min(ETc, S + rain)
    WRSI             = sum(ETa) / sum(ETc)            (Water Requirement Satisfaction Index, FAO)
    water-limited yield share  = max(0, 1 - ky x (1 - WRSI))          (FAO Paper 33)
The best planting dates are those with the highest median yield share across the 35 years. Places with two
rainy seasons (much of East Africa) get two windows.
"""
from pathlib import Path
from datetime import date, timedelta
import json, os, urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "agro_cache"
POWER = os.environ.get("POWER_URL", "https://power.larc.nasa.gov").rstrip("/")
UA = {"User-Agent": "SokoMkulima/1.0 (season planner)"}
FIRST_YEAR = int(os.environ.get("WEATHER_FIRST_YEAR", 1991))   # 1991 -> last full year (~35 years); shared with insurance
STEP = 5                     # planting dates tested every 5 days


def get_daily(lat, lon):
    """35 years of daily rain, Tmax, Tmin for the 0.5-degree NASA POWER cell. Arrays shaped (years, 365)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    glat, glon = round(lat * 2) / 2, round(lon * 2) / 2
    last = date.today().year - 1
    first = FIRST_YEAR
    cpath = CACHE / f"power_{glat:.1f}_{glon:.1f}_{first}-{last}.json"
    if cpath.exists():
        raw = json.loads(cpath.read_text())
    else:
        url = (f"{POWER}/api/temporal/daily/point?start={first}0101&end={last}1231&latitude={glat}&longitude={glon}"
               f"&community=ag&parameters=PRECTOTCORR,T2M_MAX,T2M_MIN&format=json")
        req = urllib.request.Request(url, headers=UA)
        raw = json.loads(urllib.request.urlopen(req, timeout=120).read())["properties"]["parameter"]
        cpath.write_text(json.dumps(raw))
    years = sorted({int(k[:4]) for k in raw["PRECTOTCORR"]})
    out = {}
    for key, name in (("PRECTOTCORR", "rain"), ("T2M_MAX", "tmax"), ("T2M_MIN", "tmin")):
        arr = np.full((len(years), 365), np.nan)
        for k, v in raw[key].items():
            y, m, d = int(k[:4]), int(k[4:6]), int(k[6:8])
            if m == 2 and d == 29:
                continue
            doy = (date(2001, m, d) - date(2001, 1, 1)).days          # non-leap calendar
            arr[years.index(y), doy] = np.nan if v is None or v <= -998 else v
        # fill rare gaps with the day's average over the years
        mean = np.nanmean(arr, 0)
        arr = np.where(np.isnan(arr), mean[None], arr)
        out[name] = np.nan_to_num(arr, nan=0.0 if name == "rain" else 25.0)
    out["years"] = years
    out["cell"] = [glat, glon]
    return out


def et0_hargreaves(tmax, tmin, lat):
    doy = np.arange(1, 366)
    phi = np.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    tmean = (tmax + tmin) / 2
    return np.maximum(0, 0.0023 * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0)) * 0.408 * ra[None])


def kc_curve(crop):
    n = crop["days"]; ini, mid, end = crop["kc"]; s = np.cumsum(crop["stages"]) * n
    d = np.arange(n)
    return np.interp(d, [0, s[0], s[1], s[2], n], [ini, ini, mid, mid, end])


def water_balance(w, crop, awc):
    """-> frac[start_index, year]: water-limited share of potential yield; wrsi of same shape; starts (doy)."""
    rain = w["rain"].reshape(-1); et0 = w["et0"].reshape(-1)
    n, kc = crop["days"], kc_curve(crop)
    starts = np.arange(0, 365, STEP)
    Y = w["rain"].shape[0]
    wrsi = np.full((len(starts), Y), np.nan)
    for yi in range(Y):
        idx0 = yi * 365 + starts                                   # (S,)
        ok = idx0 + n <= len(rain)
        S = np.zeros(len(starts)); need = np.zeros(len(starts)); used = np.zeros(len(starts))
        for d in range(n):
            ii = np.minimum(idx0 + d, len(rain) - 1)
            etc = kc[d] * et0[ii]
            avail = S + rain[ii]
            eta = np.minimum(etc, avail)
            S = np.minimum(awc, avail - eta)
            need += etc; used += eta
        wrsi[:, yi] = np.where(ok, used / np.maximum(need, 1e-6), np.nan)
    frac = np.clip(1 - crop["ky"] * (1 - wrsi), 0, 1)
    return starts, wrsi, frac


def _doy_to_text(doy):
    return (date(2001, 1, 1) + timedelta(days=int(doy) % 365)).strftime("%d %b").lstrip("0")


def seasons(lat, lon, crop, awc):
    w = get_daily(lat, lon)
    w["et0"] = et0_hargreaves(w["tmax"], w["tmin"], lat)
    starts, wrsi, frac = water_balance(w, crop, awc)
    med = np.nanmedian(frac, 1)
    sm = np.convolve(np.r_[med[-2:], med, med[:2]], np.ones(5) / 5, "valid")      # circular smoothing
    found = []
    for i in np.argsort(-sm):
        if sm[i] < 0.35 or len(found) == 2:
            break
        if all(min(abs(starts[i] - starts[j]), 365 - abs(starts[i] - starts[j])) >= max(crop["days"], 100) for j in found):
            found.append(i)
    out = []
    for i in found:
        lo = hi = i
        while sm[(lo - 1) % len(sm)] >= 0.9 * sm[i] and (i - lo) < 8:
            lo -= 1
        while sm[(hi + 1) % len(sm)] >= 0.9 * sm[i] and (hi - i) < 8:
            hi += 1
        best = int(np.nanargmax(np.nanmedian(frac[[k % len(sm) for k in range(lo, hi + 1)]], 1))) + lo
        f = frac[best % len(sm)]
        f = f[~np.isnan(f)]
        out.append({"start_doy": int(starts[best % len(sm)]), "window": [int(starts[lo % len(sm)]), int(starts[hi % len(sm)])],
                    "window_text": f"{_doy_to_text(starts[lo % len(sm)])} - {_doy_to_text(starts[hi % len(sm)] + STEP - 1)}",
                    "median_wrsi": round(float(np.nanmedian(wrsi[best % len(sm)])), 3),
                    "frac_by_year": [round(float(x), 3) for x in f],
                    "good_season_share": round(float(np.mean(f >= 0.7)), 2),
                    "harvest_text": _doy_to_text(starts[best % len(sm)] + crop["days"]),
                    "harvest_doy": int((starts[best % len(sm)] + crop["days"]) % 365)})
    monthly = [round(float(w["rain"][:, (date(2001, m, 1) - date(2001, 1, 1)).days:
                                    (date(2001, m % 12 + 1, 1) - date(2001, 1, 1)).days or 365].sum(1).mean()), 0)
               for m in range(1, 13)]
    return {"seasons": out, "curve": {"starts": starts.tolist(), "median_frac": [round(float(x), 3) for x in med]},
            "monthly_rain_mm": monthly, "annual_rain_mm": round(float(w["rain"].sum(1).mean())),
            "years": [w["years"][0], w["years"][-1]], "cell": w["cell"]}


def get_recent(lat, lon, start, end):
    """Daily rain/Tmax/Tmin for a date range of the CURRENT season (cached per day). -> dict of lists by date."""
    CACHE.mkdir(parents=True, exist_ok=True)
    glat, glon = round(lat * 2) / 2, round(lon * 2) / 2
    cpath = CACHE / f"recent_{glat:.1f}_{glon:.1f}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if cpath.exists():
        return json.loads(cpath.read_text())
    url = (f"{POWER}/api/temporal/daily/point?start={start:%Y%m%d}&end={end:%Y%m%d}&latitude={glat}&longitude={glon}"
           f"&community=ag&parameters=PRECTOTCORR,T2M_MAX,T2M_MIN&format=json")
    raw = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())["properties"]["parameter"]
    cpath.write_text(json.dumps(raw))
    return raw
