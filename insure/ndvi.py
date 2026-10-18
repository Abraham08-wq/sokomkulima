"""
OPTIONAL: satellite crop greenness (MODIS NDVI) as extra insurance features.

If data/ndvi_districts.csv exists (exported by insure/ndvi_gee.js from Google Earth Engine: fnid, date, ndvi), each
district-season gets two more features:
    season mean NDVI  minus the district's normal for that season (training years only)
    season peak NDVI  minus the district's normal peak
Negative = the crops were less green than usual = a likely bad harvest, even when the rain looked fine.
Without the file nothing changes.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PATHS = [ROOT / "data" / "ndvi_districts.csv", ROOT / "insure" / "data" / "ndvi_districts.csv"]


def available():
    return next((p for p in PATHS if p.exists()), None)


def season_ndvi(df):
    """df: samples with fnid, planting_year/month, harvest_year/month. -> (mean, peak) arrays, NaN where no data."""
    p = available()
    if not p:
        return None
    n = pd.read_csv(p)
    n.columns = [c.lower() for c in n.columns]
    n["date"] = pd.to_datetime(n["date"], errors="coerce")
    n = n.dropna(subset=["date", "ndvi"])
    by = {k: g.sort_values("date") for k, g in n.groupby("fnid")}
    mean, peak = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for i, r in enumerate(df.itertuples()):
        g = by.get(r.fnid)
        if g is None:
            continue
        a = pd.Timestamp(int(r.planting_year), int(r.planting_month), 1)
        b = pd.Timestamp(int(r.harvest_year), int(r.harvest_month), 1) + pd.offsets.MonthEnd(0)
        if b <= a:
            b = b + pd.DateOffset(years=1)
        v = g[(g.date >= a) & (g.date <= b)].ndvi.to_numpy()
        if len(v):
            mean[i], peak[i] = v.mean(), v.max()
    return mean, peak


def anomalies(df, train_mask):
    """-> array (n, 2): season mean and peak NDVI minus the district-season's training-years normal (0 if unknown)."""
    s = season_ndvi(df)
    if s is None:
        return None
    mean, peak = s
    key = df["key"].to_numpy()
    out = np.zeros((len(df), 2), dtype=np.float32)
    for j, v in enumerate((mean, peak)):
        t = pd.Series(v[train_mask]).groupby(key[train_mask]).mean()
        norm = pd.Series(key).map(t).to_numpy(dtype=float)
        a = v - norm
        out[:, j] = np.where(np.isnan(a), 0.0, a)
    print(f"  NDVI: satellite greenness for {np.mean(~np.isnan(mean)):.0%} of district-seasons")
    return out
