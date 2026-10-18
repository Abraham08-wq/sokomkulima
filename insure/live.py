"""
INSURANCE STEP 4 — this season, live: how is the season going in a district, and would the policy pay?

Weather so far this season comes from NASA POWER (a few days behind real time). The days still to come are filled
with the 35-year normal for those dates, so the estimate firms up as the season goes on.
"""
from pathlib import Path
from datetime import date, timedelta
import numpy as np

from agro.weather import get_daily, get_recent, et0_hargreaves
from insure.features import prepare_cell, season_features, month_doy, N_DEKADS, DEKAD, KY, STATIC

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
_live = None


def model():
    global _live
    if _live is None:
        import joblib
        _live = joblib.load(OUT / "insure_live.joblib")
    return _live


def predict(st, seq, mask, pol):
    """-> (score that triggers payment, predicted loss), for the model that prices the policies."""
    m = model(); best = m["best"]; crop = pol["crop"]
    from insure.train import district_extra, summary_X, deep_predict
    st2 = district_extra(st, np.array([pol.get("wrsi_mean", 0.8)]), np.array([pol.get("rain_mean", 1.0)]))
    if m.get("ndvi"):                       # live greenness is not available yet: assume a normal season
        st2 = np.hstack([st2, np.zeros((len(st2), 2), dtype=np.float32)])
    if best == "rain_index":
        v = m["rain_a"] * np.maximum(0, 1 - st[:, STATIC.index("rain_ratio")])
    elif best == "wrsi_index":
        v = KY[crop] * (1 - st[:, STATIC.index("wrsi")])
    elif best == "ridge":
        r, sc = m["ridge"]
        v = r.predict(sc.transform(summary_X(seq, st2)))
    else:
        prob, loss = deep_predict(m["deep"], st2, seq, mask)
        return float(prob[0]), float(np.clip(loss[0], -1, 1))
    v = float(np.clip(v[0], -1, 1))
    return v, v


def current_season(pol, today=None):
    """The season to show: the one running now, else the next one. -> (start_date, end_date, status)."""
    today = today or date.today()
    pm, hm = pol["planting_month"], pol["harvest_month"]
    for y in (today.year - 1, today.year, today.year + 1):
        start = date(y, pm, 1)
        end_y = y if hm >= pm else y + 1
        end = date(end_y, hm, 28) + timedelta(days=4)
        end = end.replace(day=1) - timedelta(days=1)
        end = min(end, start + timedelta(days=N_DEKADS * DEKAD - 1))
        if start <= today <= end:
            return start, end, "running"
        if start > today:
            return start, end, "upcoming"
    return None, None, "unknown"


def season_now(pol, today=None):
    today = today or date.today()
    start, end, state = current_season(pol, today)
    if state != "running":
        return {"state": state, "starts": start.isoformat() if start else None}
    w = get_daily(pol["lat"], pol["lon"])
    cell = prepare_cell(w, round(pol["lat"] * 2) / 2)
    L = (end - start).days + 1
    doy0 = month_doy(start.month) + start.day - 1
    clim = lambda arr2d: np.array([arr2d[:, (doy0 + i) % 365].mean() for i in range(L)])
    rain, tmax = clim(w["rain"]), clim(w["tmax"])
    et0 = clim(et0_hargreaves(w["tmax"], w["tmin"], round(pol["lat"] * 2) / 2))
    rain_clim = rain.copy()
    upto = min(today - timedelta(days=3), end)                         # POWER is a few days behind
    known = 0
    if upto >= start:
        raw = get_recent(pol["lat"], pol["lon"], start, upto)
        d = start
        while d <= upto:
            k = d.strftime("%Y%m%d"); i = (d - start).days
            r = raw["PRECTOTCORR"].get(k); tx = raw["T2M_MAX"].get(k); tn = raw["T2M_MIN"].get(k)
            if r is not None and r > -998:
                rain[i] = r; known = i + 1
            if tx is not None and tx > -998 and tn is not None and tn > -998:
                tmax[i] = tx
            d += timedelta(days=1)
    seq, mask, st = season_features(rain, et0, tmax, rain_clim, pol["crop"], pol["lat"], pol["lon"])
    score, loss = predict(st[None], seq[None], mask[None], pol)
    trig = model()["trigger"]
    if score >= trig:                         # triggered = the model judges this a bad season: loss of at least 30 %
        loss = max(loss, 0.3)
    return {"state": "running", "start": start.isoformat(), "end": end.isoformat(), "observed_days": known,
            "season_days": L, "progress": round(known / L, 2), "rain_so_far": round(float(rain[:known].sum())),
            "normal_so_far": round(float(rain_clim[:known].sum())), "pred_loss": round(loss, 3),
            "score": round(score, 3), "would_pay": score >= trig,
            "payout": round(min(max(loss, 0.3), 1), 3) if score >= trig else 0.0,
            "trigger": trig, "wrsi_so_far": round(float(st[STATIC.index("wrsi")]), 3)}
