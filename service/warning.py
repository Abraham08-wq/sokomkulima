"""
HUNGER EARLY WARNING: which districts are heading into trouble in the next 3-6 months?

For every district with harvest history, four signals the platform already produces are combined:
  1. THIS SEASON (45%)   expected harvest loss from the season's weather so far (insurance model, live).
                         Between seasons: the last season's loss, at half weight.
  2. PRICES AHEAD (25%)  forecast rise of the staple price over 3 months at the nearest market (price model).
  3. PRICES NOW (15%)    rise over the last 3 months (food is already getting expensive).
  4. HISTORY (15%)       how often this district's harvest has failed since 1991.
Score 0-100 -> Low (<25) · Watch (25-45) · Alert (45-65) · Emergency (>=65).

An indicator to point attention, not an official food-security classification (that is IPC / FEWS NET).
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json, os, time
import numpy as np

LEVELS = [(65, "Emergency"), (45, "Alert"), (25, "Watch"), (0, "Low")]
W = {"season": 0.45, "price_ahead": 0.25, "price_now": 0.15, "history": 0.15}
CROP_PRIORITY = ["maize", "sorghum", "millet"]


def level(score):
    return next(name for cut, name in LEVELS if score >= cut)


def compute(policies, bad_threshold, season_now, price_info, out_path, progress=None, today=None):
    today = today or date.today()
    by_fnid = {}
    for k, p in policies.items():                     # one policy per district: the main staple
        cur = by_fnid.get(p["fnid"])
        if cur is None or CROP_PRIORITY.index(p["crop"]) < CROP_PRIORITY.index(cur["crop"]):
            by_fnid[p["fnid"]] = p
    items = list(by_fnid.values())
    cap = int(os.environ.get("WARNING_MAX", 0) or 0)
    if cap:
        items = items[:cap]
    done = [0]

    def one(p):
        h = p["history"]
        bad_share = float(np.mean([x["actual_loss"] >= bad_threshold for x in h])) if h else 0.0
        try:
            sn = season_now(p, today)
        except Exception as e:
            sn = {"state": "unavailable", "error": type(e).__name__}
        if sn.get("state") == "running":
            loss = max(sn.get("pred_loss", 0), 0); s_season = min(loss / 0.5, 1)
            season_txt = f"season {round(100 * sn['progress'])}% in: expected loss {round(100 * loss)}%"
        else:
            last = h[-1] if h else {"pred_loss": 0, "year": None}
            loss = max(last["pred_loss"], 0); s_season = min(loss / 0.5, 1) * 0.5
            season_txt = f"between seasons; last season ({last.get('year')}) loss {round(100 * loss)}% (weather index)"
        pi = price_info(p["lat"], p["lon"]) or {}
        ahead, now = pi.get("ahead_pct"), pi.get("recent_pct")
        s_ahead = min(max((ahead or 0) / 30, 0), 1); s_now = min(max((now or 0) / 30, 0), 1)
        s_hist = min(bad_share / 0.4, 1)
        score = 100 * (W["season"] * s_season + W["price_ahead"] * s_ahead + W["price_now"] * s_now + W["history"] * s_hist)
        drivers = sorted([("this season", W["season"] * s_season, season_txt),
                          ("prices ahead", W["price_ahead"] * s_ahead,
                           f"{pi.get('crop', 'staple')} price {ahead:+.0f}% in 3 months at {pi.get('market')}" if ahead is not None else "no price data nearby"),
                          ("prices now", W["price_now"] * s_now,
                           f"{now:+.0f}% over the last 3 months" if now is not None else "no recent price data"),
                          ("history", W["history"] * s_hist, f"bad harvest in {round(100 * bad_share)}% of seasons since 1991")],
                         key=lambda d: -d[1])
        done[0] += 1
        if progress and done[0] % 50 == 0:
            progress(done[0], len(items))
        return {"fnid": p["fnid"], "name": p["name"], "admin_1": p["admin_1"], "country": p["country"], "crop": p["crop"],
                "lat": p["lat"], "lon": p["lon"], "score": round(score, 1), "level": level(score),
                "season_state": sn.get("state"), "drivers": [{"signal": d[0], "points": round(100 * d[1], 1), "text": d[2]} for d in drivers]}

    with ThreadPoolExecutor(int(os.environ.get("WARNING_THREADS", 6))) as ex:
        rows = list(ex.map(one, items))
    rows.sort(key=lambda r: -r["score"])
    out = {"computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "date": today.isoformat(),
           "weights": W, "levels": [{"from": c, "name": n} for c, n in LEVELS],
           "counts": {n: sum(r["level"] == n for r in rows) for _, n in LEVELS}, "districts": rows}
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    return out
