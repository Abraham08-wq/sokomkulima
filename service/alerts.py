"""
AUTOMATIC SMS ALERTS: the farmer registers once; the system writes at the right moments, in their language.

  plant_soon      planting window opens within 10 days        -> planting dates + fertilizer per acre
  season_failing  the weather index says this season is failing -> expected loss, insurance payout if insured
  harvest_sell    harvest is within 14 days (or up to 30 days after) -> sell or store, and the best market

Each alert is sent once per season (messages table, UNIQUE farmer/kind/key). Runs every ALERT_HOURS inside the
backend, or on demand from the admin page, where any date can be simulated for a demo.
"""
from datetime import date
from . import db, sms_gateway

PREFIX = "SokoMkulima: "


def doy(d):
    return d.timetuple().tm_yday - 1


def run(tools, today=None, farmer_id=None):
    today = today or date.today()
    t = doy(today)
    farmers = db.q("SELECT * FROM farmers WHERE active=1" + (" AND id=?" if farmer_id else ""),
                   (farmer_id,) if farmer_id else ())
    sent, errors = [], []
    for f in farmers:
        if f["lat"] is None:
            continue
        lang, crop, place = f["lang"] or "en", f["crop"], f["place"] or ""
        try:
            p = tools["plan"](f["lat"], f["lon"], crop, f["acres"] or 1, t)
        except Exception as e:
            errors.append(f"{f['phone']}: plan {type(e).__name__}"); p = None
        if p and p.get("ok"):
            s = p["season"]
            until = (s["window"][0] - t) % 365
            if until <= 10:
                yr = today.year + (1 if s["window"][0] < t - 10 else 0)
                m = sms_gateway.send(f, "plant_soon", f"{yr}-{s['window'][0]}", PREFIX + tools["plan_reply"](lang, crop, place, p))
                if m:
                    sent.append({**m, "phone": f["phone"]})
            for ss in p.get("all_seasons", [s]):                       # harvest of ANY season (not only the next one)
                h = ss.get("harvest_doy")
                if h is None or not ((h - t) % 365 <= 14 or (t - h) % 365 <= 30):
                    continue
                hy = today.year if (t - h) % 365 <= 30 and t >= h else today.year + (1 if h < t else 0)
                text = tools["price_reply"](lang, crop, place)
                kg = p["yield_gap"]["with_plan"]["kg_farm"]
                best = tools["where_best"](crop, f["lat"], f["lon"], kg)
                if best:
                    text += f" Best market: {best['name']} (+${best['gain_total']:.0f} for {kg} kg after transport)."
                if text:
                    m = sms_gateway.send(f, "harvest_sell", f"{hy}-{h}", PREFIX + text)
                    if m:
                        sent.append({**m, "phone": f["phone"]})
        try:
            r = tools["season_status"](crop, f["lat"], f["lon"], f["acres"] or 1, today)
        except Exception as e:
            errors.append(f"{f['phone']}: season {type(e).__name__}"); r = None
        if r and r["season_now"].get("state") == "running" and r["season_now"].get("would_pay"):
            m = sms_gateway.send(f, "season_failing", r["season_now"]["start"],
                                 PREFIX + tools["insure_reply"](lang, crop, place, r))
            if m:
                sent.append({**m, "phone": f["phone"]})
    return {"date": today.isoformat(), "farmers": len(farmers), "sent": sent, "errors": errors[:20]}
