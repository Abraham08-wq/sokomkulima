"""
STEP 3 — BACKEND API (Flask)

    python api/app.py        ->  http://127.0.0.1:8000

Serves the dashboard and answers:
  GET  /api/datasets                       processed crops and which models have results
  GET  /api/meta?slug=&method=             markets, months, model comparison, demo flag
  GET  /api/series?slug=&method=           all prices + all dynamic edge weights (for the animated map)
  GET  /api/market?slug=&method=&id=       one market: history, forecast, backtest, neighbours, sell/store advice
  POST /api/sms   {"text": "BEI MAHINDI NAIROBI"}   farmer SMS -> reply (English or Swahili)
  GET  /api/status                         LIVE / DEMO / UPDATING, data date, last check, next check
  POST /api/refresh                        check WFP now (retrains only if prices changed)

LIVE: a background thread re-checks WFP (HDX) every REFRESH_HOURS (default 24). When new prices are published it
rebuilds the graph and retrains in the background while the old model keeps serving, then switches over.
"""
from pathlib import Path
import json, os, re, sys, threading, time
from datetime import datetime, timezone, timedelta
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
PROC, OUT, FRONT = ROOT / "data" / "processed", ROOT / "outputs", ROOT / "frontend"
METHOD_ORDER = ["st_gnn", "gru_no_graph", "graph_ridge"]          # best first; uses whichever exists
STORAGE_PCT = 1.0      # default storage cost, % of crop value per month
LOSS_PCT = 1.5         # default storage loss (pests, moisture), % per month

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))
import update as updater                                            # pipeline/update.py
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "model"))
from selection import selection                                     # model/selection.py (honest model choice)
import languages as L                                               # api/languages.py (SMS in 11 languages)
from agro import advisor                                            # agro/ season planner (yield gap)
from agro.crops import CROPS as AGRO_CROPS, PRODUCTS, ACRE_HA, FARMGATE_SHARE
from agro.advisor import haversine, market_price

REFRESH_HOURS = float(os.environ.get("REFRESH_HOURS", 24))
app = Flask(__name__)
APP_USER, APP_PASSWORD = os.environ.get("APP_USER", ""), os.environ.get("APP_PASSWORD", "")


@app.before_request
def require_login():
    """Optional password for sharing online: set APP_USER and APP_PASSWORD (docker-compose.yml).
    /api/sms and /api/ussd stay open so SMS/USSD gateways can reach them."""
    if not APP_PASSWORD or request.path in ("/api/sms", "/api/ussd"):
        return None
    a = request.authorization
    if a and a.username == APP_USER and a.password == APP_PASSWORD:
        return None
    from flask import Response
    return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="SokoMkulima"'})
_cache = {}
_lock = threading.Lock()
_next_check = None


def refresh(force=False):
    """Runs one live update (fetch -> retrain if changed). Never two at once."""
    if not _lock.acquire(blocking=False):
        return False
    try:
        updater.update(force=force, on_change=_cache.clear)        # each newly trained crop is served at once
        _cache.clear()
    except Exception as e:
        updater.write_status(message=f"Refresh failed: {e}")
    finally:
        _lock.release()
    return True


def refresher():
    global _next_check
    while True:
        refresh()                                                   # at start-up, then every REFRESH_HOURS
        _next_check = datetime.now(timezone.utc) + timedelta(hours=REFRESH_HOURS)
        time.sleep(REFRESH_HOURS * 3600)


def slugs():
    order = {c: i for i, c in enumerate(updater.crops())}
    return sorted((p.stem for p in PROC.glob("*.json")), key=lambda s: (order.get(s.rsplit("_", 1)[0], 99), s))


def methods_for(slug):
    """Trained models for this crop, BEST FIRST. The order is decided on the first half of the test months and
    the reported error comes from the second half (model/selection.py), so the choice does not flatter itself."""
    have = [m for m in METHOD_ORDER if (OUT / f"{slug}__{m}.json").exists()]
    try:
        order = selection(slug)["order"]
        return [m for m in order if m in have] + [m for m in have if m not in order]
    except Exception:
        return have


def get(slug, method=None):
    method = method or (methods_for(slug) or [None])[0]
    key = (slug, method)
    if key not in _cache:
        meta = json.loads((PROC / f"{slug}.json").read_text())
        z = np.load(PROC / f"{slug}.npz")
        res = json.loads((OUT / f"{slug}__{method}.json").read_text())
        W = np.load(OUT / f"{slug}__{method}__edges.npy")
        comp_p = OUT / f"{slug}__comparison.json"
        comp = json.loads(comp_p.read_text()) if comp_p.exists() else {}
        _cache[key] = dict(meta=meta, logp=z["logp"], edges=z["edges"], dist=z["edge_dist"], cross=z["cross"],
                           res=res, W=W, comp=comp, method=method,
                           idx={m["id"]: i for i, m in enumerate(meta["markets"])})
    return _cache[key]


def advice(fc, storage_pct=STORAGE_PCT, loss_pct=LOSS_PCT):
    cur = fc["current"]; best = {"months": 0, "net_pct": 0.0}
    rows = []
    for h, (f, lo) in enumerate(zip(fc["forecast"], fc["low"]), start=1):
        gross = (f / cur - 1) * 100
        net = gross - (storage_pct + loss_pct) * h
        rows.append({"months": h, "forecast": f, "low": lo, "gross_pct": round(gross, 1), "net_pct": round(net, 1),
                     "safe": lo > cur * (1 + (storage_pct + loss_pct) * h / 100)})
        if net > best["net_pct"]:
            best = {"months": h, "net_pct": round(net, 1)}
    if best["months"] and best["net_pct"] >= 3:
        action = "STORE"; conf = "high" if rows[best["months"] - 1]["safe"] else "medium"
    else:
        action = "SELL"; conf = "high" if all(r["net_pct"] < 0 for r in rows) else "medium"
    return {"action": action, "store_months": best["months"] if action == "STORE" else 0,
            "expected_gain_pct": best["net_pct"] if action == "STORE" else 0.0, "confidence": conf,
            "storage_pct": storage_pct, "loss_pct": loss_pct, "by_horizon": rows}


@app.get("/")
def index():
    return send_from_directory(FRONT, "index.html")


@app.get("/api/status")
def status():
    s = updater.read_status(); s.pop("signature", None)
    return jsonify(**s, busy=_lock.locked(), refresh_hours=REFRESH_HOURS,
                   next_check=_next_check.isoformat(timespec="seconds") if _next_check else None)


@app.post("/api/refresh")
def refresh_now():
    if _lock.locked():
        return jsonify(started=False, message="An update is already running."), 409
    last = updater.read_status().get("checked_at")
    if last and datetime.now(timezone.utc) - datetime.fromisoformat(last) < timedelta(minutes=10):
        return jsonify(started=False, message="Checked less than 10 minutes ago."), 429
    threading.Thread(target=refresh, daemon=True).start()
    return jsonify(started=True, message="Checking WFP for new prices...")


@app.get("/api/datasets")
def datasets():
    return jsonify([{"slug": s, "methods": methods_for(s)} for s in slugs() if methods_for(s)])


@app.get("/api/meta")
def meta():
    d = get(request.args["slug"], request.args.get("method"))
    m = d["meta"]
    return jsonify(commodity=m["commodity"], pricetype=m["pricetype"], unit=m["unit"], months=m["months"],
                   demo=m.get("demo", False), source=m.get("source"), data_as_of=m.get("data_as_of"),
                   countries=m.get("countries", []), built_at=m.get("built_at"), markets=m["markets"], method=d["method"],
                   methods=methods_for(request.args["slug"]), metrics=d["res"]["metrics"], comparison=d["comp"],
                   selection=selection(request.args["slug"]),
                   edges=d["edges"].tolist(), cross=d["cross"].tolist(), origin=d["res"]["origin"])


@app.get("/api/series")
def series():
    d = get(request.args["slug"], request.args.get("method"))
    prices = np.where(np.isnan(d["logp"]), np.nan, np.exp(d["logp"]))
    fc = d["res"]["forecasts"]
    chg3 = [round((fc[m["id"]]["forecast"][-1] / fc[m["id"]]["current"] - 1) * 100, 1) for m in d["meta"]["markets"]]
    return jsonify(prices=[[None if np.isnan(v) else round(float(v), 4) for v in row] for row in prices],
                   weights=np.round(d["W"], 3).tolist(), forecast_change_pct=chg3)


@app.get("/api/market")
def market():
    d = get(request.args["slug"], request.args.get("method"))
    mid = request.args["id"]; i = d["idx"][mid]
    meta, logp = d["meta"], d["logp"]
    fc = d["res"]["forecasts"][mid]
    nb = []
    t = len(meta["months"]) - 1
    for k, (a, b) in enumerate(d["edges"]):
        if i in (a, b):
            j = b if a == i else a
            nb.append({"id": meta["markets"][j]["id"], "name": meta["markets"][j]["name"],
                       "country": meta["markets"][j]["country"], "km": round(float(d["dist"][k])),
                       "cross_border": bool(d["cross"][k]), "weight": round(float(d["W"][t, k]), 3)})
    nb.sort(key=lambda r: -r["weight"])
    sp = float(request.args.get("storage", STORAGE_PCT)); lp = float(request.args.get("loss", LOSS_PCT))
    return jsonify(market=meta["markets"][i], months=meta["months"],
                   history=[None if np.isnan(v) else round(float(np.exp(v)), 4) for v in logp[:, i]],
                   forecast=fc, backtest=d["res"]["backtest_h1"][mid], neighbours=nb[:8],
                   advice=advice(fc, sp, lp), method=d["method"], unit=meta["unit"])


def find_market(words, markets):
    """Longest market name (in Latin letters, as WFP spells it) found in the leftover words."""
    text = f" {' '.join(words)} "
    best = None
    for m in markets:
        name = L.norm(m["name"])
        if name and (f" {name} " in text or (name.split()[0] in words and len(name.split()[0]) > 3)):
            if best is None or len(name) > len(L.norm(best["name"])):
                best = m
    return best


def crop_of(slug):
    return slug.rsplit("_", 1)[0]


def ready_slugs():
    return [s for s in slugs() if methods_for(s)]


def sms_answer(text):
    crop, lang, words = L.parse(text)
    options = ready_slugs()
    if not options:
        return {"reply": "Service starting: no trained model yet. Try again in a few minutes.", "language": lang}
    by_crop = {crop_of(s): s for s in options}
    crop = crop or ("maize" if "maize" in by_crop else crop_of(options[0]))
    names = lambda crops: ", ".join(L.crop_name(c, lang) for c in crops)
    if crop not in by_crop and L.SIBLINGS.get(crop) in by_crop:
        crop = L.SIBLINGS[crop]
    if crop not in by_crop:
        return {"reply": L.TEXT[lang]["nodata"].format(crop=L.crop_name(crop, lang), alt=names(by_crop)), "language": lang}
    d = get(by_crop[crop])
    m = find_market(words, d["meta"]["markets"])
    if not m:                                   # market exists, but only for other crops?
        elsewhere = [c for c, s in by_crop.items() if c != crop and find_market(words, get(s)["meta"]["markets"])]
        if elsewhere:
            mk = find_market(words, get(by_crop[elsewhere[0]])["meta"]["markets"])
            return {"reply": L.TEXT[lang]["nocrop"].format(crop=L.crop_name(crop, lang), market=mk["name"],
                                                          alt=names(elsewhere)), "language": lang}
        ex = next((x for x in d["meta"]["markets"] if L.COUNTRY_LANG.get(x["country"]) == lang), d["meta"]["markets"][0])
        return {"reply": L.TEXT[lang]["nomarket"].format(ex=L.example(lang, crop, ex["name"])), "language": lang}
    fc = d["res"]["forecasts"][m["id"]]; a = advice(fc)
    reply = L.reply(lang, crop, m["name"], fc["current"], fc["forecast"][-1], a["action"],
                    a["store_months"], a["expected_gain_pct"])
    reply += " [DEMO]" if d["meta"].get("demo") else f" (WFP {d['meta'].get('data_as_of', '')})"
    return {"reply": reply, "market": m["id"], "slug": by_crop[crop], "language": lang}


def all_markets():
    """Every market of every crop (for turning a place name into coordinates)."""
    seen = {}
    for s in ready_slugs():
        for m in get(s)["meta"]["markets"]:
            seen.setdefault(m["id"], m)
    return list(seen.values())


def sms_plan(text):
    crop, lang, words = L.parse(text)
    crop = crop or "maize"
    m = find_market(words, all_markets())
    if not m:
        return {"reply": L.TEXT[lang]["nomarket"].format(ex=f"{L.PLAN_WORDS[lang][0].upper()} {L.crop_name(crop, lang).upper()} KITALE"),
                "language": lang}
    try:
        p = advisor.plan(m["lat"], m["lon"], crop, farm_ha=ACRE_HA)
    except Exception as e:
        return {"reply": f"Season planner unavailable ({type(e).__name__}). Try again later.", "language": lang}
    return {"reply": L.plan_reply(lang, crop, m["name"], p), "language": lang, "kind": "plan",
            "place": {"name": m["name"], "lat": m["lat"], "lon": m["lon"]}, "crop": crop}


@app.post("/api/sms")
def sms():
    """Web simulator: JSON {"text", "from"?} -> JSON reply.
    Africa's Talking incoming-SMS callback: form fields from, to, text -> the reply is SENT back by SMS."""
    j = request.get_json(silent=True) or {}
    text = j.get("text") or request.form.get("text", "")
    phone = j.get("from") or request.form.get("from", "")
    if L.is_stop(text):
        r = sms_stop(text, phone)
    elif L.is_join(text):
        r = sms_join(text, phone)
    else:
        r = sms_insure(text) if L.is_insure(text) else sms_plan(text) if L.is_plan(text) else sms_answer(text)
    if request.form.get("from") and sms_gateway.configured():          # a real phone: answer by SMS
        f = db.q("SELECT * FROM farmers WHERE phone=?", (db.norm_phone(phone),), one=True) or {"id": 0, "phone": db.norm_phone(phone)}
        sms_gateway.send(f, "reply", db.now() + phone, r["reply"])
        return Response("OK", mimetype="text/plain")
    return jsonify(**r, language_name=L.LANGS.get(r["language"], r["language"]))


@app.get("/plan")
def plan_page():
    return send_from_directory(FRONT, "plan.html")


@app.get("/api/plan")
def plan_api():
    """Season plan for one field.  /api/plan?lat=1.02&lon=35&crop=maize&acres=1  [&urea=38&dap=48&npk=42&mop=40
    &ph=&soc=&p=&k= (soil test) &price= (farm-gate USD/kg) &season=0|1]"""
    a = request.args
    try:
        lat, lon = float(a["lat"]), float(a["lon"])
    except (KeyError, ValueError):
        return jsonify(error="lat and lon are required"), 400
    crop = a.get("crop", "maize")
    if crop not in AGRO_CROPS:
        return jsonify(error=f"unknown crop {crop}"), 400
    acres = float(a.get("acres", 1) or 1)
    soil = {"ph": a.get("ph"), "soc": a.get("soc"), "p_mehlich3": a.get("p"), "k_mehlich3": a.get("k")}
    try:
        p = advisor.plan(lat, lon, crop, farm_ha=acres * ACRE_HA, fert_prices={k: a.get(k) for k in PRODUCTS},
                         soil_overrides=soil, crop_price=a.get("price"),
                         season_index=int(a["season"]) if a.get("season", "") != "" else None)
    except Exception as e:
        return jsonify(error=f"Could not reach the weather or soil service ({type(e).__name__}: {e}). "
                             "Check the internet connection of the server."), 503
    p["acres"] = acres
    p["products_catalog"] = PRODUCTS
    return jsonify(p)


# ---------------- page 3: crop insurance ----------------
_ins = {}
MAX_RATE = 0.20          # above a 20 % premium a policy is not affordable -> not offered


def insure_data():
    p = OUT / "insure_model.json"
    if not p.exists():
        return None
    m = p.stat().st_mtime
    if _ins.get("mtime") != m:
        d = json.loads(p.read_text())
        _ins.update(mtime=m, data=d, units=json.loads((ROOT / "insure" / "data" / "units.json").read_text()))
        try:
            import insure.live as live
            live._live = None                    # reload the model after retraining
        except Exception:
            pass
    return _ins["data"]


def district_skill(pol, trig_bad):
    h = pol["history"]
    bad = [x for x in h if x["actual_loss"] >= trig_bad]
    hits = [x for x in bad if x["payout"] > 0]
    false = [x for x in h if x["payout"] > 0 and x["actual_loss"] < trig_bad]
    return {"years": len(h), "bad": len(bad), "hits": len(hits), "false_alarms": len(false),
            "hit_rate": len(hits) / len(bad) if bad else None}


def offered(pol, sk):
    return pol["premium_rate"] <= MAX_RATE and (sk["bad"] < 2 or (sk["hit_rate"] or 0) >= 0.4)


def policy_quote(key, acres=1.0, price=None, live_now=True):
    d = insure_data(); pol = d["policies"][key]; meta = d["meta"]
    mp = market_price(pol["crop"], pol["lat"], pol["lon"])
    farmgate = float(price) if price not in (None, "") else FARMGATE_SHARE * float(np.mean([mp["retail_now"], mp["retail_3m"]]))
    ha = acres * ACRE_HA
    harvest_kg = pol["expected_t_ha"] * 1000 * ha
    si = harvest_kg * farmgate
    sk = district_skill(pol, meta["bad_threshold"])
    q = {**pol, "acres": acres, "expected_kg": round(harvest_kg), "price_kg": round(farmgate, 3), "price_info": mp,
         "sum_insured": round(si, 2), "premium": round(pol["premium_rate"] * si, 2), "trigger": meta["trigger"],
         "trigger_on": meta.get("trigger_on", "predicted loss"), "bad_threshold": meta["bad_threshold"],
         "model": meta["best"], "skill": sk, "offered": offered(pol, sk),
         "history_usd": [{**x, "usd": round(x["payout"] * si, 2)} for x in pol["history"]]}
    if live_now:
        try:
            import insure.live as live
            q["season_now"] = live.season_now(pol)
        except Exception as e:
            q["season_now"] = {"state": "unavailable", "error": f"{type(e).__name__}: {e}"}
    return q


@app.get("/insure")
def insure_page():
    return send_from_directory(FRONT, "insure.html")


@app.get("/api/insure/meta")
def insure_meta():
    d = insure_data()
    if not d:
        return jsonify(ready=False, message="The insurance model is still training (it downloads rainfall for ~650 "
                                            "places the first time). This page fills in by itself.")
    meta = d["meta"]; units = _ins["units"]
    pols = []
    for k, p in d["policies"].items():
        sk = district_skill(p, meta["bad_threshold"])
        pols.append({"key": k, "fnid": p["fnid"], "crop": p["crop"], "season": p["season"], "name": p["name"],
                     "country": p["country"], "lat": p["lat"], "lon": p["lon"], "rate": p["premium_rate"],
                     "offered": offered(p, sk), "hit_rate": sk["hit_rate"]})
    used = {p["fnid"] for p in pols}
    return jsonify(ready=True, meta=meta, policies=pols,
                   outlines={f: units[f]["outline"] for f in used if f in units})


@app.get("/api/insure/policy")
def insure_policy():
    if not insure_data():
        return jsonify(error="insurance model not trained yet"), 503
    key = request.args.get("key", "")
    if key not in _ins["data"]["policies"]:
        return jsonify(error="unknown district policy"), 404
    return jsonify(policy_quote(key, float(request.args.get("acres", 1) or 1), request.args.get("price")))


def sms_insure(text):
    crop, lang, words = L.parse(text)
    crop = crop if crop in ("maize", "sorghum", "millet") else "maize"
    d = insure_data()
    if not d:
        return {"reply": "Insurance model is still training. Try again later.", "language": lang}
    m = find_market(words, all_markets())
    if not m:
        return {"reply": L.TEXT[lang]["nomarket"].format(ex=f"{L.INSURE_WORDS[lang][0].upper()} {L.crop_name(crop, lang).upper()} KITALE"),
                "language": lang}
    cands = [(haversine(m["lat"], m["lon"], p["lat"], p["lon"]), k) for k, p in d["policies"].items() if p["crop"] == crop]
    if not cands or min(cands)[0] > 150:
        return {"reply": L.INSURE_TEXT[lang]["no"].format(crop=L.crop_name(crop, lang), place=m["name"]), "language": lang}
    key = min(cands)[1]
    q = policy_quote(key, 1.0)
    return {"reply": L.insure_reply(lang, crop, m["name"], q), "language": lang, "kind": "insure", "policy": key}


@app.get("/api/plan/meta")
def plan_meta():
    return jsonify(crops=list(AGRO_CROPS), products=PRODUCTS,
                   places=[{"name": m["name"], "country": m["country"], "lat": m["lat"], "lon": m["lon"]}
                           for m in all_markets()] if ready_slugs() else [])


@app.get("/api/sms/examples")
def sms_examples():
    """One example message per language, using real markets of this crop in countries that speak it."""
    slug = request.args.get("slug") or (ready_slugs() or [None])[0]
    if not slug or not methods_for(slug):
        return jsonify([])
    crop = crop_of(slug); mk = get(slug)["meta"]["markets"]; out = []
    for lang in L.LANGS:
        if crop not in L.CROP_WORDS or lang not in L.CROP_WORDS[crop]:
            continue
        m = next((x for x in mk if L.COUNTRY_LANG.get(x["country"]) == lang), None)
        if m is None and lang == "en":
            m = mk[0]
        if m:
            out.append({"lang": lang, "language": L.LANGS[lang], "text": L.example(lang, crop, m["name"])})
    # season-plan examples (fertilizer + planting dates)
    for lang in ("en", "sw", "fr"):
        m = next((x for x in mk if L.COUNTRY_LANG.get(x["country"]) == lang), None)
        if m and lang in L.CROP_WORDS.get(crop, {}):
            out.append({"lang": lang, "language": L.LANGS[lang], "kind": "plan",
                        "text": f"{L.PLAN_WORDS[lang][0].upper()} {L.crop_name(crop, lang).upper()} {m['name'].upper()}"})
    return jsonify(out)



# =====================================================================================================
# ADDITIONS: where to sell · fertilizer + insurance bundle · My farm (whole year) · USSD menus
# =====================================================================================================
TRANSPORT_USD_PER_TKM = float(os.environ.get("TRANSPORT_USD_PER_TKM", 0.12))   # small loads on rural roads
TRIP_FIXED_USD = float(os.environ.get("TRIP_FIXED_USD", 5.0))                  # loading, market fee, own travel
ROAD_FACTOR = 1.3                                                               # roads are ~30 % longer than a straight line


def where_to_sell(slug, mid, qty_kg=500.0, rate=TRANSPORT_USD_PER_TKM, max_km=300):
    """Nearby markets for this crop ranked by what the farmer keeps after transport (selling now)."""
    d = get(slug); meta = d["meta"]; i = d["idx"][mid]; home = meta["markets"][i]
    fc = d["res"]["forecasts"]
    p_home = fc[mid]["current"] * FARMGATE_SHARE
    rows = []
    for m in meta["markets"]:
        km = haversine(home["lat"], home["lon"], m["lat"], m["lon"])
        if m["id"] == mid or km > max_km:
            continue
        p = fc[m["id"]]["current"] * FARMGATE_SHARE
        cost_kg = rate * km * ROAD_FACTOR / 1000 + TRIP_FIXED_USD / max(qty_kg, 1)
        gain_kg = p - p_home - cost_kg
        rows.append({"id": m["id"], "name": m["name"], "country": m["country"], "km": round(km), "price_kg": round(p, 3),
                     "transport_kg": round(cost_kg, 3), "gain_kg": round(gain_kg, 3), "gain_total": round(gain_kg * qty_kg, 2),
                     "cross_border": m["country"] != home["country"]})
    rows.sort(key=lambda r: -r["gain_kg"])
    best = rows[0] if rows and rows[0]["gain_kg"] > 0.005 else None
    return {"home": {"id": mid, "name": home["name"], "country": home["country"], "price_kg": round(p_home, 3)},
            "qty_kg": qty_kg, "rate_usd_tkm": rate, "best": best, "options": rows[:8],
            "note": "Farm-gate prices (~75% of retail), selling now. Cross-border trade may need permits or pay duties."}


@app.get("/api/where_to_sell")
def where_to_sell_api():
    a = request.args
    try:
        return jsonify(where_to_sell(a["slug"], a["id"], float(a.get("qty", 500) or 500),
                                     float(a.get("rate", TRANSPORT_USD_PER_TKM) or TRANSPORT_USD_PER_TKM)))
    except KeyError:
        return jsonify(error="slug and id are required"), 400


def nearest_policy(crop, lat, lon, max_km=150):
    d = insure_data()
    if not d:
        return None
    c = [(haversine(lat, lon, p["lat"], p["lon"]), k) for k, p in d["policies"].items() if p["crop"] == crop]
    if not c or min(c)[0] > max_km:
        return None
    return min(c)[1]


def bundle(crop, lat, lon, fert_cost):
    """Fertilizer + insurance: if the district's weather index says the season failed, the farmer gets the
    fertilizer money back. Premium = how often it would have paid x cost x loading."""
    key = nearest_policy(crop, lat, lon)
    if not key or fert_cost <= 0:
        return {"available": False, "reason": "no insured district nearby for this crop" if not key else "no fertilizer bought"}
    d = insure_data(); pol = d["policies"][key]; meta = d["meta"]
    sk = district_skill(pol, meta["bad_threshold"])
    freq = float(np.mean([h["payout"] > 0 for h in pol["history"]])) if pol["history"] else 0.0
    premium = freq * fert_cost * meta["loading"]
    ok = offered(pol, sk)
    return {"available": True, "offered": ok, "district": pol["name"], "country": pol["country"], "policy": key,
            "fert_cost": round(fert_cost, 2), "premium": round(premium, 2), "pay_frequency": round(freq, 3),
            "skill": sk, "total": round(fert_cost + premium, 2),
            "rule": "If the season's weather shows the harvest failed, your fertilizer money is paid back by mobile money."}


@app.get("/api/bundle")
def bundle_api():
    a = request.args
    try:
        return jsonify(bundle(a.get("crop", "maize"), float(a["lat"]), float(a["lon"]), float(a.get("cost", 0) or 0)))
    except (KeyError, ValueError):
        return jsonify(error="lat, lon and cost are required"), 400


def nearest_market(crop, lat, lon, max_km=400):
    slug = next((s for s in ready_slugs() if crop_of(s) == crop), None)
    if not slug:
        return None, None
    ms = get(slug)["meta"]["markets"]
    m = min(ms, key=lambda x: haversine(lat, lon, x["lat"], x["lon"]))
    return (slug, m) if haversine(lat, lon, m["lat"], m["lon"]) <= max_km else (slug, None)


@app.get("/farm")
def farm_page():
    return send_from_directory(FRONT, "farm.html")


@app.get("/api/farm")
def farm_api():
    """The farmer's whole year in one answer: plan the season, protect it, then sell or store and where."""
    a = request.args
    try:
        lat, lon = float(a["lat"]), float(a["lon"])
    except (KeyError, ValueError):
        return jsonify(error="lat and lon are required"), 400
    crop = a.get("crop", "maize"); acres = float(a.get("acres", 1) or 1)
    out = {"crop": crop, "acres": acres, "lat": lat, "lon": lon}
    try:
        p = advisor.plan(lat, lon, crop, farm_ha=acres * ACRE_HA)
        out["plan"] = {k: p[k] for k in ("ok", "season", "all_seasons", "yield_gap", "plan", "soil_supply_kg_ha",
                                          "advice_flags", "price") if k in p} if p.get("ok") else {"ok": False, "message": p.get("message")}
    except Exception as e:
        out["plan"] = {"ok": False, "message": f"Season planner unavailable ({type(e).__name__})."}
    cost = (out["plan"].get("plan") or {}).get("cost_farm", 0) if out["plan"].get("ok") else 0
    out["bundle"] = bundle(crop, lat, lon, cost) if crop in ("maize", "sorghum", "millet") else {"available": False, "reason": "insurance covers maize, sorghum and millet"}
    key = nearest_policy(crop, lat, lon) if crop in ("maize", "sorghum", "millet") else None
    if key:
        q = policy_quote(key, acres)
        out["insurance"] = {k: q[k] for k in ("name", "country", "season", "offered", "premium", "sum_insured", "premium_rate",
                                             "skill", "season_now", "trigger", "trigger_on", "bad_threshold")}
    slug, m = nearest_market(crop, lat, lon)
    if m:
        d = get(slug); fc = d["res"]["forecasts"][m["id"]]
        harvest_kg = (out["plan"].get("yield_gap") or {}).get("with_plan", {}).get("kg_farm") or 500
        out["market"] = {"name": m["name"], "country": m["country"], "id": m["id"], "slug": slug,
                         "km": round(haversine(lat, lon, m["lat"], m["lon"])), "forecast": fc, "advice": advice(fc),
                         "method": d["method"], "demo": d["meta"].get("demo", False), "as_of": d["meta"].get("data_as_of")}
        out["where_to_sell"] = where_to_sell(slug, m["id"], float(harvest_kg))
    return jsonify(out)


# ---------------- USSD (*384#): numbered menus for basic phones, Africa's Talking webhook format ----------------
USSD_TEXT = {
    "en": dict(main="Welcome to SokoMkulima\n1. Prices: sell or store\n2. Plan my season\n3. Crop insurance\n4. Language",
               crop="Choose crop:\n1. Maize\n2. Sorghum\n3. Millet\n4. Rice\n5. Beans", place="Enter your market or town (e.g. KITALE):",
               lang="Language:\n1. English\n2. Kiswahili\n3. Francais", bad="Invalid choice. Dial again.", sms="Full reply sent by SMS."),
    "sw": dict(main="Karibu SokoMkulima\n1. Bei: uza au hifadhi\n2. Panga msimu wangu\n3. Bima ya mazao\n4. Lugha",
               crop="Chagua zao:\n1. Mahindi\n2. Mtama\n3. Uwele\n4. Mchele\n5. Maharagwe", place="Andika soko au mji wako (mfano KITALE):",
               lang="Lugha:\n1. English\n2. Kiswahili\n3. Francais", bad="Chaguo si sahihi. Piga tena.", sms="Jibu kamili limetumwa kwa SMS."),
    "fr": dict(main="Bienvenue sur SokoMkulima\n1. Prix : vendre ou stocker\n2. Planifier ma saison\n3. Assurance recolte\n4. Langue",
               crop="Choisissez la culture :\n1. Mais\n2. Sorgho\n3. Mil\n4. Riz\n5. Haricots", place="Entrez votre marche ou ville (ex. BAMAKO) :",
               lang="Langue :\n1. English\n2. Kiswahili\n3. Francais", bad="Choix invalide. Recomposez.", sms="Reponse complete envoyee par SMS."),
}
USSD_CROPS = ["maize", "sorghum", "millet", "rice", "beans"]
USSD_LANGS = ["en", "sw", "fr"]
USSD_MAX = 182                                    # characters on one USSD screen


def ussd_reply(text):
    parts = [p.strip() for p in (text or "").split("*")] if text else []
    lang = "en"
    while parts and parts[0] == "4":                 # language menu (can be chosen before anything else)
        if len(parts) == 1:
            return "CON " + USSD_TEXT[lang]["lang"]
        try:
            lang = USSD_LANGS[int(parts[1]) - 1]
        except (ValueError, IndexError):
            return "END " + USSD_TEXT[lang]["bad"]
        parts = parts[2:]
    T = USSD_TEXT[lang]
    if not parts:
        return "CON " + T["main"]
    service = parts[0]
    if service not in ("1", "2", "3"):
        return "END " + T["bad"]
    if len(parts) == 1:
        return "CON " + T["crop"]
    try:
        crop = USSD_CROPS[int(parts[1]) - 1]
    except (ValueError, IndexError):
        return "END " + T["bad"]
    if len(parts) == 2:
        return "CON " + T["place"]
    place = " ".join(parts[2:])
    word = {"1": L.PRICE_WORDS, "2": L.PLAN_WORDS, "3": L.INSURE_WORDS}[service][lang][0]
    msg = f"{word} {L.crop_name(crop, lang)} {place}"
    r = sms_insure(msg) if service == "3" else sms_plan(msg) if service == "2" else sms_answer(msg)
    full = r["reply"]
    short = full if len(full) <= USSD_MAX else full[:USSD_MAX - len(T["sms"]) - 2].rstrip() + "… " + T["sms"]
    return "END " + short, full


@app.post("/api/ussd")
def ussd():
    """Africa's Talking USSD callback: form fields sessionId, serviceCode, phoneNumber, text ('1*2*KITALE').
    Replies plain text starting with CON (show menu, wait) or END (final screen)."""
    text = request.form.get("text", request.args.get("text", ""))
    r = ussd_reply(text)
    if isinstance(r, tuple):
        screen, full = r
        if request.args.get("debug"):
            return jsonify(screen=screen, sms=full)
        return Response(screen, mimetype="text/plain")
    if request.args.get("debug"):
        return jsonify(screen=r)
    return Response(r, mimetype="text/plain")


# =====================================================================================================
# FARMERS, AUTOMATIC SMS ALERTS, ADMIN, COOPERATIVES
# =====================================================================================================
from datetime import date as _date
from service import db, sms_gateway, alerts as alert_engine

ALERT_HOURS = float(os.environ.get("ALERT_HOURS", 24))
TRUCK_USD_PER_TKM = float(os.environ.get("TRUCK_USD_PER_TKM", 0.06))     # full truck: cheaper per tonne


def place_coords(place=None, lat=None, lon=None):
    if lat not in (None, "") and lon not in (None, ""):
        return float(lat), float(lon), place or f"{float(lat):.2f},{float(lon):.2f}"
    m = next((x for x in all_markets() if x["name"].lower() == str(place or "").strip().lower()), None)
    if not m:
        m = find_market(L.norm(place or "").split(), all_markets())
    return (m["lat"], m["lon"], m["name"]) if m else (None, None, place)


def register_farmer(phone, crop, place=None, lat=None, lon=None, lang="en", name="", acres=1.0, coop_id=None):
    phone = db.norm_phone(phone)
    if not phone:
        raise ValueError("phone number needed")
    la, lo, pname = place_coords(place, lat, lon)
    if la is None:
        raise ValueError(f"place '{place}' not found: use a market town name or coordinates")
    old = db.q("SELECT id FROM farmers WHERE phone=?", (phone,), one=True)
    if old:
        db.q("UPDATE farmers SET crop=?, place=?, lat=?, lon=?, lang=?, name=COALESCE(NULLIF(?,''),name), acres=?, "
             "coop_id=COALESCE(?,coop_id), active=1 WHERE id=?", (crop, pname, la, lo, lang, name, acres, coop_id, old["id"]))
        fid = old["id"]
    else:
        fid = db.insert("INSERT INTO farmers(phone,name,lang,crop,place,lat,lon,acres,coop_id,active,created) "
                        "VALUES(?,?,?,?,?,?,?,?,?,1,?)", (phone, name, lang, crop, pname, la, lo, acres, coop_id, db.now()))
    return db.q("SELECT * FROM farmers WHERE id=?", (fid,), one=True)


def _alert_tools():
    import insure.live as live

    def plan(lat, lon, crop, acres, t):
        return advisor.plan(lat, lon, crop, farm_ha=acres * ACRE_HA, today_doy=t)

    def price_reply(lang, crop, place):
        word = L.PRICE_WORDS[lang][0]
        r = sms_answer(f"{word} {L.crop_name(crop, lang)} {place}")
        return r["reply"] if r.get("market") else ""

    def where_best(crop, lat, lon, qty):
        slug, m = nearest_market(crop, lat, lon)
        if not m:
            return None
        return where_to_sell(slug, m["id"], float(qty or 500))["best"]

    def season_status(crop, lat, lon, acres, today):
        if crop not in ("maize", "sorghum", "millet"):
            return None
        key = nearest_policy(crop, lat, lon)
        if not key:
            return None
        q = policy_quote(key, acres, live_now=False)
        q["season_now"] = live.season_now(_ins["data"]["policies"][key], today)
        return q

    return {"plan": plan, "plan_reply": L.plan_reply, "price_reply": price_reply, "where_best": where_best,
            "season_status": season_status, "insure_reply": L.insure_reply}


def run_alerts(today=None, farmer_id=None):
    return alert_engine.run(_alert_tools(), today, farmer_id)


def alerter():
    time.sleep(120)                                   # let the first data update start first
    while True:
        try:
            r = run_alerts()
            print(f"[alerts] {len(r['sent'])} SMS for {r['farmers']} farmers", flush=True)
        except Exception as e:
            print(f"[alerts] failed: {e}", flush=True)
        try:
            if run_warning():
                print("[warning] early-warning map updated", flush=True)
        except Exception as e:
            print(f"[warning] failed: {e}", flush=True)
        time.sleep(ALERT_HOURS * 3600)


def sms_join(text, phone):
    lang = L.is_join(text)
    crop, plang, words = L.parse(text)
    lang = plang if plang != "en" else lang
    crop = crop or "maize"
    m = find_market(words, all_markets())
    if not phone:
        return {"reply": "Send this from your phone to register.", "language": lang}
    if not m:
        return {"reply": L.TEXT[lang]["nomarket"].format(ex=f"{L.JOIN_WORDS[lang][0].upper()} {L.crop_name(crop, lang).upper()} KITALE"),
                "language": lang}
    register_farmer(phone, crop, m["name"], m["lat"], m["lon"], lang)
    return {"reply": L.JOIN_TEXT[lang]["ok"].format(crop=L.crop_name(crop, lang), place=m["name"]), "language": lang,
            "kind": "join"}


def sms_stop(text, phone):
    lang = L.is_stop(text)
    if phone:
        db.q("UPDATE farmers SET active=0 WHERE phone=?", (db.norm_phone(phone),))
    return {"reply": L.JOIN_TEXT[lang]["stop"], "language": lang, "kind": "stop"}


@app.get("/admin")
def admin_page():
    return send_from_directory(FRONT, "admin.html")


@app.get("/api/farmers")
def farmers_list():
    rows = db.q("SELECT f.*, c.name AS coop FROM farmers f LEFT JOIN coops c ON c.id=f.coop_id ORDER BY f.id DESC")
    return jsonify(farmers=rows, gateway=("Africa's Talking " + ("sandbox" if sms_gateway.USER == "sandbox" else "live"))
                   if sms_gateway.configured() else "none (messages kept in the outbox)")


@app.post("/api/farmers")
def farmers_add():
    a = request.get_json(silent=True) or request.form
    try:
        f = register_farmer(a.get("phone"), a.get("crop", "maize"), a.get("place"), a.get("lat"), a.get("lon"),
                            a.get("lang", "en"), a.get("name", ""), float(a.get("acres") or 1),
                            int(a["coop_id"]) if a.get("coop_id") else None)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    lang = f["lang"] if f["lang"] in L.JOIN_TEXT else "en"
    msg = L.JOIN_TEXT[lang]["ok"].format(crop=L.crop_name(f["crop"], lang), place=f["place"])
    sms_gateway.send(f, "welcome", f"{f['created']}", "SokoMkulima: " + msg)
    return jsonify(farmer=f, message=msg)


@app.post("/api/farmers/<int:fid>/stop")
def farmers_stop(fid):
    db.q("UPDATE farmers SET active=0 WHERE id=?", (fid,))
    return jsonify(ok=True)


@app.get("/api/messages")
def messages_list():
    return jsonify(messages=db.q("SELECT * FROM messages ORDER BY id DESC LIMIT 300"))


@app.post("/api/alerts/run")
def alerts_run():
    """Run the alert engine now. ?today=2026-03-20 simulates a date (for demos); &farmer=<id> for one farmer."""
    a = request.args
    try:
        today = _date.fromisoformat(a["today"]) if a.get("today") else None
    except ValueError:
        return jsonify(error="today must be YYYY-MM-DD"), 400
    return jsonify(run_alerts(today, int(a["farmer"]) if a.get("farmer") else None))


# ---------------- cooperatives ----------------
@app.get("/coop")
def coop_page():
    return send_from_directory(FRONT, "coop.html")


@app.get("/api/coops")
def coops_list():
    return jsonify(coops=db.q("SELECT c.*, COUNT(f.id) AS members, COALESCE(SUM(f.acres),0) AS acres FROM coops c "
                              "LEFT JOIN farmers f ON f.coop_id=c.id AND f.active=1 GROUP BY c.id ORDER BY c.id DESC"))


@app.post("/api/coops")
def coops_add():
    a = request.get_json(silent=True) or {}
    la, lo, pname = place_coords(a.get("place"), a.get("lat"), a.get("lon"))
    if la is None or not a.get("name"):
        return jsonify(error="name and a known place (or coordinates) are needed"), 400
    cid = db.insert("INSERT INTO coops(name,crop,place,lat,lon,discount_pct,created) VALUES(?,?,?,?,?,?,?)",
                    (a["name"], a.get("crop", "maize"), pname, la, lo, float(a.get("discount_pct") or 10), db.now()))
    return jsonify(coop=db.q("SELECT * FROM coops WHERE id=?", (cid,), one=True))


@app.post("/api/coops/<int:cid>/members")
def coop_member_add(cid):
    c = db.q("SELECT * FROM coops WHERE id=?", (cid,), one=True)
    if not c:
        return jsonify(error="no such cooperative"), 404
    a = request.get_json(silent=True) or {}
    try:
        f = register_farmer(a.get("phone"), c["crop"], c["place"], c["lat"], c["lon"], a.get("lang", "en"),
                            a.get("name", ""), float(a.get("acres") or 1), cid)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(farmer=f)


@app.post("/api/coops/<int:cid>/members/<int:fid>/remove")
def coop_member_remove(cid, fid):
    db.q("UPDATE farmers SET coop_id=NULL WHERE id=? AND coop_id=?", (fid, cid))
    return jsonify(ok=True)


@app.post("/api/coops/<int:cid>/message")
def coop_message(cid):
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
    if not text:
        return jsonify(error="empty message"), 400
    key = db.now()
    sent = [sms_gateway.send(f, "coop", key, "SokoMkulima: " + text)
            for f in db.q("SELECT * FROM farmers WHERE coop_id=? AND active=1", (cid,))]
    return jsonify(sent=len([x for x in sent if x]))


@app.get("/api/coops/<int:cid>")
def coop_dashboard(cid):
    """Whole-group view: one bulk fertilizer order with discount, group harvest, selling together with a truck."""
    c = db.q("SELECT * FROM coops WHERE id=?", (cid,), one=True)
    if not c:
        return jsonify(error="no such cooperative"), 404
    members = db.q("SELECT * FROM farmers WHERE coop_id=? AND active=1 ORDER BY name", (cid,))
    acres = sum(m["acres"] or 1 for m in members) or 1.0
    out = {"coop": c, "members": members, "acres": acres}
    try:
        p = advisor.plan(c["lat"], c["lon"], c["crop"], farm_ha=acres * ACRE_HA)
    except Exception as e:
        return jsonify(**out, error=f"season planner unavailable ({type(e).__name__})")
    if not p.get("ok"):
        return jsonify(**out, plan={"ok": False, "message": p.get("message")})
    pl, disc = p["plan"], (c["discount_pct"] or 0) / 100
    bulk = {"products": pl["products"], "list_cost": pl["cost_farm"], "discount_pct": c["discount_pct"],
            "bulk_cost": round(pl["cost_farm"] * (1 - disc), 2), "saving": round(pl["cost_farm"] * disc, 2)}
    shares = [{"id": m["id"], "name": m["name"] or m["phone"], "phone": m["phone"], "acres": m["acres"] or 1,
               "pays": round(bulk["bulk_cost"] * (m["acres"] or 1) / acres, 2),
               "harvest_kg": round(p["yield_gap"]["with_plan"]["kg_farm"] * (m["acres"] or 1) / acres)} for m in members]
    out.update(plan={"ok": True, "season": p["season"], "yield_gap": p["yield_gap"], "price": p["price"],
                     "profit_farm": pl["profit_farm"], "recommended": pl["recommended"]}, bulk=bulk, shares=shares)
    harvest = p["yield_gap"]["with_plan"]["kg_farm"]
    slug, m = nearest_market(c["crop"], c["lat"], c["lon"])
    if m:
        group = where_to_sell(slug, m["id"], float(harvest), TRUCK_USD_PER_TKM)
        alone = where_to_sell(slug, m["id"], float(harvest) / max(len(members), 1), TRANSPORT_USD_PER_TKM)
        out["selling"] = {"market": m["name"], "group": group, "alone": alone, "truck_rate": TRUCK_USD_PER_TKM,
                          "small_rate": TRANSPORT_USD_PER_TKM}
    if c["crop"] in ("maize", "sorghum", "millet"):
        out["bundle"] = bundle(c["crop"], c["lat"], c["lon"], bulk["bulk_cost"])
    return jsonify(out)


# ---------------- hunger early warning ----------------
from service import warning as early_warning
WARN_PATH = OUT / "warning.json"
_warn_lock = threading.Lock()
_warn_state = {"running": False, "progress": None}


def price_info(lat, lon, max_km=300):
    for crop in ("maize", "sorghum", "millet", "rice"):
        slug, m = nearest_market(crop, lat, lon, max_km)
        if not m:
            continue
        d = get(slug); i = d["idx"][m["id"]]; fc = d["res"]["forecasts"][m["id"]]
        col = d["logp"][:, i]; ok = np.where(~np.isnan(col))[0]
        recent = None
        if len(ok) >= 2:
            last = ok[-1]; prev = ok[ok <= last - 3]
            if len(prev):
                recent = float((np.exp(col[last] - col[prev[-1]]) - 1) * 100)
        return {"crop": crop, "market": m["name"], "ahead_pct": (fc["forecast"][-1] / fc["current"] - 1) * 100,
                "recent_pct": recent}
    return None


def run_warning(today=None):
    if not _warn_lock.acquire(blocking=False):
        return False
    try:
        d = insure_data()
        if not d:
            return False
        import insure.live as live
        _warn_state.update(running=True, progress="starting")
        early_warning.compute(d["policies"], d["meta"]["bad_threshold"], live.season_now, price_info, WARN_PATH,
                              progress=lambda a, b: _warn_state.update(progress=f"{a}/{b} districts"), today=today)
    except Exception as e:
        _warn_state.update(error=f"{type(e).__name__}: {e}")
    finally:
        _warn_state.update(running=False, progress=None)
        _warn_lock.release()
    return True


@app.get("/warning")
def warning_page():
    return send_from_directory(FRONT, "warning.html")


@app.get("/api/warning")
def warning_api():
    base = {"running": _warn_state["running"], "progress": _warn_state["progress"], "error": _warn_state.get("error")}
    if not WARN_PATH.exists():
        return jsonify(ready=False, **base, insurance_ready=bool(insure_data()))
    w = json.loads(WARN_PATH.read_text())
    insure_data()
    units = _ins.get("units", {})
    return jsonify(ready=True, **base, **w, outlines={r["fnid"]: units[r["fnid"]]["outline"] for r in w["districts"] if r["fnid"] in units})


@app.post("/api/warning/run")
def warning_run():
    if _warn_state["running"]:
        return jsonify(started=False, message="Already running"), 409
    if not insure_data():
        return jsonify(started=False, message="The insurance model must be trained first"), 409
    threading.Thread(target=run_warning, daemon=True).start()
    return jsonify(started=True)


# ---------------- crop disease from a photo ----------------
from disease import predict as disease_model, advice as disease_advice


@app.get("/diagnose")
def diagnose_page():
    return send_from_directory(FRONT, "diagnose.html")


from disease import universal, care as leaf_care, plants as leaf_plants


def _spec_crops():
    return sorted({c.split("_")[0] for c in disease_model.meta()["classes"]}) if disease_model.ready() else []


@app.get("/api/diagnose/meta")
def diagnose_meta():
    """The universal check (any plant, CLIP) works as soon as CLIP is installed; the specialist CNN after training."""
    spec, uni = disease_model.ready(), universal.ready()
    plants = sorted(((k, v[0]) for k, v in leaf_plants.PLANTS.items()), key=lambda x: x[1])
    if not spec and not uni:
        st = updater.read_status()
        return jsonify(ready=False, message="The photo models are still being set up (downloading and training). "
                                            "This page updates by itself.", status=st.get("message"))
    out = dict(ready=True, universal=universal.meta(), plants=plants, specialist=None, crops=_spec_crops())
    if spec:
        m = disease_model.meta()
        out["specialist"] = dict(crops=out["crops"], test_accuracy=m["test_accuracy"], by_source=m.get("by_source", {}),
                                 n_test=m.get("n_test"), classes=len(m["classes"]))
    return jsonify(out)


def _leaf(data, lang, crop):
    """Universal check (plant? which? healthy?) + specialist CNN for its crops -> one answer with advice."""
    T = leaf_care.TEXT[leaf_care.lang_ok(lang)]
    spec_crops = _spec_crops()
    u = None
    if universal.ready():
        try:
            u = universal.analyse(data, lang, crop)
        except Exception as e:
            print(f"universal leaf check failed: {type(e).__name__}: {e}", flush=True)
    if u and not u["is_plant"] and not crop:
        return dict(kind="not_plant", message=T["not_plant"], p_plant=u["p_plant"])
    plant = u["plant"] if u else ({"key": crop, "name": leaf_plants.name(crop, lang), "prob": 1.0, "source": "you"} if crop else None)
    key = plant["key"] if plant else None
    # specialist: when the plant is one of its crops (or nothing else is available)
    spec = None
    near = {a["key"] for a in ((u or {}).get("alternatives") or []) if a.get("prob", 0) >= 0.1}
    if spec_crops and u and key not in spec_crops and plant["source"] == "clip" and near & set(spec_crops):
        key2 = sorted(near & set(spec_crops))[0]          # e.g. cassava as 2nd guess: let the specialist look too
    else:
        key2 = None
    if spec_crops and ((key in spec_crops and (plant["prob"] >= 0.15 or plant["source"] != "clip")) or key2 or u is None):
        r = disease_model.diagnose(data, key if key in spec_crops else key2)
        spec = dict(sure=r["sure"], p_sick=round(r.get("p_sick", 0.5), 3),
                    top=[{"label": l, "prob": round(p, 3), **disease_advice.advice(l, lang)} for l, p in r["top"]])
        if u is None and not plant:
            if not r["sure"]:
                # only 4 crops known and no clear match: do NOT force it into one of them
                return dict(kind="unknown", message=T["spec_only"].format(list=", ".join(spec_crops)))
            c = r["top"][0][0].split("_")[0]
            plant = {"key": c, "name": leaf_plants.name(c, lang), "prob": None, "source": "specialist"}
            key = c
    # health: BOTH models vote when both looked at the photo. The specialist alone was trained on close-ups of single
    # (mostly sick) leaves, so on a whole healthy field it can be confidently wrong (healthy cassava -> "green mite").
    t_u = universal.threshold()
    if spec and u:
        p_sick, src, t = 0.5 * spec["p_sick"] + 0.5 * u["p_sick"], "both", (0.5 + t_u) / 2
    elif u:
        p_sick, src, t = u["p_sick"], "universal", t_u
    elif spec:
        p_sick, src, t = spec["p_sick"], "specialist", 0.5
    else:
        raise RuntimeError("no photo model available")
    status = universal.status_of(p_sick, t)
    if u and u.get("autumn"):
        status = "unsure"          # whole leaf yellow/orange/red: most likely natural, never a confident verdict
    symptom = others = None
    if u and status != "healthy":
        g, pr = u["symptoms"][0]
        symptom = dict(leaf_care.symptom(g, lang), prob=pr)
        others = [dict(group=g2, name=leaf_care.symptom(g2, lang)["name"], prob=p2) for g2, p2 in u["symptoms"][1:] if p2 >= 0.05]
    supported = key in spec_crops
    pname = plant["name"] if plant else T["unknown_plant"]
    old_leaf = bool(u and (u.get("autumn") or u.get("p_old", 0) >= 0.4))
    care_list = leaf_care.care(key, lang)
    # ONE clear answer for the farmer: what it is + one thing to do
    if status == "sick":
        d0 = spec["top"][0] if spec and spec["sure"] and spec["p_sick"] >= 0.5 and not spec["top"][0]["label"].endswith("_healthy") else None
        verdict = dict(kind="sick", title=T["v_sick"], name=d0["name"] if d0 else (symptom or {}).get("name"),
                       action=d0["todo"] if d0 else (symptom["steps"][0] if symptom else T["call"]))
    elif status == "healthy":
        verdict = dict(kind="healthy", title=T["v_healthy"], name=None, action=care_list[0])
    elif old_leaf:
        verdict = dict(kind="natural", title=T["v_natural"], name=None, action=T["old"])
    else:
        verdict = dict(kind="photo", title=T["v_photo"], name=None, action=T["photo_short"])
    note = (T["inside"] if supported else T["outside"]).format(plant=pname.split(" (")[0].capitalize(),
                                                               list=f" ({', '.join(spec_crops)})" if spec_crops else "")
    return dict(kind="plant", plant=plant, alternatives=(u or {}).get("alternatives", []), plant_sure=bool(
        plant and (plant["prob"] is None or plant["prob"] >= 0.3)), supported=supported, note=note,
        health=dict(status=status, label=T[{"sick": "sick", "healthy": "healthy", "unsure": "maybe"}[status]],
                    p_sick=round(p_sick, 3), source=src),
        symptom=symptom, other_symptoms=others, disease=spec,
        old_leaf=old_leaf, old_text=T["old"], verdict=verdict,
        care=care_list, call=T["call"], unsure_text=T["unsure"],
        engine=dict(universal=bool(u), probe=bool(u and u["probe"]), plantnet=bool(plant and plant.get("source") == "plantnet")))


@app.post("/api/diagnose")
def diagnose():
    if not disease_model.ready() and not universal.ready():
        return jsonify(error="photo models not ready yet"), 503
    f = request.files.get("photo")
    if not f:
        return jsonify(error="send a photo (form field 'photo')"), 400
    data = f.read()
    if len(data) > 12 * 1024 * 1024:
        return jsonify(error="photo too large (max 12 MB)"), 413
    lang = request.form.get("lang", "en")
    crop = request.form.get("crop") or None
    if crop and crop not in leaf_plants.PLANTS:
        crop = None
    try:
        universal.load_image(data)
    except Exception as e:
        return jsonify(error=f"could not read this image ({type(e).__name__})"), 400
    try:
        return jsonify(_leaf(data, lang, crop))
    except Exception as e:
        return jsonify(error=f"diagnosis failed ({type(e).__name__}: {e})"), 500


_started = False


def start_background():
    """Start the live updater once (works under gunicorn and under `python api/app.py`)."""
    global _started
    if not _started and os.environ.get("AUTO_UPDATE", "1") == "1":
        _started = True
        threading.Thread(target=refresher, daemon=True).start()
        threading.Thread(target=alerter, daemon=True).start()


start_background()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
