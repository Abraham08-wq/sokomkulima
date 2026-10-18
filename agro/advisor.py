"""
SEASON PLANNER: closes the yield gap for one field.

    soil (agro/soil.py) ──► soil N, P, K supply (QUEFTS)
    rain (agro/weather.py) ──► best planting window + water-limited yield in each of 35 past seasons
    QUEFTS (agro/quefts.py) ──► yield for every fertilizer bundle (in half-bag steps)
    SokoMkulima price forecast ──► value of the extra harvest at the nearest market
    ──► the bundle with the highest expected profit that is SAFE:
          value-cost ratio >= 2 in the median season, and money lost in at most 1 season in 4.

The four yields shown to the farmer (the yield gap):
    potential      : ideal water and nutrients
    water-limited  : what this place's rain allows in a median season
    with the plan  : what the recommended fertilizer reaches
    without        : what the soil alone gives (today's typical farmer)
"""
from pathlib import Path
import itertools, json, math
import numpy as np

from .crops import CROPS, PRODUCTS, STEP_KG, MAX_KG, BAG_KG, ACRE_HA, FARMGATE_SHARE, MIN_VCR
from .quefts import quefts, soil_supply
from .soil import get_soil
from .weather import seasons

ROOT = Path(__file__).resolve().parent.parent
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
DEFAULT_PRICE = {"maize": 0.35, "sorghum": 0.45, "millet": 0.5, "rice": 0.9, "wheat": 0.5, "teff": 1.2,
                 "beans": 1.0, "cowpea": 0.9, "groundnut": 1.2, "cassava": 0.25}       # USD/kg retail, fallback
LEGUMES = {"beans", "cowpea", "groundnut"}


def haversine(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def market_price(crop, lat, lon):
    """Nearest SokoMkulima market for this crop (<= 400 km): current price and the best model's 3-month forecast."""
    meta_p = next(iter(sorted(PROC.glob(f"{crop}_*.json"))), None)
    if meta_p:
        slug = meta_p.stem
        meta = json.loads(meta_p.read_text())
        methods = [m for m in ("st_gnn", "gru_no_graph", "graph_ridge") if (OUT / f"{slug}__{m}.json").exists()]
        try:                                       # same honest choice as the price page (model/selection.py)
            import sys as _s
            _s.path.insert(0, str(ROOT / "model"))
            from selection import selection
            order = selection(slug)["order"]
            methods.sort(key=lambda m: order.index(m) if m in order else 99)
        except Exception:
            pass
        if methods:
            res = json.loads((OUT / f"{slug}__{methods[0]}.json").read_text())
            best = min(meta["markets"], key=lambda m: haversine(lat, lon, m["lat"], m["lon"]))
            km = haversine(lat, lon, best["lat"], best["lon"])
            if km <= 400:
                fc = res["forecasts"][best["id"]]
                return {"market": best["name"], "country": best["country"], "km": round(km), "method": methods[0],
                        "retail_now": fc["current"], "retail_3m": fc["forecast"][-1], "low_3m": fc["low"][-1],
                        "as_of": meta.get("data_as_of"), "demo": meta.get("demo", False)}
    p = DEFAULT_PRICE.get(crop, 0.4)
    return {"market": None, "retail_now": p, "retail_3m": p, "low_3m": p * 0.8, "as_of": None, "default": True}


def bundles():
    grid = {k: range(0, MAX_KG[k] + 1, STEP_KG) for k in PRODUCTS}
    for combo in itertools.product(*grid.values()):
        b = dict(zip(grid.keys(), combo))
        if b["dap"] and b["npk"]:            # one phosphorus product at a time keeps advice simple
            continue
        yield b


def nutrients(b):
    return {e: sum(kg * PRODUCTS[p][e] / 100 for p, kg in b.items()) for e in ("N", "P", "K")}


def cost(b, prices):
    return sum(kg / BAG_KG * prices.get(p, PRODUCTS[p]["price"]) for p, kg in b.items())


def plan(lat, lon, crop="maize", farm_ha=ACRE_HA, fert_prices=None, soil_overrides=None, crop_price=None,
         season_index=None, today_doy=None):
    c = CROPS[crop]
    fert_prices = {k: float(v) for k, v in (fert_prices or {}).items() if v not in (None, "")}
    soil = get_soil(lat, lon, soil_overrides)
    q = soil["quefts_inputs"]
    sN, sP, sK = soil_supply(q["pH"], q["soc_gkg"], q["k_mmolkg"], q["p_olsen_mgkg"])
    clim = seasons(lat, lon, c, soil["awc_mm"])
    if not clim["seasons"]:
        return {"ok": False, "reason": "dry", "soil": soil, "climate": clim,
                "message": f"Rain here is too low or unreliable for rain-fed {crop} (no planting date gives half "
                           "of the potential yield in a typical year). Consider a drought-tolerant crop or irrigation."}
    # which season: the next one to open from today
    if season_index is None:
        import datetime as _dt
        t = today_doy if today_doy is not None else _dt.date.today().timetuple().tm_yday
        season_index = min(range(len(clim["seasons"])), key=lambda i: (clim["seasons"][i]["window"][0] - t) % 365)
    s = clim["seasons"][season_index]
    yw_years = np.array(s["frac_by_year"]) * c["ypot"]                       # kg DM/ha, one per past season
    yw_med = float(np.median(yw_years))

    mp = market_price(crop, lat, lon)
    farmgate = float(crop_price) if crop_price not in (None, "") else FARMGATE_SHARE * float(np.mean([mp["retail_now"], mp["retail_3m"]]))
    to_sold = 1 / c["dm"]                                                     # kg DM -> kg as sold

    def y(b, yw):
        n = nutrients(b)
        return quefts(c["quefts"], sN, sP, sK, n["N"], n["P"], n["K"], yw, c["days"])[0]

    y0_med, d0 = quefts(c["quefts"], sN, sP, sK, 0, 0, 0, yw_med, c["days"])
    cands = []
    for b in bundles():
        if crop in LEGUMES and b["urea"]:        # legumes fix their own nitrogen
            continue
        cst = cost(b, fert_prices)
        gain = (y(b, yw_med) - y0_med) * to_sold * farmgate
        cands.append((gain - cst, b, cst, gain))
    cands.sort(key=lambda t: -t[0])
    best, rows = None, []
    for _, b, cst, gain in cands[:40]:
        if cst == 0:
            continue
        per_year = np.array([(y(b, yw) - y(dict.fromkeys(b, 0), yw)) * to_sold * farmgate - cst for yw in yw_years])
        vcr = gain / cst
        r = {"bundle": b, "cost_ha": cst, "gain_ha": gain, "profit_ha": gain - cst, "vcr": vcr,
             "expected_profit_ha": float(per_year.mean()), "loss_share": float(np.mean(per_year < 0))}
        rows.append(r)
        if vcr >= MIN_VCR and r["loss_share"] <= 0.25 and (best is None or r["expected_profit_ha"] > best["expected_profit_ha"]):
            best = r
    b = best["bundle"] if best else dict.fromkeys(PRODUCTS, 0)
    n = nutrients(b)
    y_plan_med, dplan = quefts(c["quefts"], sN, sP, sK, n["N"], n["P"], n["K"], yw_med, c["days"])

    def sold(kg_dm_ha):
        return {"kg_ha": round(kg_dm_ha * to_sold), "kg_farm": round(kg_dm_ha * to_sold * farm_ha),
                "kg_acre": round(kg_dm_ha * to_sold * ACRE_HA)}

    products = [{"product": k, "name": PRODUCTS[k]["name"], "kg_ha": v, "kg_farm": round(v * farm_ha, 1),
                 "bags_farm": round(v * farm_ha / BAG_KG, 2), "kg_acre": round(v * ACRE_HA, 1),
                 "when": ("at planting, in the planting hole, covered with soil" if k in ("dap", "npk", "mop")
                          else "top-dress 3-5 weeks after the crop comes up, when the soil is moist")}
                for k, v in b.items() if v]
    advice = []
    if "acid" in soil["flags"]:
        advice.append("acid")
    if "low_carbon" in soil["flags"]:
        advice.append("manure")
    if crop in LEGUMES:
        advice.append("legume")
    return {
        "ok": True, "crop": crop, "lat": lat, "lon": lon, "farm_ha": farm_ha,
        "season": s, "season_index": season_index, "all_seasons": clim["seasons"],
        "climate": {k: clim[k] for k in ("curve", "monthly_rain_mm", "annual_rain_mm", "years", "cell")},
        "soil": soil, "soil_supply_kg_ha": {"N": round(float(sN)), "P": round(float(sP), 1), "K": round(float(sK))},
        "limiting_without": d0.get("limiting"), "limiting_with": dplan.get("limiting"),
        "yield_gap": {"potential": sold(c["ypot"]), "water_limited": sold(yw_med), "with_plan": sold(y_plan_med),
                      "without": sold(y0_med)},
        "plan": {"products": products, "nutrients_kg_ha": {k: round(v, 1) for k, v in n.items()},
                 "cost_farm": round(best["cost_ha"] * farm_ha, 2) if best else 0.0,
                 "extra_value_farm": round(best["gain_ha"] * farm_ha, 2) if best else 0.0,
                 "profit_farm": round(best["profit_ha"] * farm_ha, 2) if best else 0.0,
                 "vcr": round(best["vcr"], 2) if best else None,
                 "loss_share": best["loss_share"] if best else None,
                 "recommended": best is not None},
        "price": {**mp, "farmgate_used": round(farmgate, 3)},
        "alternatives": [{"bundle": {k: v for k, v in r["bundle"].items() if v}, "cost_ha": round(r["cost_ha"], 1),
                          "profit_ha": round(r["profit_ha"], 1), "vcr": round(r["vcr"], 2),
                          "loss_share": r["loss_share"]} for r in sorted(rows, key=lambda r: -r["expected_profit_ha"])[:6]],
        "advice_flags": advice, "crop_note": c.get("note"),
    }
