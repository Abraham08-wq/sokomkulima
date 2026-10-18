"""
SOIL at a point, anywhere in Africa.

1st choice: iSDAsoil (iSDA Africa, CC-BY 4.0): machine-learning soil maps of Africa at 30 m, 0-20 cm depth,
            read straight from the public cloud files (one small HTTP range-read per layer, no download).
            Gives pH, organic carbon, extractable P and K, total N, clay, sand.
2nd choice: SoilGrids (ISRIC, 250 m, worldwide REST API): pH, organic carbon, N, clay, sand. No P or K, so
            typical values are used for those and flagged.
3rd choice: typical African upland soil, clearly flagged.
A soil test always wins: values the user types in replace the map values.

Results are cached in data/agro_cache/ so each place is fetched once.
"""
from pathlib import Path
import json, os, urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "agro_cache"
ISDA = os.environ.get("ISDA_URL", "https://isdasoil.s3.amazonaws.com").rstrip("/")
SOILGRIDS = os.environ.get("SOILGRIDS_URL", "https://rest.isric.org").rstrip("/")
UA = {"User-Agent": "SokoMkulima/1.0 (season planner)"}

# iSDAsoil layer -> (our name, back-transformation, unit). Band 1 = predicted mean at 0-20 cm.
ISDA_LAYERS = {
    "ph":                      ("ph", "x/10", "pH (water)"),
    "carbon_organic":          ("soc", "expm1(x/10)", "g/kg"),
    "phosphorous_extractable": ("p_mehlich3", "expm1(x/10)", "mg/kg"),
    "potassium_extractable":   ("k_mehlich3", "expm1(x/10)", "mg/kg"),
    "nitrogen_total":          ("n_total", "expm1(x/100)", "g/kg"),
    "clay_content":            ("clay", "x", "%"),
    "sand_content":            ("sand", "x", "%"),
}
BACK = {"x": lambda x: x, "x/10": lambda x: x / 10, "x/100": lambda x: x / 100,
        "expm1(x/10)": lambda x: float(np.expm1(x / 10)), "expm1(x/100)": lambda x: float(np.expm1(x / 100))}
TYPICAL = {"ph": 5.9, "soc": 10.0, "p_mehlich3": 10.0, "k_mehlich3": 150.0, "n_total": 0.9, "clay": 25.0, "sand": 50.0}
VALID = {"ph": (3.5, 9.5), "soc": (0.5, 150), "p_mehlich3": (0.5, 300), "k_mehlich3": (10, 2000), "n_total": (0.05, 15),
         "clay": (1, 90), "sand": (1, 98)}
OLSEN_PER_MEHLICH3 = 0.5      # Olsen P is roughly half of Mehlich-3 P on African soils (approximation)


def _isda_point(lat, lon):
    import rasterio
    from rasterio.warp import transform
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    if os.path.exists("/etc/ssl/certs/ca-certificates.crt"):          # let GDAL's curl verify HTTPS in Docker
        os.environ.setdefault("CURL_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")
    out = {}
    for layer, (name, back, _) in ISDA_LAYERS.items():
        url = f"{ISDA}/soil_data/{layer}/{layer}.tif"
        with rasterio.open(url) as src:
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            v = float(next(src.sample([(xs[0], ys[0])], indexes=1))[0])
            if src.nodata is not None and v == src.nodata:
                continue
            val = BACK[back](v)
            lo, hi = VALID[name]
            if lo <= val <= hi:
                out[name] = round(val, 3)
    return out


def _soilgrids_point(lat, lon):
    props = ["phh2o", "soc", "nitrogen", "clay", "sand"]
    q = "&".join([f"lon={lon}", f"lat={lat}"] + [f"property={p}" for p in props] +
                 [f"depth={d}" for d in ("0-5cm", "5-15cm", "15-30cm")] + ["value=mean"])
    req = urllib.request.Request(f"{SOILGRIDS}/soilgrids/v2.0/properties/query?{q}", headers=UA)
    data = json.loads(urllib.request.urlopen(req, timeout=40).read())
    conv = {"phh2o": ("ph", 1), "soc": ("soc", 1), "nitrogen": ("n_total", 1), "clay": ("clay", 1), "sand": ("sand", 1)}
    # mapped value / d_factor = conventional unit: pH, soc g/kg, nitrogen g/kg, clay %, sand %
    target = {"soc": 1.0, "nitrogen": 1.0, "clay": 1.0, "sand": 1.0, "phh2o": 1.0}
    out = {}
    for layer in data["properties"]["layers"]:
        name = layer["name"]
        if name not in conv:
            continue
        d = layer["unit_measure"]["d_factor"] or 1
        w = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 5}                      # weights to approximate 0-20 cm
        vals = [(dp["values"]["mean"], w.get(dp["label"], 0)) for dp in layer["depths"] if dp["values"]["mean"] is not None]
        if vals:
            v = sum(a * b for a, b in vals) / sum(b for _, b in vals) / d * target[name]
            key = conv[name][0]
            if VALID[key][0] <= v <= VALID[key][1]:
                out[key] = round(v, 3)
    return out


def get_soil(lat, lon, overrides=None):
    """-> dict with soil values, the source of each value, and the QUEFTS inputs."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath = CACHE / f"soil_{lat:.3f}_{lon:.3f}.json"
    if cpath.exists():
        cached = json.loads(cpath.read_text())
    else:
        cached, notes = {"values": {}, "source": {}}, []
        for name, fn in (("iSDAsoil (30 m)", _isda_point), ("SoilGrids (250 m)", _soilgrids_point)):
            try:
                got = fn(lat, lon)
            except Exception as e:
                notes.append(f"{name} unavailable: {type(e).__name__}")
                continue
            for k, v in got.items():
                if k not in cached["values"]:
                    cached["values"][k] = v; cached["source"][k] = name
            if all(k in cached["values"] for k in ("ph", "soc", "p_mehlich3", "k_mehlich3")):
                break
        cached["notes"] = notes
        if cached["values"]:                              # only cache real map values
            cpath.write_text(json.dumps(cached))
    vals, src = dict(cached["values"]), dict(cached["source"])
    for k, v in TYPICAL.items():
        if k not in vals:
            vals[k] = v; src[k] = "typical value (no map data)"
    for k, v in (overrides or {}).items():
        if v not in (None, ""):
            vals[k] = float(v); src[k] = "your soil test"
    p_olsen = vals["p_mehlich3"] * OLSEN_PER_MEHLICH3
    k_mmol = vals["k_mehlich3"] / 39.1
    flags = []
    if vals["ph"] < 5.5:
        flags.append("acid")          # fertilizer works poorly; lime advised
    if vals["soc"] < 6:
        flags.append("low_carbon")    # add manure / crop residues
    awc = float(np.clip(50 + 1.2 * vals["clay"] - 0.3 * vals["sand"] + 4 * vals["soc"] ** 0.5, 50, 150))
    return {"values": vals, "source": src, "notes": cached.get("notes", []), "flags": flags,
            "quefts_inputs": {"pH": vals["ph"], "soc_gkg": vals["soc"], "p_olsen_mgkg": round(p_olsen, 2),
                              "k_mmolkg": round(k_mmol, 3)},
            "awc_mm": round(awc)}
