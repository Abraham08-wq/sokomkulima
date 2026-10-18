"""
INSURANCE STEP 1 — the ground truth: how good was each district's harvest, each year?

Source: HarvestStat Africa v1.2 (Lee et al. 2025, Scientific Data 12:690, MIT licence): harmonised official
crop statistics for districts (admin-1/2 units) in 33 Sub-Saharan countries.

    python insure/prepare_labels.py            # uses the copy bundled in insure/data/
    python insure/prepare_labels.py --download # fetches the latest release from GitHub first

Output (insure/data/):
  labels.csv.gz : one row per district x crop x season x year (rain-fed, quality flag 0, 1991+)
  units.json    : each district's centroid and a simplified outline (for the map)
No GIS library is needed: the GeoPackage is SQLite, and the shapes are read straight from their WKB bytes.
"""
from pathlib import Path
import argparse, io, json, sqlite3, struct, sys, urllib.request
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = HERE.parent / "data" / "harveststat"
GH = "https://raw.githubusercontent.com/HarvestStat/HarvestStat-Africa/main/public"
CROPS = {"Maize": "maize", "Sorghum": "sorghum", "Millet": "millet"}
SYSTEM_PRIORITY = ["Rainfed (PS)", "traditional_rainfed", "All (PS)", "none", "mechanized_rainfed"]
MIN_YEARS = 8


# ---------- GeoPackage geometry (GP header + WKB) -> rings ----------
def _wkb_rings(b, o=0):
    """Returns (list of polygons, each a list of rings [[x,y],...]), new offset."""
    le = b[o] == 1; o += 1
    E = "<" if le else ">"
    t = struct.unpack_from(E + "I", b, o)[0] % 1000; o += 4
    if t == 3:                                           # Polygon
        n = struct.unpack_from(E + "I", b, o)[0]; o += 4
        rings = []
        for _ in range(n):
            k = struct.unpack_from(E + "I", b, o)[0]; o += 4
            pts = np.frombuffer(b, dtype=E + "f8", count=2 * k, offset=o).reshape(-1, 2); o += 16 * k
            rings.append(pts)
        return [rings], o
    if t == 6:                                           # MultiPolygon
        n = struct.unpack_from(E + "I", b, o)[0]; o += 4
        polys = []
        for _ in range(n):
            p, o = _wkb_rings(b, o)
            polys += p
        return polys, o
    raise ValueError(f"geometry type {t} not supported")


def gpkg_geom(blob):
    flags = blob[3]
    env = (flags >> 1) & 7
    hdr = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
    return _wkb_rings(blob, hdr)[0]


def centroid_and_outline(polys, max_pts=60):
    A = cx = cy = 0.0
    for poly in polys:
        x, y = poly[0][:, 0], poly[0][:, 1]                 # outer ring
        cr = x[:-1] * y[1:] - x[1:] * y[:-1]
        a = cr.sum() / 2
        if abs(a) < 1e-12:
            continue
        A += a; cx += ((x[:-1] + x[1:]) * cr).sum() / 6; cy += ((y[:-1] + y[1:]) * cr).sum() / 6
    if abs(A) < 1e-12:
        pts = np.vstack([p[0] for p in polys]); lon, lat = pts[:, 0].mean(), pts[:, 1].mean()
    else:
        lon, lat = cx / A, cy / A
    outline = []
    for poly in sorted(polys, key=lambda p: -len(p[0]))[:6]:                   # biggest parts only
        r = poly[0]
        step = max(1, len(r) // max_pts)
        outline.append(np.round(r[::step], 3).tolist())
    return float(lat), float(lon), outline


def fetch(name):
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / name
    if not p.exists():
        print(f"  downloading {name} from HarvestStat GitHub...")
        p.write_bytes(urllib.request.urlopen(f"{GH}/{name}", timeout=300).read())
    return p


def main(src_csv, src_gpkg):
    d = pd.read_csv(src_csv, low_memory=False)
    d = d[(d.qc_flag == 0) & (d.harvest_year >= 1991) & d["product"].isin(CROPS) & d["yield"].notna() & (d["yield"] > 0)]
    d = d[d.crop_production_system.isin(SYSTEM_PRIORITY)].copy()
    d["prio"] = d.crop_production_system.map({s: i for i, s in enumerate(SYSTEM_PRIORITY)})
    d = d.sort_values("prio").drop_duplicates(["fnid", "product", "season_name", "harvest_year"])
    d["crop"] = d["product"].map(CROPS)
    n = d.groupby(["fnid", "crop", "season_name"]).harvest_year.transform("count")
    d = d[n >= MIN_YEARS]
    keep = ["fnid", "country", "country_code", "admin_1", "admin_2", "crop", "season_name", "planting_year",
            "planting_month", "harvest_year", "harvest_month", "area", "production", "yield"]
    DATA.mkdir(parents=True, exist_ok=True)
    d[keep].sort_values(["fnid", "crop", "season_name", "harvest_year"]).to_csv(DATA / "labels.csv.gz", index=False)
    print(f"labels: {len(d):,} district-seasons, {d.fnid.nunique()} districts, {d.country.nunique()} countries")

    con = sqlite3.connect(src_gpkg)
    t = con.execute("select table_name from gpkg_contents where data_type='features'").fetchone()[0]
    need = set(d.fnid)
    units = {}
    for fnid, country, a1, a2, geom in con.execute(f"select fnid, country, admin_1, admin_2, geom from '{t}'"):
        if fnid in need and geom:
            lat, lon, outline = centroid_and_outline(gpkg_geom(geom))
            units[fnid] = {"name": a2 if a2 and a2 != "none" else a1, "admin_1": a1, "country": country,
                           "lat": round(lat, 4), "lon": round(lon, 4), "outline": outline}
    (DATA / "units.json").write_text(json.dumps(units, separators=(",", ":")))
    missing = need - set(units)
    print(f"units: {len(units)} district outlines ({len(missing)} without a shape are skipped)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--csv"); ap.add_argument("--gpkg")
    a = ap.parse_args()
    if a.download:
        main(fetch("hvstat_africa_data_v1.2.csv"), fetch("hvstat_africa_boundary_v1.2.gpkg"))
    elif a.csv and a.gpkg:
        main(a.csv, a.gpkg)
    elif (DATA / "labels.csv.gz").exists():
        print("Using the bundled HarvestStat labels (insure/data/). Run with --download for the latest release.")
    else:
        sys.exit("No labels: run with --download")
