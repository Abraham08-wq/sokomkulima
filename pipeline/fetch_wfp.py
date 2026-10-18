"""
STEP 0 — LIVE DATA: download the latest WFP food prices for every African country from HDX
(the Humanitarian Data Exchange, data.humdata.org, where the World Food Programme publishes its market prices).

    python pipeline/fetch_wfp.py                 # all African countries
    COUNTRIES=KEN,UGA,TZA python pipeline/fetch_wfp.py

How it works:
  1. Asks the HDX API (package_search) for the WFP "Food Prices" dataset of each country (HDX groups = ISO3 code).
  2. Compares the dataset's "last modified" time with the last download (data/raw/live_manifest.json).
  3. Downloads only countries that changed, and saves data/raw/LIVE_wfp_food_prices_<iso3>.csv.
  4. When real data arrives, the DEMO file is deleted so it can never mix with real prices.

Exit code 0 = real data available, 2 = nothing could be downloaded (the system then falls back to DEMO data).
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io, json, os, sys, time, urllib.parse, urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "live_manifest.json"
HDX = os.environ.get("HDX_URL", "https://data.humdata.org").rstrip("/")
UA = {"User-Agent": "SokoMkulima/1.0 (food price forecasting for farmers)"}

AFRICA = """DZA AGO BEN BWA BFA BDI CPV CMR CAF TCD COM COG COD CIV DJI EGY GNQ ERI SWZ ETH GAB GMB GHA GIN GNB
KEN LSO LBR LBY MDG MWI MLI MRT MUS MAR MOZ NAM NER NGA RWA STP SEN SYC SLE SOM ZAF SSD SDN TZA TGO TUN UGA ZMB ZWE""".split()


def get(url, timeout=60, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def find_dataset(iso):
    """Returns the WFP food-price dataset (CKAN package dict) for one country, or None."""
    q = urllib.parse.urlencode({"q": "food prices", "fq": f"organization:wfp groups:{iso.lower()}", "rows": 20})
    res = json.loads(get(f"{HDX}/api/action/package_search?{q}"))["result"]["results"]
    res = [p for p in res if p.get("name", "").startswith("wfp-food-prices")]
    return res[0] if res else None


def csv_urls(pkg):
    out = []
    for r in pkg.get("resources", []):
        name = (r.get("name") or "").lower()
        url = r.get("download_url") or r.get("url") or ""
        if (r.get("format") or "").upper() == "CSV" and "quickchart" not in name and "food_prices" in url.lower():
            out.append(url)
    return out


def fetch_country(iso, known):
    try:
        pkg = find_dataset(iso)
        if not pkg:
            return iso, {"status": "none"}
        modified = pkg.get("metadata_modified") or pkg.get("last_modified") or ""
        target = RAW / f"LIVE_wfp_food_prices_{iso.lower()}.csv"
        if target.exists() and known.get("modified") == modified:
            return iso, {**known, "status": "unchanged"}
        frames = []
        for url in csv_urls(pkg):
            df = pd.read_csv(io.BytesIO(get(url, timeout=180)), low_memory=False)
            frames.append(df[~df.iloc[:, 0].astype(str).str.startswith("#")])      # drop the HXL tag row
        if not frames:
            return iso, {"status": "no csv"}
        df = pd.concat(frames, ignore_index=True)
        df["countryiso3"] = iso
        df.to_csv(target, index=False)
        last = str(pd.to_datetime(df["date"], errors="coerce").max())[:10]
        return iso, {"status": "updated", "dataset": pkg["name"], "modified": modified, "rows": len(df),
                     "latest_price": last}
    except Exception as e:
        return iso, {**known, "status": f"error: {type(e).__name__}: {e}"[:200]}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    countries = [c.strip().upper() for c in os.environ.get("COUNTRIES", "").split(",") if c.strip()] or AFRICA
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    prev = manifest.get("countries", {})
    print(f"Checking WFP food prices on {HDX} for {len(countries)} African countries...")
    with ThreadPoolExecutor(6) as ex:
        results = dict(ex.map(lambda c: fetch_country(c, prev.get(c, {})), countries))
    for iso, r in sorted(results.items()):
        if r["status"] != "none":
            print(f"  {iso}: {r['status']}" + (f"  ({r.get('rows', 0):,} rows, latest {r.get('latest_price')})"
                                               if r["status"] in ("updated", "unchanged") else ""))
    have = {iso: r for iso, r in results.items() if (RAW / f"LIVE_wfp_food_prices_{iso.lower()}.csv").exists()}
    changed = [iso for iso, r in results.items() if r["status"] == "updated"]
    manifest = {"source": f"WFP food prices via HDX ({HDX})",
                "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "countries": {iso: r for iso, r in results.items() if r["status"] != "none"},
                "changed": changed, "signature": "|".join(f"{i}:{have[i].get('modified', '')}" for i in sorted(have))}
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"{len(have)} countries with real data, {len(changed)} updated this check.")
    if have:
        for f in RAW.glob("DEMO_*.csv"):
            f.unlink(); print(f"  removed {f.name} (real data is available)")
        return 0
    print("No real data could be downloaded (no internet, or HDX unreachable).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
