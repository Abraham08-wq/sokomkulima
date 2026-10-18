"""
STEP 1 — DATA + GRAPH
Turns WFP food-price CSVs (data.humdata.org) into a market graph ready for the models.

    python pipeline/prepare.py --commodity maize
    python pipeline/prepare.py --commodity sorghum --pricetype Wholesale

Reads every CSV in data/raw/, keeps African countries, converts all prices to USD per kg,
builds a monthly price table (markets x months) and a graph of markets:
  - nodes  = markets (with latitude/longitude)
  - edges  = each market's nearest markets (k-nearest by real distance) + all markets within RADIUS_KM
  - dynamic edge weights = rolling 12-month correlation of price changes between linked markets,
    so the strength of every connection changes month by month.
"""
from pathlib import Path
import argparse, json, os, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"

AFRICA = set("""DZA AGO BEN BWA BFA BDI CPV CMR CAF TCD COM COG COD CIV DJI EGY GNQ ERI SWZ ETH GAB GMB GHA GIN GNB
KEN LSO LBR LBY MDG MWI MLI MRT MUS MAR MOZ NAM NER NGA RWA STP SEN SYC SLE SOM ZAF SSD SDN TZA TGO TUN UGA ZMB ZWE""".split())

K_NEAREST, RADIUS_KM = 5, 250
MIN_MONTHS = 36          # a market needs at least 3 years of prices to be kept
MAX_GAP = 3              # gaps up to 3 months are filled by interpolation
ROLL = 12                # window (months) for the dynamic edge weights
STALE_MONTHS = int(os.environ.get("STALE_MONTHS", 6))   # drop markets with no price in the last 6 months (advice must be current)
PROCESSED = re.compile(r"flour|meal|bran|bread|grits|semolina|seed|oil|paste|porridge")  # processed products are a different market


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def kg_factor(unit: str):
    """'KG' -> 1, '50 KG' -> 50, '90 KG' -> 90; anything not in kg -> None (dropped)."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)?\s*KG\s*", str(unit).upper())
    if not m:
        return None
    return float(m.group(1)) if m.group(1) else 1.0


def load_raw() -> pd.DataFrame:
    frames = []
    for f in sorted(RAW.glob("*.csv")):
        df = pd.read_csv(f, low_memory=False)
        df = df[~df.iloc[:, 0].astype(str).str.startswith("#")]          # drop the HXL tag row
        if "countryiso3" not in df.columns:                              # per-country files: ISO3 from file name
            iso = re.search(r"_([a-z]{3})\.csv$", f.name.lower())
            df["countryiso3"] = iso.group(1).upper() if iso else "UNK"
        df["source_file"] = f.name
        frames.append(df)
        print(f"  read {f.name}: {len(df):,} rows")
    if not frames:
        raise SystemExit("No CSV files in data/raw/. Download WFP food prices from data.humdata.org "
                         "or run  python pipeline/demo_data.py")
    return pd.concat(frames, ignore_index=True)


def main(commodity="maize", pricetype="Retail", start="2015-01"):
    print("Loading WFP files...")
    df = load_raw()
    df = df[df["countryiso3"].isin(AFRICA)]
    com = df["commodity"].astype(str).str.lower()
    df = df[com.str.contains(commodity.lower(), na=False) & ~com.str.contains(PROCESSED)]
    df = df[df["pricetype"].astype(str).str.lower() == pricetype.lower()]
    df = df[~df["market"].astype(str).str.lower().str.contains("average")]   # "National Average" is not a market
    df["kg"] = df["unit"].map(kg_factor)
    for c in ["usdprice", "latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["kg", "usdprice", "latitude", "longitude"])
    df = df[(df["usdprice"] > 0) & ~((df["latitude"] == 0) & (df["longitude"] == 0))]
    df["usd_per_kg"] = df["usdprice"] / df["kg"]
    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M")
    df = df.dropna(subset=["month"])
    df = df[df["month"] >= pd.Period(start, "M")]
    df["market_id"] = df["countryiso3"] + "|" + df["market"].astype(str)
    # one product per market: keep the commodity variant with the most prices (e.g. "Maize (white)")
    top = df.groupby(["market_id", "commodity"]).size().reset_index(name="n").sort_values("n").drop_duplicates("market_id", keep="last")
    df = df.merge(top[["market_id", "commodity"]], on=["market_id", "commodity"])
    if df.empty:
        raise SystemExit(f"No {commodity} {pricetype} prices in kg for African countries in data/raw/.")
    print(f"  {len(df):,} {commodity} {pricetype} price rows in Africa ({df['countryiso3'].nunique()} countries)")

    # monthly median price per market
    monthly = df.groupby(["market_id", "month"])["usd_per_kg"].median().unstack("market_id")
    months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(months)
    # end the series at the last month most markets have reported (WFP data arrives with a lag)
    cov = monthly.notna().sum(axis=1)
    end = cov.index[cov >= 0.5 * cov.iloc[-36:].max()][-1]
    monthly = monthly.loc[:end]; months = monthly.index
    recent = monthly.iloc[-STALE_MONTHS:].notna().any()
    keep = (monthly.notna().sum() >= MIN_MONTHS) & recent
    monthly = monthly.loc[:, keep]
    logp = np.log(monthly).interpolate(limit=MAX_GAP, limit_area="inside")
    print(f"  {monthly.shape[1]} markets with {MIN_MONTHS}+ months, {len(months)} months "
          f"({months[0]} to {months[-1]})")

    if monthly.shape[1] < 8:
        raise SystemExit(f"Only {monthly.shape[1]} markets have enough recent {commodity} prices; try another crop.")
    info = (df.groupby("market_id").agg(market=("market", "first"), countryiso3=("countryiso3", "first"),
                                         latitude=("latitude", "median"), longitude=("longitude", "median"))
              .loc[logp.columns])
    lat, lon = info["latitude"].to_numpy(float), info["longitude"].to_numpy(float)
    N = len(info)

    # ----- graph: k nearest + radius -----
    D = haversine(lat[:, None], lon[:, None], lat[None], lon[None])
    edges = set()
    for i in range(N):
        order = np.argsort(D[i])
        for j in order[1:K_NEAREST + 1]:
            edges.add((min(i, j), max(i, j)))
        for j in np.where((D[i] <= RADIUS_KM) & (np.arange(N) != i))[0]:
            edges.add((min(i, j), max(i, j)))
    edges = np.array(sorted(edges))
    iso = info["countryiso3"].to_numpy()
    edge_dist = D[edges[:, 0], edges[:, 1]]
    cross = (iso[edges[:, 0]] != iso[edges[:, 1]]).astype(int)
    print(f"  graph: {N} nodes, {len(edges)} edges ({cross.sum()} cross-border)")

    # ----- dynamic edge weights: rolling correlation of monthly price changes -----
    dlog = logp.diff().to_numpy()
    T = len(months)
    dyn = np.full((T, len(edges)), np.nan, dtype=np.float32)
    for t in range(ROLL, T):
        win = dlog[t - ROLL + 1:t + 1]
        a, b = win[:, edges[:, 0]], win[:, edges[:, 1]]
        ok = ~np.isnan(a) & ~np.isnan(b)
        n = ok.sum(0)
        a0 = np.where(ok, a - np.nanmean(np.where(ok, a, np.nan), 0), 0)
        b0 = np.where(ok, b - np.nanmean(np.where(ok, b, np.nan), 0), 0)
        num = (a0 * b0).sum(0); den = np.sqrt((a0 ** 2).sum(0) * (b0 ** 2).sum(0))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where((n >= 8) & (den > 0), num / den, np.nan)
        dyn[t] = r

    OUT.mkdir(parents=True, exist_ok=True)
    slug = f"{commodity.lower()}_{pricetype.lower()}"
    np.savez_compressed(OUT / f"{slug}.npz", logp=logp.to_numpy(np.float32), edges=edges,
                        edge_dist=edge_dist.astype(np.float32), cross=cross, dyn=dyn)
    meta = {"commodity": commodity, "pricetype": pricetype, "unit": "USD per kg",
            "months": [str(m) for m in months],
            "demo": bool(df["source_file"].str.startswith("DEMO").all()),
            "source": "DEMO (synthetic)" if df["source_file"].str.startswith("DEMO").all() else "WFP food prices via HDX",
            "data_as_of": str(months[-1]), "countries": sorted(info["countryiso3"].unique().tolist()),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "markets": [{"id": k, "name": r.market, "country": r.countryiso3, "lat": float(r.latitude),
                         "lon": float(r.longitude), "n_obs": int(monthly[k].notna().sum())}
                        for k, r in info.iterrows()]}
    (OUT / f"{slug}.json").write_text(json.dumps(meta))
    print(f"Saved data/processed/{slug}.npz and .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--commodity", default="maize")
    ap.add_argument("--pricetype", default="Retail")
    ap.add_argument("--start", default="2015-01")
    a = ap.parse_args()
    main(a.commodity, a.pricetype, a.start)
