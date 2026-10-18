"""
DEMO DATA ONLY: generates a synthetic file in the same format as the WFP food-price CSVs
(data.humdata.org), so the system can be tested before the real data is downloaded.
Prices are invented. Replace with the real WFP files for any real use.

    python pipeline/demo_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "DEMO_wfp_food_prices_africa.csv"

# (country ISO3, market, lat, lon, region) — approximate city locations
MARKETS = [
    ("KEN", "Nairobi", -1.29, 36.82, "E"), ("KEN", "Eldoret", 0.52, 35.27, "E"), ("KEN", "Kitale", 1.02, 35.0, "E"),
    ("KEN", "Mombasa", -4.04, 39.67, "E"), ("KEN", "Kisumu", -0.09, 34.77, "E"), ("KEN", "Garissa", -0.45, 39.65, "E"),
    ("UGA", "Kampala", 0.35, 32.58, "E"), ("UGA", "Mbale", 1.08, 34.18, "E"), ("UGA", "Gulu", 2.78, 32.3, "E"),
    ("TZA", "Dar es Salaam", -6.79, 39.21, "E"), ("TZA", "Arusha", -3.37, 36.68, "E"), ("TZA", "Dodoma", -6.16, 35.75, "E"),
    ("TZA", "Mbeya", -8.9, 33.46, "S"), ("RWA", "Kigali", -1.95, 30.06, "E"), ("BDI", "Bujumbura", -3.38, 29.36, "E"),
    ("ETH", "Addis Ababa", 9.03, 38.74, "H"), ("ETH", "Dire Dawa", 9.6, 41.85, "H"), ("ETH", "Mekele", 13.5, 39.47, "H"),
    ("SOM", "Mogadishu", 2.05, 45.32, "H"), ("SOM", "Baidoa", 3.12, 43.65, "H"), ("SSD", "Juba", 4.85, 31.58, "E"),
    ("SDN", "Khartoum", 15.5, 32.56, "H"),
    ("ZMB", "Lusaka", -15.39, 28.32, "S"), ("ZMB", "Chipata", -13.64, 32.65, "S"), ("MWI", "Lilongwe", -13.96, 33.79, "S"),
    ("MWI", "Blantyre", -15.79, 35.0, "S"), ("MOZ", "Maputo", -25.97, 32.57, "S"), ("MOZ", "Nampula", -15.12, 39.27, "S"),
    ("ZWE", "Harare", -17.83, 31.05, "S"), ("ZWE", "Bulawayo", -20.15, 28.58, "S"), ("COD", "Lubumbashi", -11.66, 27.48, "S"),
    ("COD", "Goma", -1.68, 29.22, "E"), ("MDG", "Antananarivo", -18.88, 47.51, "S"),
    ("NGA", "Kano", 12.0, 8.52, "W"), ("NGA", "Kaduna", 10.52, 7.44, "W"), ("NGA", "Lagos", 6.52, 3.38, "W"),
    ("NGA", "Maiduguri", 11.85, 13.16, "W"), ("GHA", "Accra", 5.6, -0.19, "W"), ("GHA", "Tamale", 9.4, -0.85, "W"),
    ("GHA", "Kumasi", 6.69, -1.62, "W"), ("BFA", "Ouagadougou", 12.37, -1.52, "W"), ("BFA", "Bobo-Dioulasso", 11.18, -4.3, "W"),
    ("MLI", "Bamako", 12.64, -8.0, "W"), ("MLI", "Mopti", 14.49, -4.18, "W"), ("NER", "Niamey", 13.51, 2.11, "W"),
    ("NER", "Maradi", 13.5, 7.1, "W"), ("TCD", "N'Djamena", 12.13, 15.05, "W"), ("SEN", "Dakar", 14.72, -17.47, "W"),
    ("BEN", "Cotonou", 6.37, 2.39, "W"), ("TGO", "Lome", 6.13, 1.22, "W"), ("CMR", "Garoua", 9.3, 13.4, "W"),
    ("CIV", "Abidjan", 5.36, -4.01, "W"),
]
REGION_PHASE = {"E": 7, "H": 10, "S": 4, "W": 9}   # month index where prices usually peak (lean season)


def main(start="2015-01", end="2025-06", seed=7):
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    T, N = len(months), len(MARKETS)
    lat = np.array([m[2] for m in MARKETS]); lon = np.array([m[3] for m in MARKETS])
    # distance-based coupling (shocks spread to nearby markets with a lag)
    d = np.hypot(lat[:, None] - lat[None], (lon[:, None] - lon[None]) * np.cos(np.radians(lat[:, None])))
    W = np.exp(-d / 4.0); np.fill_diagonal(W, 0); W /= W.sum(1, keepdims=True)
    base = rng.uniform(0.22, 0.45, N)
    level = np.log(base); shock = np.zeros(N); logp = np.zeros((T, N))
    for t in range(T):
        m = months[t].month
        season = np.array([0.18 * np.cos(2 * np.pi * (m - REGION_PHASE[r[4]]) / 12) for r in MARKETS])
        if rng.random() < 0.06:                          # a regional shock (drought, border closure...)
            c = rng.integers(N); shock += 0.35 * np.exp(-d[c] / 3.0)
        shock = 0.82 * shock + 0.12 * (W @ shock) + rng.normal(0, 0.025, N)   # spreads to neighbours
        level += 0.004 + rng.normal(0, 0.01, N) * 0.3
        logp[t] = level + season + shock + rng.normal(0, 0.03, N)
    rows = []
    for t, per in enumerate(months):
        for i, (iso, mk, la, lo, _) in enumerate(MARKETS):
            if rng.random() < 0.06:          # some months missing, like real data
                continue
            usd = float(np.exp(logp[t, i]))
            rows.append({"date": f"{per.year}-{per.month:02d}-15", "countryiso3": iso, "admin1": "", "admin2": "",
                         "market": mk, "latitude": la, "longitude": lo, "category": "cereals and tubers",
                         "commodity": "Maize (white)", "unit": "KG", "priceflag": "actual", "pricetype": "Retail",
                         "currency": "USD", "price": round(usd, 4), "usdprice": round(usd, 4)})
    df = pd.DataFrame(rows)
    hxl = {c: "#" + c for c in df.columns}
    pd.concat([pd.DataFrame([hxl]), df]).to_csv(OUT, index=False)
    print(f"DEMO data written: {OUT}  ({len(df):,} rows, {N} markets, {T} months)")


if __name__ == "__main__":
    main()
