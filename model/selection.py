"""
HONEST MODEL CHOICE for the price forecasts.

The 24 test months are split in two:
  - SELECTION half (first 12 months): each model's 1-month error decides which model a crop uses.
  - REPORT half (last 12 months): the chosen model's error is reported here, on months that played
    no part in choosing it.
So the error shown for the chosen model is not flattered by having been picked on the same data.
Uses the saved 1-month backtests, so nothing needs retraining.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
METHODS = ["st_gnn", "gru_no_graph", "graph_ridge"]
_cache = {}


def _mape(pairs):
    p = np.array([a for a, _ in pairs]); y = np.array([b for _, b in pairs])
    return float(np.mean(np.abs(p - y) / y) * 100) if len(p) else float("nan")


def selection(slug):
    files = [OUT / f"{slug}__{m}.json" for m in METHODS] + [PROC / f"{slug}.npz"]
    key = (slug, tuple(f.stat().st_mtime if f.exists() else 0 for f in files))
    if _cache.get(slug, (None,))[0] == key:
        return _cache[slug][1]
    meta = json.loads((PROC / f"{slug}.json").read_text())
    logp = np.load(PROC / f"{slug}.npz")["logp"]
    midx = {m: i for i, m in enumerate(meta["months"])}
    out = {}
    for m in METHODS:
        p = OUT / f"{slug}__{m}.json"
        if not p.exists():
            continue
        bt = json.loads(p.read_text())["backtest_h1"]
        months = sorted({mo for rows in bt.values() for mo, _ in rows})
        half = set(months[:len(months) // 2])
        sel, rep = [], []
        for n, mk in enumerate(meta["markets"]):
            for mo, pred in bt.get(mk["id"], []):
                v = logp[midx[mo], n]
                if np.isnan(v):
                    continue
                (sel if mo in half else rep).append((pred, float(np.exp(v))))
        out[m] = {"selection_mape_h1": round(_mape(sel), 3), "report_mape_h1": round(_mape(rep), 3),
                  "selection_months": [months[0], months[len(months) // 2 - 1]] if months else None,
                  "report_months": [months[len(months) // 2], months[-1]] if months else None}
    order = sorted(out, key=lambda m: (out[m]["selection_mape_h1"], METHODS.index(m)))
    res = {"order": order, "chosen": order[0] if order else None, "methods": out}
    _cache[slug] = (key, res)
    return res
