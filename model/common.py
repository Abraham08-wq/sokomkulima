"""Shared helpers: load the processed graph data, build features, score forecasts, save results."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"
HORIZONS = (1, 2, 3)
TEST_MONTHS = 24          # the last 24 forecast origins are held out for testing


def load(slug):
    z = np.load(PROC / f"{slug}.npz")
    meta = json.loads((PROC / f"{slug}.json").read_text())
    return z["logp"], z["edges"], z["dyn"], z["edge_dist"], z["cross"], meta


def filled(logp):
    """Forward-fill then back-fill each market so features never contain NaN (targets keep NaN)."""
    x = logp.copy()
    for j in range(x.shape[1]):
        col = x[:, j]
        idx = np.where(~np.isnan(col), np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        col = col[idx]
        first = np.argmax(~np.isnan(col))
        col[:first] = col[first]
        x[:, j] = col
    return x


def month_numbers(meta):
    return np.array([int(m[-2:]) for m in meta["months"]])


def split_origins(T, lookback=12):
    """Forecast origins t (we know prices up to t, predict t+h). Last TEST_MONTHS origins are the test set."""
    last = T - 1 - max(HORIZONS)            # need t+3 to exist for scoring
    test = np.arange(last - TEST_MONTHS + 1, last + 1)
    train = np.arange(lookback, test[0] - max(HORIZONS) + 1)   # no train target overlaps the test period
    return train, test


def score(pred_dlog, logp, origins, filled_logp):
    """pred_dlog[k, n, h] = predicted log change from origin k. Returns MAPE and MAE (USD/kg) per horizon."""
    out = {}
    for hi, h in enumerate(HORIZONS):
        true = logp[origins + h]
        base = filled_logp[origins]
        ok = ~np.isnan(true)
        p = np.exp(base + pred_dlog[:, :, hi])[ok]; y = np.exp(true)[ok]
        out[h] = {"mape": float(np.mean(np.abs(p - y) / y) * 100), "mae": float(np.mean(np.abs(p - y)))}
    return out


def save_result(slug, method, metrics, logp, filled_logp, meta, final_dlog, resid_sd, backtest_dlog, test_origins,
                edge_weights):
    """Writes outputs/<slug>__<method>.json (forecasts, metrics, backtest) and the per-month edge weights."""
    OUT.mkdir(exist_ok=True)
    T = logp.shape[0]
    cur = filled_logp[-1]
    fc = {}
    for n, m in enumerate(meta["markets"]):
        f = np.exp(cur[n] + final_dlog[n]); lo = np.exp(cur[n] + final_dlog[n] - 1.28 * resid_sd)
        hi = np.exp(cur[n] + final_dlog[n] + 1.28 * resid_sd)
        fc[m["id"]] = {"current": float(np.exp(cur[n])), "forecast": f.round(4).tolist(),
                       "low": lo.round(4).tolist(), "high": hi.round(4).tolist()}
    bt = {m["id"]: [[meta["months"][t + 1], round(float(np.exp(filled_logp[t, n] + backtest_dlog[k, n, 0])), 4)]
                    for k, t in enumerate(test_origins)] for n, m in enumerate(meta["markets"])}
    res = {"method": method, "horizons": list(HORIZONS), "origin": meta["months"][-1],
           "metrics": {str(h): v for h, v in metrics.items()}, "resid_sd": resid_sd.round(4).tolist(),
           "forecasts": fc, "backtest_h1": bt}
    (OUT / f"{slug}__{method}.json").write_text(json.dumps(res))
    np.save(OUT / f"{slug}__{method}__edges.npy", edge_weights.astype(np.float32))
    print(f"  saved outputs/{slug}__{method}.json")
