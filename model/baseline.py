"""
STEP 2a — MODELS THAT RUN WITHOUT PYTORCH (always available)

    python model/baseline.py --slug maize_retail

  naive     : next months = this month's price
  seasonal  : this month's price + the market's usual seasonal change for those months
  graph_ridge : a graph-aware regression. For every market it uses its own recent price changes AND the
                recent price changes of its neighbours in the market graph (weighted by the dynamic edge
                strength), plus the calendar month. One model is trained for all of Africa's markets.
The deep-learning graph model (model/train_gnn.py) is compared against these.
"""
import argparse
import numpy as np
from sklearn.linear_model import Ridge
from common import load, filled, month_numbers, split_origins, score, save_result, HORIZONS

LAGS = 6


def seasonal_profile(logp, months, train_end):
    """Average log change from calendar month a to month a+h for each market, using training months only."""
    T, N = logp.shape
    prof = np.zeros((12, N, len(HORIZONS)))
    for hi, h in enumerate(HORIZONS):
        d = logp[h:train_end] - logp[:train_end - h]
        for a in range(1, 13):
            rows = d[months[:train_end - h] == a]
            prof[a - 1, :, hi] = np.nan_to_num(np.nanmean(rows, 0)) if len(rows) else 0
    return prof


def neighbour_weights(edges, dyn, N, t):
    w = np.nan_to_num(np.clip(dyn[t], 0, None)) + 0.05          # dynamic strength at month t
    A = np.zeros((N, N)); A[edges[:, 0], edges[:, 1]] = w; A[edges[:, 1], edges[:, 0]] = w
    s = A.sum(1, keepdims=True); s[s == 0] = 1
    return A / s


def features(F, edges, dyn, months, t):
    N = F.shape[1]
    d = np.diff(F[t - LAGS:t + 1], axis=0)                       # own last LAGS monthly changes  (LAGS, N)
    A = neighbour_weights(edges, dyn, N, t)
    nd = (A @ d.T).T                                             # neighbours' changes
    gap = F[t] - (A @ F[t])                                      # price above/below neighbours (log)
    lvl = F[t] - F[max(0, t - 36):t + 1].mean(0)                 # price above/below own 3-year average
    m = months[t]
    cal = np.repeat([[np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]], N, 0)
    return np.column_stack([d.T, nd.T, gap, lvl, cal])


def main(slug):
    logp, edges, dyn, dist, cross, meta = load(slug)
    T, N = logp.shape
    F = filled(logp); months = month_numbers(meta)
    train, test = split_origins(T, LAGS + 1)
    prof = seasonal_profile(logp, months, test[0])
    results = {}

    # naive + seasonal
    naive = np.zeros((len(test), N, 3))
    seas = np.stack([prof[(months[t] - 1)] for t in test])
    results["naive"] = (score(naive, logp, test, F), naive)
    results["seasonal"] = (score(seas, logp, test, F), seas)

    # graph ridge
    X = np.concatenate([features(F, edges, dyn, months, t) for t in train])
    preds_test, preds_final, sds = [], [], []
    for hi, h in enumerate(HORIZONS):
        Y = np.concatenate([logp[t + h] - F[t] for t in train])
        ok = ~np.isnan(Y)
        mdl = Ridge(alpha=1.0).fit(X[ok], Y[ok])
        sds.append(np.std(Y[ok] - mdl.predict(X[ok])))
        preds_test.append(np.stack([mdl.predict(features(F, edges, dyn, months, t)) for t in test]))
        preds_final.append(mdl.predict(features(F, edges, dyn, months, T - 1)))
    gr = np.stack(preds_test, -1)
    results["graph_ridge"] = (score(gr, logp, test, F), gr)

    print(f"\nTest set: last {len(test)} forecast months ({meta['months'][test[0]]} to {meta['months'][test[-1]]})")
    print(f"{'model':<14}" + "".join(f"  MAPE h={h}" for h in HORIZONS))
    for k, (m, _) in results.items():
        print(f"{k:<14}" + "".join(f"  {m[h]['mape']:8.2f}%" for h in HORIZONS))

    edge_w = np.clip(np.nan_to_num(dyn), 0, 1)
    save_result(slug, "graph_ridge", results["graph_ridge"][0], logp, F, meta,
                np.stack(preds_final, -1), np.array(sds), gr, test, edge_w)
    # save the simple baselines' scores too, for the comparison table
    import json
    from common import OUT
    (OUT / f"{slug}__comparison.json").write_text(json.dumps(
        {k: {str(h): v for h, v in m.items()} for k, (m, _) in results.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--slug", default="maize_retail")
    main(ap.parse_args().slug)
