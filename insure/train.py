"""
INSURANCE STEP 3 — learn when a season failed, test it on years it never saw, price a policy for every district.

    python insure/train.py                 # downloads weather for ~650 places once (cached), then trains (minutes)
    INSURE_MAX_CELLS=80 python insure/train.py   # quick trial on fewer places

Label  : loss = 1 - yield / district trend   (trend fitted on TRAINING years only, so no peeking)
         "bad season" = loss >= 30 %  (harvest at least 30 % below normal: what insurance is for)
Split  : train <= 2012 | validation 2013-2015 | TEST >= 2016 (the most recent seasons)
Models : rain_index   - classic index insurance: payout from the season's rainfall deficit
         wrsi_index   - FAO water-satisfaction index, loss = Ky x (1 - WRSI), no training
         ridge        - linear model on season weather summaries
         deep_cnn_gru - deep learning: 1-D convolution + GRU over the 18 ten-day steps of the season, trained to
                        answer "was this a bad season?" (bad seasons weighted ~5x, they are rare) plus the loss size;
                        also sees each district's weather vs its OWN normal (sklearn MLP if PyTorch is missing)
Trigger: each model pays when its predicted loss >= a threshold chosen on train+validation years (best F1, and
         it may not pay more than 1.5x as often as real bad seasons happen, or the premium becomes unaffordable).
Basis risk is measured on the TEST years: bad seasons paid (hit rate), payouts in good seasons (false alarms).
The model with the best VALIDATION F1 prices the policies.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, os, sys, time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pipeline"))
from agro.weather import get_daily                                    # noqa: E402
from insure.features import prepare_cell, from_cell, KY, SEQ, STATIC, N_DEKADS   # noqa: E402

DATA, OUT = ROOT / "insure" / "data", ROOT / "outputs"
TRAIN_END, VAL_END = 2012, 2015
BAD = 0.30                         # a "bad season" = harvest 30 %+ below the district trend (severe loss)
LOADING = 1.3                      # premium = expected payout x 1.3 (admin costs + risk margin)
np.random.seed(42)


def status(msg):
    print(msg, flush=True)
    try:
        import update
        update.write_status(message=msg)
    except Exception:
        pass


def load_weather(units):
    cells = {}
    for fnid, u in units.items():
        cells.setdefault((round(u["lat"] * 2) / 2, round(u["lon"] * 2) / 2), []).append(fnid)
    keys = sorted(cells)
    cap = int(os.environ.get("INSURE_MAX_CELLS", 0) or 0)
    if cap:
        keys = keys[:cap]
    got, done = {}, [0]

    def one(k):
        for attempt in range(3):
            try:
                w = get_daily(*k)
                return k, prepare_cell(w, k[0])
            except Exception as e:
                time.sleep(3 * (attempt + 1)); err = e
        print(f"  weather failed for {k}: {err}")
        return k, None

    with ThreadPoolExecutor(int(os.environ.get("INSURE_THREADS", 4))) as ex:
        for k, cell in ex.map(one, keys):
            done[0] += 1
            if cell is not None:
                got[k] = cell
            if done[0] % 25 == 0 or done[0] == len(keys):
                status(f"Insurance: rainfall history for {done[0]}/{len(keys)} places...")
    return got


def build_samples(labels, units, cells):
    rows, seqs, masks, stats = [], [], [], []
    for r in labels.itertuples():
        u = units.get(r.fnid)
        if u is None:
            continue
        k = (round(u["lat"] * 2) / 2, round(u["lon"] * 2) / 2)
        if k not in cells:
            continue
        f = from_cell(cells[k], r.crop, u["lat"], u["lon"], int(r.planting_year), int(r.planting_month),
                      int(r.harvest_year), int(r.harvest_month))
        if f is None:
            continue
        seqs.append(f[0]); masks.append(f[1]); stats.append(f[2]); rows.append(r.Index)
    df = labels.loc[rows].reset_index(drop=True)
    return df, np.stack(seqs), np.stack(masks), np.stack(stats)


def add_targets(df):
    df["key"] = df.fnid + "|" + df.crop + "|" + df.season_name
    trend = {}
    for key, g in df.groupby("key"):
        tr = g[g.harvest_year <= TRAIN_END]
        if len(tr) < 5:
            continue
        y, t = tr["yield"].to_numpy(), tr.harvest_year.to_numpy()
        if t.max() - t.min() >= 8:
            b, a = np.polyfit(t, y, 1)
            b = float(np.clip(b, -0.03 * y.mean(), 0.05 * y.mean()))           # keep trends plausible
            a = y.mean() - b * t.mean()
        else:
            b, a = 0.0, y.mean()
        trend[key] = (a, b, y.mean())
    df = df[df.key.isin(trend)].copy()
    exp = np.array([max(trend[k][0] + trend[k][1] * yr, 0.3 * trend[k][2]) for k, yr in zip(df.key, df.harvest_year)])
    df["expected"] = exp
    df["loss"] = np.clip(1 - df["yield"] / exp, -1.0, 1.0)
    return df, trend


# ---------------- models ----------------
def fit_rain_index(st, y, tr):
    x = np.maximum(0, 1 - st[:, STATIC.index("rain_ratio")])
    a = float((x[tr] * y[tr]).sum() / max((x[tr] ** 2).sum(), 1e-9))
    return lambda S, *_: a * np.maximum(0, 1 - S[:, STATIC.index("rain_ratio")]), {"a": a}


def wrsi_index(st, crops):
    ky = np.array([KY[c] for c in crops])
    return ky * (1 - st[:, STATIC.index("wrsi")])


def district_extra(st, wrsi_mean, rain_mean):
    """District-relative weather: this season's WRSI and rain compared with the district's own normal.
    (A WRSI of 0.7 is a disaster in a wet district but normal in a dry one.)"""
    return np.hstack([st, np.stack([st[:, STATIC.index("wrsi")] - wrsi_mean,
                                    st[:, STATIC.index("rain_ratio")] - rain_mean], 1)]).astype(np.float32)


def summary_X(seq, st):
    thirds = [seq[:, :6], seq[:, 6:12], seq[:, 12:]]
    parts = [st] + [t[:, :, [1, 3, 4, 5]].mean(1) for t in thirds]
    return np.hstack(parts)


def fit_ridge(seq, st, y, tr):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    X = summary_X(seq, st); sc = StandardScaler().fit(X[tr])
    m = Ridge(alpha=3.0).fit(sc.transform(X[tr]), y[tr])
    return lambda S, Q, M: m.predict(sc.transform(summary_X(Q, S))), (m, sc)


def make_net(fd, fs, h=32):
    """1-D convolution (patterns over neighbouring ten-day steps) -> GRU (the season in order) -> two heads:
    [0] logit of 'this is a bad season'   [1] size of the loss."""
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(fd, h, 3, padding=1), nn.ReLU(), nn.Conv1d(h, h, 3, padding=1), nn.ReLU())
            self.gru = nn.GRU(h, h, batch_first=True)
            self.head = nn.Sequential(nn.Linear(h + fs, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 2))

        def forward(self, q, m, s):
            x = self.conv(q.transpose(1, 2)).transpose(1, 2) * m.unsqueeze(-1)
            lengths = m.sum(1).clamp(min=1).long().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            _, hn = self.gru(packed)
            return self.head(torch.cat([hn[-1], s], 1))
    return Net()


def normalise(model, Q, M, S):
    mu_q, sd_q, mu_s, sd_s = model["norm"]
    return (((Q - mu_q) / sd_q) * M[..., None]).astype(np.float32), ((S - mu_s) / sd_s).astype(np.float32)


def deep_predict(model, S, Q, M):
    """-> (probability of a bad season, predicted loss)"""
    Qn, Sn = normalise(model, Q, M, S)
    if model["kind"] == "torch":
        import torch
        if "net" not in model:
            net = make_net(len(SEQ), S.shape[1]); net.load_state_dict(torch.load(model["state_path"])); net.eval()
            model["net"] = net
        with torch.no_grad():
            o = model["net"](torch.tensor(Qn), torch.tensor(M.astype(np.float32)), torch.tensor(Sn)).numpy()
        return 1 / (1 + np.exp(-o[:, 0])), o[:, 1]
    X = np.hstack([Qn.reshape(len(Qn), -1), Sn])
    return model["clf"].predict_proba(X)[:, 1], model["reg"].predict(X)


def fit_deep(seq, mask, st, y, tr, va):
    """Deep learning model trained to answer 'was this a bad season?' (rare seasons weighted up) and 'how big was
    the loss?'. PyTorch CNN+GRU if available, else sklearn neural networks (MLP)."""
    mu_q = (seq[tr] * mask[tr][..., None]).sum((0, 1)) / mask[tr].sum()
    sd_q = np.sqrt(((seq[tr] - mu_q) ** 2 * mask[tr][..., None]).sum((0, 1)) / mask[tr].sum()) + 1e-6
    mu_s, sd_s = st[tr].mean(0), st[tr].std(0) + 1e-6
    model = {"norm": (mu_q, sd_q, mu_s, sd_s)}
    Qn, Sn = normalise(model, seq, mask, st)
    bad = (y >= BAD).astype(np.float32)
    pos_w = float((1 - bad[tr]).sum() / max(bad[tr].sum(), 1))           # ~5: a bad season counts 5x
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(42)
        q, m, s = torch.tensor(Qn), torch.tensor(mask.astype(np.float32)), torch.tensor(Sn)
        yb, yl = torch.tensor(bad), torch.tensor(y.astype(np.float32))
        net = make_net(len(SEQ), st.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w)); hub = nn.HuberLoss(delta=0.3)
        lossf = lambda o, i: bce(o[:, 0], yb[i]) + hub(o[:, 1], yl[i])
        tri, vai = np.where(tr)[0], np.where(va)[0]
        best, state, wait = 1e9, None, 0
        for ep in range(80):
            net.train()
            for b in np.array_split(np.random.permutation(tri), max(1, len(tri) // 256)):
                opt.zero_grad(); l = lossf(net(q[b], m[b], s[b]), b); l.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = float(lossf(net(q[vai], m[vai], s[vai]), vai))
            if v < best - 1e-5:
                best, state, wait = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
            else:
                wait += 1
            if ep % 5 == 0:
                print(f"  [deep_cnn_gru] epoch {ep:2d}  validation loss {v:.4f}")
            if wait >= 8:
                print(f"  [deep_cnn_gru] early stop at epoch {ep}: keeping the best epoch's weights"); break
        net.load_state_dict(state); net.eval()
        OUT.mkdir(exist_ok=True)
        torch.save(net.state_dict(), OUT / "insure_deep.pt")
        model.update(kind="torch", net=net, state_path=str(OUT / "insure_deep.pt"))
        return model, "deep_cnn_gru"
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    X = np.hstack([Qn.reshape(len(Qn), -1), Sn])
    tri = np.where(tr)[0]
    up = np.r_[tri, np.repeat(tri[bad[tri] == 1], max(1, int(round(pos_w)) - 1))]   # oversample bad seasons
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=300,
                        random_state=42).fit(X[up], bad[up].astype(int))
    reg = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=300,
                       random_state=42).fit(X[tri], y[tri])
    print(f"  [deep_mlp] trained (PyTorch not installed: sklearn neural networks)")
    model.update(kind="mlp", clf=clf, reg=reg)
    return model, "deep_mlp"


def trigger_and_metrics(score, y, fit_mask, loss=None):
    """score: what triggers payment (predicted loss, or probability of a bad season for the deep model)."""
    loss = score if loss is None else loss
    bad = y >= BAD
    best_t, best_f = 0.6, -1
    for t in np.arange(0.05, 0.95, 0.01):
        p = score[fit_mask] >= t; b = bad[fit_mask]
        if p.mean() > 1.5 * b.mean():           # a trigger that pays far more often than real bad seasons is unaffordable
            continue
        tp = (p & b).sum(); f = 2 * tp / max(p.sum() + b.sum(), 1)
        if f > best_f:
            best_t, best_f = float(t), f

    def stats(sel):
        p, b = score[sel] >= best_t, bad[sel]
        tp, fp, fn = int((p & b).sum()), int((p & ~b).sum()), int((~p & b).sum())
        r = float(np.corrcoef(loss[sel], y[sel])[0, 1]) if sel.sum() > 2 else float("nan")
        return {"n": int(sel.sum()), "bad_seasons": int(b.sum()), "paid": int(p.sum()), "hits": tp, "false_alarms": fp,
                "missed": fn, "hit_rate": tp / max(tp + fn, 1), "false_alarm_share": fp / max(tp + fp, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1), "corr": r, "mae": float(np.abs(loss[sel] - y[sel]).mean())}
    return best_t, stats


def samples(labels, units, cells):
    """Season features for every district-season (cached: building them takes minutes)."""
    import joblib
    cp = OUT / "insure_samples.joblib"
    sig = (len(labels), int(labels["yield"].sum()), tuple(sorted(cells)))
    if cp.exists():
        c = joblib.load(cp)
        if c.get("sig") == sig:
            print("  using cached season features")
            return c["df"], c["seq"], c["mask"], c["st"]
    status("Insurance: building season weather features for every district-season...")
    df, seq, mask, st = build_samples(labels, units, cells)
    OUT.mkdir(exist_ok=True)
    joblib.dump({"sig": sig, "df": df, "seq": seq, "mask": mask, "st": st}, cp)
    return df, seq, mask, st


def main():
    labels = pd.read_csv(DATA / "labels.csv.gz")
    units = json.loads((DATA / "units.json").read_text())
    status(f"Insurance: {len(labels):,} district harvests from HarvestStat; fetching rainfall history...")
    cells = load_weather(units)
    df, seq, mask, st = samples(labels, units, cells)
    df = df.copy(); df["i"] = np.arange(len(df))
    df, trend = add_targets(df)
    idx = df.i.to_numpy(); seq, mask, st = seq[idx], mask[idx], st[idx]
    y = df.loss.to_numpy(); yr = df.harvest_year.to_numpy()
    tr, va, te = yr <= TRAIN_END, (yr > TRAIN_END) & (yr <= VAL_END), yr > VAL_END
    # each district's own normal (training years only)
    df["_w"] = st[:, STATIC.index("wrsi")]; df["_r"] = st[:, STATIC.index("rain_ratio")]
    norms = df[tr].groupby("key")[["_w", "_r"]].mean()
    wm = df.key.map(norms["_w"]).fillna(df["_w"].mean()).to_numpy(); rm = df.key.map(norms["_r"]).fillna(1.0).to_numpy()
    st2 = district_extra(st, wm, rm)
    from insure.ndvi import anomalies                         # optional satellite greenness (insure/ndvi_gee.js)
    nd = anomalies(df, tr)
    use_ndvi = nd is not None
    if use_ndvi:
        st2 = np.hstack([st2, nd]).astype(np.float32)
    status(f"Insurance: training on {tr.sum():,} seasons, validating on {va.sum():,}, testing on {te.sum():,} (2016+)...")
    print(f"  bad seasons (>= {BAD:.0%} loss): train {np.mean(y[tr] >= BAD):.1%}, test {np.mean(y[te] >= BAD):.1%}")

    scores, losses, info = {}, {}, {}
    f, p = fit_rain_index(st, y, tr); scores["rain_index"] = f(st); info["rain_index"] = p
    scores["wrsi_index"] = wrsi_index(st, df.crop.to_numpy())
    f, ridge = fit_ridge(seq, st2, y, tr); scores["ridge"] = f(st2, seq, mask)
    deep, name = fit_deep(seq, mask, st2, y, tr, va)
    prob, dl = deep_predict(deep, st2, seq, mask)
    scores[name] = prob; losses[name] = np.clip(dl, -1, 1)
    for k in list(scores):
        if k != name:
            scores[k] = np.clip(scores[k], -1, 1); losses[k] = scores[k]

    results = {}
    for m in scores:
        t, stats = trigger_and_metrics(scores[m], y, tr | va, losses[m])
        results[m] = {"trigger": round(t, 3), "validation": stats(va), "test": stats(te), "train": stats(tr),
                      "trigger_on": "probability of a bad season" if m == name else "predicted loss"}
        r = results[m]["test"]
        print(f"  {m:<14} trigger {t:.2f} | TEST: paid {r['hits']}/{r['bad_seasons']} bad seasons "
              f"(hit rate {r['hit_rate']:.0%}), false alarms {r['false_alarm_share']:.0%}, F1 {r['f1']:.2f}, corr {r['corr']:.2f}")
    best = max(results, key=lambda m: results[m]["validation"]["f1"])
    t = results[best]["trigger"]
    print(f"  -> policies priced with: {best} (best validation F1)")

    # when triggered, pay the predicted loss share of the insured value (at least the 30 % that defines a bad season)
    pay = np.where(scores[best] >= t, np.clip(np.maximum(losses[best], BAD), 0, 1), 0.0)
    df["pred"] = losses[best]; df["score"] = scores[best]; df["payout"] = pay
    policies = {}
    for key, g in df.groupby("key"):
        fnid, crop, season = key.split("|")
        u = units[fnid]; a, b, mean = trend[key]
        last = g.sort_values("harvest_year").iloc[-1]
        exp_now = max(a + b * (last.harvest_year + 1), 0.3 * mean)
        policies[key] = {"fnid": fnid, "crop": crop, "season": season, "name": u["name"], "admin_1": u["admin_1"],
                         "country": u["country"], "lat": u["lat"], "lon": u["lon"],
                         "planting_month": int(last.planting_month), "harvest_month": int(last.harvest_month),
                         "wrsi_mean": round(float(norms["_w"].get(key, df["_w"].mean())), 4),
                         "rain_mean": round(float(norms["_r"].get(key, 1.0)), 4),
                         "expected_t_ha": round(float(exp_now), 3), "pure_rate": round(float(g.payout.mean()), 4),
                         "premium_rate": round(float(g.payout.mean() * LOADING), 4),
                         "history": [{"year": int(r.harvest_year), "actual_loss": round(float(r.loss), 3),
                                      "pred_loss": round(float(r.pred), 3), "score": round(float(r.score), 3),
                                      "payout": round(float(r.payout), 3)}
                                     for r in g.sort_values("harvest_year").itertuples()]}
    OUT.mkdir(exist_ok=True)
    meta = {"best": best, "trigger": t, "trigger_on": results[best]["trigger_on"], "ndvi": use_ndvi, "bad_threshold": BAD,
            "loading": LOADING, "split": {"train_end": TRAIN_END, "val_end": VAL_END}, "models": results,
            "n_policies": len(policies), "n_seasons": int(len(df)), "countries": sorted(df.country.unique().tolist()),
            "weather_cells": len(cells), "source": "HarvestStat Africa v1.2 + NASA POWER"}
    (OUT / "insure_model.json").write_text(json.dumps({"meta": meta, "policies": policies}, separators=(",", ":")))
    import joblib
    keep = {"best": best, "trigger": t, "rain_a": info["rain_index"]["a"], "ridge": ridge,
            "deep": {k: v for k, v in deep.items() if k != "net"}, "deep_name": name, "ndvi": use_ndvi}
    joblib.dump(keep, OUT / "insure_live.joblib")
    status(f"Insurance ready: {len(policies)} district policies, best model {best}.")


if __name__ == "__main__":
    main()
