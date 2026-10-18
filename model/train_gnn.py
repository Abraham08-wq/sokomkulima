"""
STEP 2b — DEEP LEARNING: DYNAMIC SPATIO-TEMPORAL GRAPH NEURAL NETWORK  (needs PyTorch + PyTorch Geometric)

    python model/train_gnn.py --slug maize_retail

Model (per forecast month):
  1. Every market gets its last 12 months of features (price change, price level, season, data flag)
     plus a learned "market embedding" that captures each market's own character.
  2. GRAPH ATTENTION (GATv2) runs on the market graph at EVERY month in the window. Each edge carries
     [distance, cross-border flag, this month's price correlation], and the attention decides how much each
     market listens to each neighbour. Because the inputs change monthly, the learned links change monthly:
     this is the dynamic graph.
  3. A GRU (recurrent network) reads the 12 graph-updated months in order and learns trends and seasons.
  4. The output head predicts the log price change 1, 2 and 3 months ahead for every market at once.

For a fair comparison the same network is also trained WITHOUT the graph layer ("gru_no_graph").
Both are scored on the same held-out months as the baselines in model/baseline.py.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from common import load, filled, month_numbers, split_origins, score, save_result, HORIZONS, OUT

LOOKBACK = 12
PATIENCE = int(os.environ.get("PATIENCE", 8))   # stop after 8 epochs without improvement on the validation months
torch.manual_seed(42); np.random.seed(42)


class DynamicSTGNN(nn.Module):
    def __init__(self, n_nodes, in_dim, hid=32, heads=4, emb=8, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        self.emb = nn.Embedding(n_nodes, emb)
        self.inp = nn.Linear(in_dim + emb, hid)
        if use_graph:
            self.gat = GATv2Conv(hid, hid // heads, heads=heads, edge_dim=3, add_self_loops=True, dropout=0.2)
            self.norm = nn.LayerNorm(hid)
        self.gru = nn.GRU(hid, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, len(HORIZONS)))

    def forward(self, x, edge_index, edge_attr):
        # x: (L, N, F)   edge_attr: (L, E, 3)
        L, N, _ = x.shape
        e = self.emb.weight.unsqueeze(0).expand(L, N, -1)
        h = torch.relu(self.inp(torch.cat([x, e], -1)))
        att = None
        if self.use_graph:
            steps = []
            for l in range(L):
                g, att = self.gat(h[l], edge_index, edge_attr[l], return_attention_weights=True)
                steps.append(self.norm(torch.relu(g) + h[l]))       # residual: own signal + neighbours'
            h = torch.stack(steps)
        out, _ = self.gru(h.permute(1, 0, 2))                         # (N, L, hid)
        return self.head(out[:, -1]) / 10.0, att                      # scaled log-change forecasts


def build_inputs(logp, F, dyn, edges, dist, cross, months):
    T, N = logp.shape
    d = np.vstack([np.zeros((1, N)), np.diff(F, axis=0)])
    lvl = np.stack([F[t] - F[max(0, t - 36):t + 1].mean(0) for t in range(T)])
    s = np.sin(2 * np.pi * months / 12)[:, None].repeat(N, 1); c = np.cos(2 * np.pi * months / 12)[:, None].repeat(N, 1)
    obs = (~np.isnan(logp)).astype(float)
    X = np.stack([d * 10, lvl * 5, s, c, obs], -1).astype(np.float32)               # (T, N, 5)
    ei = np.concatenate([edges, edges[:, ::-1]]).T                                  # both directions
    ea_static = np.stack([np.concatenate([dist, dist]) / 1000.0, np.concatenate([cross, cross])], -1)
    dynw = np.nan_to_num(np.concatenate([dyn, dyn], 1))                             # (T, 2E)
    EA = np.concatenate([np.repeat(ea_static[None], T, 0), dynw[..., None]], -1).astype(np.float32)
    return torch.tensor(X), torch.tensor(ei.copy(), dtype=torch.long), torch.tensor(EA)


def targets(logp, F, t):
    y = np.stack([logp[t + h] - F[t] for h in HORIZONS], -1)
    return torch.tensor(np.nan_to_num(y), dtype=torch.float32), torch.tensor(~np.isnan(y))


def run(t, model, X, EI, EA):
    return model(X[t - LOOKBACK + 1:t + 1], EI, EA[t - LOOKBACK + 1:t + 1])


def train_one(name, use_graph, logp, F, X, EI, EA, train, test, epochs, lr):
    N = logp.shape[1]
    model = DynamicSTGNN(N, X.shape[-1], use_graph=use_graph)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)                 # stronger L2 penalty against over-fitting
    val = train[-12:]; fit = train[:-12]
    best, best_state, wait = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for t in np.random.permutation(fit):
            y, m = targets(logp, F, t)
            p, _ = run(t, model, X, EI, EA)
            loss = ((p - y) ** 2 * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = []
            for t in val:
                y, m = targets(logp, F, t); p, _ = run(t, model, X, EI, EA)
                vl.append((((p - y) ** 2 * m).sum() / m.sum().clamp(min=1)).item())
        v = float(np.mean(vl))
        if v < best - 1e-6:
            best, best_state, wait = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            wait += 1
        if ep % 10 == 0:
            print(f"  [{name}] epoch {ep:3d}  val loss {v:.5f}")
        if wait >= PATIENCE:
            print(f"  [{name}] early stop at epoch {ep}: best validation loss {best:.5f} at epoch {ep - PATIENCE}"
                  " -> keeping that epoch's weights"); break
    model.load_state_dict(best_state); model.eval()
    return model


def predict(model, ts, X, EI, EA):
    with torch.no_grad():
        return np.stack([run(t, model, X, EI, EA)[0].numpy() for t in ts])


def attention_by_month(model, T, X, EI, EA, edges):
    """Average attention weight of every undirected market link, for every month (the dynamic graph)."""
    E = len(edges); W = np.zeros((T, E), dtype=np.float32)
    with torch.no_grad():
        for t in range(LOOKBACK - 1, T):
            _, (ei, a) = run(t, model, X, EI, EA)
            a = a.mean(1).numpy(); ei = ei.numpy()
            lookup = {(int(i), int(j)): float(w) for i, j, w in zip(ei[0], ei[1], a)}
            for k, (i, j) in enumerate(edges):
                W[t, k] = (lookup.get((i, j), 0) + lookup.get((j, i), 0)) / 2
    mx = W.max() or 1
    return W / mx


def main(slug, epochs, lr):
    logp, edges, dyn, dist, cross, meta = load(slug)
    T, N = logp.shape
    F = filled(logp); months = month_numbers(meta)
    train, test = split_origins(T, LOOKBACK)
    X, EI, EA = build_inputs(logp, F, dyn, edges, dist, cross, months)
    comp_path = OUT / f"{slug}__comparison.json"
    comparison = json.loads(comp_path.read_text()) if comp_path.exists() else {}

    for name, use_graph in [("gru_no_graph", False), ("st_gnn", True)]:
        print(f"\nTraining {name}...")
        model = train_one(name, use_graph, logp, F, X, EI, EA, train, test, epochs, lr)
        pt = predict(model, test, X, EI, EA)
        metrics = score(pt, logp, test, F)
        tr = predict(model, train, X, EI, EA)
        ytr = np.stack([np.stack([logp[t + h] - F[t] for h in HORIZONS], -1) for t in train])
        sd = np.array([np.nanstd(ytr[..., i] - tr[..., i]) for i in range(len(HORIZONS))])
        final = predict(model, [T - 1], X, EI, EA)[0]
        W = attention_by_month(model, T, X, EI, EA, edges) if use_graph else np.clip(np.nan_to_num(dyn), 0, 1)
        save_result(slug, name, metrics, logp, F, meta, final, sd, pt, test, W)
        comparison[name] = {str(h): v for h, v in metrics.items()}
        torch.save(model.state_dict(), OUT / f"{slug}__{name}.pt")
        print("  MAPE: " + "  ".join(f"h={h}: {metrics[h]['mape']:.2f}%" for h in HORIZONS))

    comp_path.write_text(json.dumps(comparison, indent=1))
    print("\nComparison (test MAPE %, lower is better):")
    for k, v in comparison.items():
        print(f"  {k:<14}" + "".join(f"  {v[str(h)]['mape']:7.2f}" for h in HORIZONS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="maize_retail")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    main(a.slug, a.epochs, a.lr)
