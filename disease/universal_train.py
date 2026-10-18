"""
UNIVERSAL LEAF DOCTOR — train and TEST the healthy/sick check so it works on plants it has never seen.

Data : PlantVillage, all 14 plants / 38 classes (PER_CLASS photos each, only those files are downloaded), plus iCassava
       field photos when data/disease has them (15 plants). Each label is mapped to a symptom group (disease/plants.py).
Model: CLIP image features (frozen) -> a small softmax layer over the symptom groups (the PROBE). The final answer
       blends the probe with CLIP zero-shot (text prompts), with blend weight w.
Test : LEAVE-ONE-PLANT-OUT. For each of the 15 plants: train the probe on the other 14, then judge healthy vs sick on
       the plant it never saw. This is exactly the "new plant" situation. w is chosen on this test, then the probe
       is refitted on all plants. Also reported: how often CLIP NAMES the plant right, and how many leaves pass the
       "is it a plant" check.

    python disease/universal_train.py          # CPU: ~10-25 min (download ~6,000 photos + CLIP features)
"""
from pathlib import Path
import csv, hashlib, json, os, random, subprocess, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from disease import plants as P                    # noqa: E402
from disease import universal as U                # noqa: E402

D = ROOT / "data" / "disease"
REPO = D / "pv_sample"
PV_REPO = "https://github.com/spMohanty/PlantVillage-Dataset"
PER_CLASS = int(os.environ.get("UNIVERSAL_PER_CLASS", 120))
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def git(*a, stdin=None, cwd=REPO):
    return subprocess.run(["git", *a], cwd=cwd, input=stdin, text=True, check=True, capture_output=True).stdout


def esc(path):
    s = "".join("\\" + c if c in "*?[\\" else c for c in path)
    return "/" + s


def sample_plantvillage():
    """Download only PER_CLASS photos per PlantVillage class (blobless clone + sparse checkout of chosen files)."""
    if not (REPO / ".git").exists():
        D.mkdir(parents=True, exist_ok=True)
        print("cloning PlantVillage file list (no photos yet)...", flush=True)
        git("clone", "--depth", "1", "--filter=blob:none", "--no-checkout", PV_REPO, str(REPO), cwd=D)
    files = git("ls-tree", "-r", "--name-only", "HEAD", "raw/color").splitlines()
    by = {}
    for f in files:
        by.setdefault(f.split("/")[2], []).append(f)
    rng = random.Random(7)
    rows, pick = [], []
    for folder in sorted(by):
        sp, g = P.pv_group(folder)
        if not sp or not g:
            print(f"  skip {folder} (no mapping)"); continue
        fs = sorted(by[folder]); rng.shuffle(fs)
        for f in fs[:PER_CLASS]:
            pick.append(f); rows.append({"path": str(REPO / f), "species": sp, "group": g, "source": "plantvillage"})
    print(f"downloading {len(pick):,} PlantVillage photos ({len(by)} classes)...", flush=True)
    git("sparse-checkout", "set", "--no-cone", "--stdin", stdin="".join(esc(f) + "\n" for f in pick))
    git("checkout", "-q")
    return [r for r in rows if Path(r["path"]).exists()]


def cassava_rows():
    idx = D / "index.csv"
    if not idx.exists():
        return []
    rng = random.Random(7)
    by = {}
    for r in csv.DictReader(open(idx)):
        if r["source"] == "icassava":
            by.setdefault(r["label"], []).append(r)
    rows = []
    for lab, rs in sorted(by.items()):
        rng.shuffle(rs)
        rows += [{"path": str(D / r["path"]), "species": "cassava", "group": P.PV_SYMPTOM[lab], "source": "icassava"}
                 for r in rs[:PER_CLASS]]
    return rows


def embed(e, rows):
    from PIL import Image
    key = hashlib.md5(("|".join(r["path"] for r in rows) + U.ARCH + U.TAG).encode()).hexdigest()[:12]
    cache = U.OUT / f"leaf_embed_{key}.npy"
    if cache.exists():
        return np.load(cache)
    out, t = [], time.time()
    for i in range(0, len(rows), 32):
        ims = [Image.open(r["path"]).convert("RGB") for r in rows[i:i + 32]]
        out.append(e.b.images(ims))
        if (i // 32) % 20 == 0:
            print(f"  CLIP features {i + len(ims)}/{len(rows)} ({time.time() - t:.0f}s)", flush=True)
    V = np.concatenate(out)
    np.save(cache, V)
    return V


def fit(V, y, classes):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=float(os.environ.get("UNIVERSAL_C", 2.0)), max_iter=3000, class_weight="balanced")
    m.fit(V, y)
    W = np.zeros((len(classes), V.shape[1])); b = np.full(len(classes), -30.0)
    for j, c in enumerate(m.classes_):
        k = classes.index(c)
        if len(m.classes_) == 2:          # sklearn binary case: one row
            W[k] = m.coef_[0] * (1 if j == 1 else -1) / 2; b[k] = m.intercept_[0] * (1 if j == 1 else -1) / 2
        else:
            W[k] = m.coef_[j]; b[k] = m.intercept_[j]
    return W, b


def main():
    t0 = time.time()
    rows = sample_plantvillage() + cassava_rows()
    species = sorted({r["species"] for r in rows})
    print(f"{len(rows):,} photos · {len(species)} plants: {', '.join(species)}", flush=True)
    e = U.Engine(U.Clip())
    e.probe = None
    V = embed(e, rows)
    sp = np.array([r["species"] for r in rows]); grp = np.array([r["group"] for r in rows])
    sick = grp != "healthy"
    classes = [g for g in U.GROUPS if g in set(grp)]

    # zero-shot health for every photo, using its own plant's name (as in the app once the plant is named)
    ZS = np.zeros((len(rows), len(U.GROUPS)))
    for s in species:
        m = sp == s
        ZS[m] = e.zero_shot(V[m], P.PLANTS[P.SPECIES_KEY[s]][0])

    # leave-one-plant-out: probe trained without plant s, tested on s
    PB = np.zeros_like(ZS)
    for s in species:
        tr = sp != s
        W, b = fit(V[tr], grp[tr], classes)
        pr = U.softmax(V[~tr] @ W.T + b)
        for j, c in enumerate(classes):
            PB[~tr, U.GROUPS.index(c)] = pr[:, j]

    def score(probs, t=0.5):
        ps = U.blend_sick(probs)
        per = {}
        for s in species:
            m = sp == s
            acc_sick = float((ps[m & sick] >= t).mean()) if (m & sick).any() else None
            acc_well = float((ps[m & ~sick] < t).mean()) if (m & ~sick).any() else None
            parts = [a for a in (acc_sick, acc_well) if a is not None]
            per[s] = {"balanced_acc": round(float(np.mean(parts)), 3), "sick_found": acc_sick, "healthy_found": acc_well,
                      "n": int(m.sum())}
        top = probs.copy(); top[:, U.GROUPS.index("healthy")] = -1
        sym = float((np.array(U.GROUPS)[top.argmax(1)][sick] == grp[sick]).mean())
        return float(np.mean([v["balanced_acc"] for v in per.values()])), per, sym

    res = {}
    for w in WEIGHTS:
        res[w] = score(w * PB + (1 - w) * ZS)
        print(f"  blend w={w:.2f} (0 = CLIP zero-shot only, 1 = probe only): healthy/sick on UNSEEN plants "
              f"{res[w][0]:.1%} · symptom type right {res[w][2]:.1%}", flush=True)
    best = max(WEIGHTS, key=lambda w: res[w][0])
    th = max(np.arange(0.3, 0.71, 0.05), key=lambda t: score(best * PB + (1 - best) * ZS, t)[0])
    res[best] = score(best * PB + (1 - best) * ZS, th)
    acc, per, sym = res[best]
    ps = U.blend_sick(best * PB + (1 - best) * ZS)
    clear = np.abs(ps - th) >= U.BAND
    clear_acc = float(((ps >= th) == sick)[clear].mean()) if clear.any() else 0.0
    print(f"  decision threshold {th:.2f}: clear answer for {clear.mean():.0%} of photos, right {clear_acc:.1%} of those", flush=True)

    # naming: text only, then + photo prototypes (learned on half the photos, tested on the other half)
    keys, L = e.plant_logits(V)
    truth = np.array([keys.index(P.SPECIES_KEY[s]) for s in sp])
    half = np.random.default_rng(3).random(len(rows)) < 0.5

    def protos(mask):
        out = {}
        for s in species:
            m = mask & (sp == s)
            if m.sum() < 5:
                continue
            v = V[m].mean(0); v = v / np.linalg.norm(v)
            other = V[mask & (sp != s)] @ v
            out[P.SPECIES_KEY[s]] = (v, float(other.mean()), float(other.std() + 1e-6))
        return out

    def naming(Lx, mask):
        rank = (-Lx[mask]).argsort(1); t = truth[mask]
        return float((rank[:, 0] == t).mean()), float((rank[:, :3] == t[:, None]).any(1).mean())

    text1, text3 = naming(L, ~half)
    pa = protos(half)
    alphas = {a: naming(L + e.proto_boost(V, keys, a, pa), ~half) for a in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)}
    for a, (t1, t3) in alphas.items():
        print(f"  plant naming, photo weight {a}: right {t1:.1%} (top-3 {t3:.1%}) on held-out photos", flush=True)
    alpha = max(alphas, key=lambda a: (round(alphas[a][0], 3), -a))
    top1, top3 = alphas[alpha]
    Lb = L + e.proto_boost(V, keys, alpha, pa)
    print(f"  plant named right, per plant (photo weight {alpha}):", flush=True)
    naming_per = {}
    for s_ in species:
        m = (~half) & (sp == s_)
        r = (-Lb[m]).argsort(1)[:, 0]
        naming_per[s_] = round(float((r == truth[m]).mean()), 3)
        wrong = np.bincount(r[r != truth[m]], minlength=len(keys))
        often = keys[int(wrong.argmax())] if wrong.any() else "-"
        print(f"    {s_:11s} {naming_per[s_]:.0%}  (most common wrong name: {often})", flush=True)
    pfull = protos(np.ones(len(rows), bool))
    namer = {}
    if U.NAMER != "off" and U.NAMER_FLAG.exists():
        from PIL import Image
        e.namer = U.Clip(U.NAMER)
        idx = []
        for s_ in species:
            ii = np.where((~half) & (sp == s_))[0][:30]; idx += ii.tolist()
        Vn = np.concatenate([e.namer.images([Image.open(rows[i]["path"]).convert("RGB") for i in idx[j:j + 32]])
                             for j in range(0, len(idx), 32)])
        _, Ln = e.namer_logits(Vn)
        r = (-Ln).argsort(1); t = truth[idx]
        namer = {"top1": round(float((r[:, 0] == t).mean()), 3), "top3": round(float((r[:, :3] == t[:, None]).any(1).mean()), 3)}
        print(f"  species model (BioCLIP): plant named right {namer['top1']:.1%} (top-3 {namer['top3']:.1%}) on {len(idx)} held-out photos", flush=True)
        for s_ in species:
            m = sp[idx] == s_
            print(f"    {s_:11s} {float((r[m, 0] == t[m]).mean()):.0%}  (most common answer: {keys[int(np.bincount(r[m, 0]).argmax())]})", flush=True)
    gate = float((e.p_plant(V) >= U.PLANT_MIN).mean())

    W, b = fit(V, grp, classes)
    out = {"classes": classes, "W": W.round(6).tolist(), "b": b.round(6).tolist(), "blend_w": best, "threshold": round(float(th), 2),
           "loso": {"balanced_acc": round(acc, 3), "zero_shot_only": round(res[0.0][0], 3),
                    "probe_only": round(res[1.0][0], 3), "symptom_acc": round(sym, 3), "per_plant": per,
                    "clear_share": round(float(clear.mean()), 3), "clear_acc": round(clear_acc, 3)},
           "naming": {"top1": round(top1, 3), "top3": round(top3, 3), "plants": len(species),
                      "text_only_top1": round(text1, 3), "photo_weight": alpha, "per_plant": naming_per, "species_model": namer},
           "proto": {k: {"v": v.round(5).tolist(), "mu": mu, "sd": sd} for k, (v, mu, sd) in pfull.items()},
           "proto_alpha": alpha,
           "gate": {"leaves_passed": round(gate, 3)}, "n": len(rows), "arch": U.ARCH, "tag": U.TAG,
           "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    U.OUT.mkdir(exist_ok=True)
    U.PROBE.write_text(json.dumps(out))
    print("\nper plant (never seen in training):")
    for s, v in per.items():
        print(f"  {s:11s} healthy/sick {v['balanced_acc']:.0%}  (n={v['n']})")
    print(f"\nUNIVERSAL TEST healthy/sick on unseen plants {acc:.1%} (zero-shot only {res[0.0][0]:.1%}, "
          f"probe only {res[1.0][0]:.1%}, chosen blend w={best}, threshold {th:.2f}) · clear answer {clear.mean():.0%} of photos, right {clear_acc:.1%} · symptom type {sym:.1%} · "
          f"plant named right {top1:.1%} (top-3 {top3:.1%}; names only {text1:.1%}) · leaves recognised as plants {gate:.1%} · "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
