"""
LIVE UPDATE for many crops: fetch -> (re)build graphs -> (re)train, only for what actually changed.

    python pipeline/update.py            # normally run by the backend at start-up and every REFRESH_HOURS
    python pipeline/update.py --force    # retrain everything

Order (so the dashboard is useful quickly):
  Phase 1: for every crop, build its market graph + train the fast models (graph regression)  -> minutes
  Phase 2: for every crop, train the deep graph neural network                                -> slower
After each crop, the dashboard is told to reload (status.json "version" goes up).

Environment: LIVE, CROPS (comma list), PRICETYPE, TRAIN_GNN, EPOCHS, COUNTRIES.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess, sys

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT, PROC = ROOT / "data" / "raw", ROOT / "outputs", ROOT / "data" / "processed"
STATUS = OUT / "status.json"
PY = sys.executable
DEFAULT_CROPS = "maize,sorghum,millet,rice,wheat,teff,beans,cowpea,groundnut,cassava"   # storable staples


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def crops():
    c = os.environ.get("CROPS") or os.environ.get("CROP") or DEFAULT_CROPS
    return [x.strip().lower() for x in c.split(",") if x.strip()]


def read_status():
    try:
        return json.loads(STATUS.read_text())
    except Exception:
        return {}


def write_status(**kw):
    OUT.mkdir(exist_ok=True)
    s = {**read_status(), **kw}
    STATUS.write_text(json.dumps(s, indent=1))
    return s


def bump(on_change=None, **kw):
    s = write_status(version=read_status().get("version", 0) + 1, **kw)
    if on_change:
        on_change()
    return s


def run(args, cwd=ROOT, timeout=None):
    print("$ " + " ".join(map(str, args)), flush=True)
    try:
        p = subprocess.run(args, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s (will retry on the next update)"
    if p.stderr:
        sys.stderr.write(p.stderr)
    return p.returncode, (p.stderr.strip().splitlines() or [""])[-1]


def has_gnn():
    try:
        import torch_geometric  # noqa: F401
        return os.environ.get("TRAIN_GNN", "1") == "1"
    except Exception:
        return False


def data_signature():
    manifest_p = RAW / "live_manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}
    real = [f for f in RAW.glob("*.csv") if not f.name.startswith("DEMO_")]
    if not real:
        return "DEMO"
    return manifest.get("signature", "") + "|" + "|".join(f"{f.name}:{f.stat().st_size}" for f in sorted(real))


def summary(cs):
    ready = {k: v for k, v in cs.items() if v.get("state") == "ready"}
    metas = [json.loads((PROC / f"{k}.json").read_text()) for k in ready if (PROC / f"{k}.json").exists()]
    return dict(data_as_of=max([m.get("data_as_of", "") for m in metas], default=None),
                countries=len({c for m in metas for c in m.get("countries", [])}),
                markets=sum(len(m["markets"]) for m in metas), crops_ready=len(ready))


def insurance(on_change=None, force=False):
    """Phase 3: train the crop-insurance model (once, and again each new year of weather or new labels)."""
    if os.environ.get("INSURE", "1") != "1":
        return
    labels = ROOT / "insure" / "data" / "labels.csv.gz"
    try:
        import torch  # noqa: F401
        deep = "torch"
    except Exception:
        deep = "mlp"
    from datetime import date as _d
    nd = next((p for p in (ROOT / "data" / "ndvi_districts.csv", ROOT / "insure" / "data" / "ndvi_districts.csv") if p.exists()), None)
    sig = (f"v2|{labels.stat().st_size if labels.exists() else 0}|{_d.today().year - 1}|{deep}"   # v2: bad-season detector
           f"|ndvi={nd.stat().st_size if nd else 0}")                                          # retrain when NDVI arrives
    st = read_status()
    if not force and st.get("insure_sig") == sig and (OUT / "insure_model.json").exists():
        return
    write_status(state="training", message="Training the crop-insurance model (first time: downloads rainfall history)...")
    code, err = run([PY, "insure/train.py"])
    if code == 0:
        bump(on_change, insure_sig=sig, insure_trained_at=now())
    else:
        write_status(insure_error=err[-200:])


def disease(on_change=None):
    """Phase 4: download leaf photos and train the disease CNN (once; DISEASE=0 to skip)."""
    if os.environ.get("DISEASE", "1") != "1" or (OUT / "disease_model.pt").exists():
        return
    try:
        import torchvision  # noqa: F401
    except Exception:
        write_status(disease_error="torchvision not installed: photo diagnosis skipped"); return
    write_status(state="training", message="Downloading leaf photos for the disease model (PlantVillage, iCassava)...")
    code, err = run([PY, "disease/fetch.py"])
    if code != 0:
        write_status(disease_error=err[-200:]); return
    write_status(state="training", message="Training the crop-disease CNN on leaf photos (CPU: 20-60 min)...")
    code, err = run([PY, "disease/train.py"])
    if code == 0:
        bump(on_change, disease_trained_at=now())
    else:
        write_status(disease_error=err[-200:])


def leaf_warm():
    """Universal leaf doctor, step 1: download CLIP (~600 MB, once) so any-plant photo checks work early."""
    namer_needed = os.environ.get("NAMER", "hf-hub:imageomics/bioclip") != "off" and not (OUT / "namer_ready.json").exists()
    if os.environ.get("UNIVERSAL", "1") != "1" or ((OUT / "clip_ready.json").exists() and not namer_needed):
        return
    try:
        import open_clip  # noqa: F401
    except Exception:
        write_status(universal_error="open_clip not installed: universal leaf check skipped"); return
    write_status(message="Downloading the plant models (CLIP and the BioCLIP species model, ~600 MB each, once)...")
    code, err = run([PY, "disease/universal.py", "--warm"], timeout=int(os.environ.get("CLIP_TIMEOUT", 4800)))
    if code != 0:
        write_status(universal_error=err[-200:])


def leaf_probe(on_change=None):
    """Universal leaf doctor, step 2: train + test the healthy/sick layer on 15 plants (leave-one-plant-out)."""
    if os.environ.get("UNIVERSAL", "1") != "1" or (OUT / "leaf_probe.json").exists() or not (OUT / "clip_ready.json").exists():
        return
    write_status(state="training", message="Training the universal healthy/sick check on 15 plants (CPU: 10-25 min)...")
    code, err = run([PY, "disease/universal_train.py"])
    if code == 0:
        bump(on_change, universal_trained_at=now())
    else:
        write_status(universal_error=err[-200:])


def update(force=False, on_change=None):
    leaf_warm()
    ptype = os.environ.get("PRICETYPE", "Retail")
    write_status(state="checking", message="Checking WFP for new prices...", checked_at=now())
    live_ok = False
    if os.environ.get("LIVE", "1") == "1":
        live_ok = subprocess.run([PY, "pipeline/fetch_wfp.py"], cwd=ROOT).returncode == 0
    if not list(RAW.glob("*.csv")):
        print("No real WFP data reachable -> generating DEMO data (maize only)")
        run([PY, "pipeline/demo_data.py"])
    sig = data_signature()
    demo = sig == "DEMO"
    gnn = has_gnn()
    st = read_status()
    cs = st.get("crops", {})
    # migrate a model trained by the single-crop version, so it is not retrained for nothing
    old = st.get("signature", "")
    if old and not cs:
        for c in crops():
            slug = f"{c}_{ptype.lower()}"
            if old.startswith(sig + f"|{slug}|") and (OUT / f"{slug}__graph_ridge.json").exists():
                cs[slug] = {"state": "ready", "signature": sig, "gnn": (OUT / f"{slug}__st_gnn.json").exists()}
    todo = [c for c in crops() if force or cs.get(f"{c}_{ptype.lower()}", {}).get("signature") != sig
            or (cs[f"{c}_{ptype.lower()}"].get("state") == "ready"
                and not (OUT / f"{c}_{ptype.lower()}__graph_ridge.json").exists())]
    need_gnn = [c for c in crops() if gnn and cs.get(f"{c}_{ptype.lower()}", {}).get("state") == "ready"
                and not cs[f"{c}_{ptype.lower()}"].get("gnn") and c not in todo]

    if not todo and not need_gnn:
        insurance(on_change, force)
        disease(on_change)
        leaf_probe(on_change)
        return write_status(state="demo" if demo else "live", crops=cs, live_fetch_ok=live_ok,
                            message="Up to date: no new WFP prices.", **summary(cs))

    # ---- phase 1: graph + fast models for every crop that changed ----
    for i, c in enumerate(todo, 1):
        slug = f"{c}_{ptype.lower()}"
        write_status(state="training", crops=cs, training_started=read_status().get("training_started") or now(),
                     message=f"New WFP prices: building the {c} market graph ({i}/{len(todo)})...")
        code, err = run([PY, "pipeline/prepare.py", "--commodity", c, "--pricetype", ptype])
        if code != 0:
            cs[slug] = {"state": "skipped", "signature": sig, "reason": err[-160:]}
            print(f"  skipped {c}: {err}")
            continue
        code, err = run([PY, "baseline.py", "--slug", slug], cwd=ROOT / "model")
        if code != 0:
            cs[slug] = {"state": "error", "signature": sig, "reason": err[-160:]}
            continue
        for f in OUT.glob(f"{slug}__st_gnn*"):     # an old deep model belongs to old data
            f.unlink()
        for f in OUT.glob(f"{slug}__gru_no_graph*"):
            f.unlink()
        cs[slug] = {"state": "ready", "signature": sig, "gnn": False, "trained_at": now()}
        bump(on_change, crops=cs, trained_at=now(), **summary(cs))
        if gnn:
            need_gnn.append(c)

    # ---- phase 2: deep graph neural network, crop by crop ----
    for i, c in enumerate(need_gnn, 1):
        slug = f"{c}_{ptype.lower()}"
        write_status(state="training", crops=cs,
                     message=f"Training the graph neural network for {c} ({i}/{len(need_gnn)}). Other models are live.")
        code, err = run([PY, "train_gnn.py", "--slug", slug, "--epochs", os.environ.get("EPOCHS", "60")],
                        cwd=ROOT / "model")
        if code == 0:
            cs[slug]["gnn"] = True
            bump(on_change, crops=cs, trained_at=now())
        else:
            cs[slug]["gnn_error"] = err[-160:]

    insurance(on_change, force)
    disease(on_change)
    leaf_probe(on_change)
    ready = [k for k, v in cs.items() if v.get("state") == "ready"]
    skipped = [k.split("_")[0] for k, v in cs.items() if v.get("state") == "skipped"]
    return write_status(state=("demo" if demo else "live") if ready else "error", crops=cs, live_fetch_ok=live_ok,
                        training_started=None, **summary(cs),
                        message=(f"Trained on the latest WFP prices: {len(ready)} crops." if not demo
                                 else "Running on DEMO data: WFP could not be reached.")
                        + (f" Not enough data for: {', '.join(skipped)}." if skipped else ""))


if __name__ == "__main__":
    s = update(force="--force" in sys.argv)
    print(f"status: {s['state']} - {s['message']}")
    sys.exit(0 if s["state"] in ("live", "demo") else 1)
