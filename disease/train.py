"""
DISEASE STEP 2 — train a convolutional neural network to name the disease from a leaf photo.

Model : MobileNetV3-Small (a CNN made for phones), pre-trained on ImageNet (1.2 million everyday photos), then
        TRANSFER LEARNING on the leaf photos:
          phase 1: freeze the pre-trained layers, train only the new last layer (fast)
          phase 2: unfreeze the last blocks and fine-tune them with a smaller learning rate
Data  : data/disease/index.csv (disease/fetch.py). At most MAX_PER_CLASS training photos per class, so big classes
        don't swamp small ones. Augmentation: random crops, flips, rotations, colour changes (photos in the
        field are never framed or lit the same way).
Test  : photos the model never saw. Accuracy is reported SEPARATELY for PlantVillage (plain-background photos,
        easy) and iCassava (real field photos, hard): real-world accuracy is closer to the field number.

    python disease/train.py            # CPU: roughly 20-60 minutes
"""
from pathlib import Path
import csv, json, os, random, time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D, OUT = ROOT / "data" / "disease", ROOT / "outputs"
SIZE = 192
MAX_PER_CLASS = int(os.environ.get("DISEASE_MAX_PER_CLASS", 700))
EVAL_PER_CLASS = 200
EPOCHS_HEAD, EPOCHS_FT = int(os.environ.get("DISEASE_EPOCHS_HEAD", 2)), int(os.environ.get("DISEASE_EPOCHS_FT", 4))
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(n_classes, pretrained=True):
    import torch.nn as nn
    from torchvision import models
    w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v3_small(weights=w)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    return m


def eval_transform():
    from torchvision import transforms as T
    return T.Compose([T.Resize(int(SIZE * 1.1)), T.CenterCrop(SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms as T
    from PIL import Image
    torch.manual_seed(42); random.seed(42)
    rows = list(csv.DictReader(open(D / "index.csv")))
    classes = sorted({r["label"] for r in rows})
    cid = {c: i for i, c in enumerate(classes)}
    rng = random.Random(42)

    def take(split, cap):
        out = []
        for c in classes:
            rs = [r for r in rows if r["split"] == split and r["label"] == c]
            rng.shuffle(rs); out += rs[:cap]
        return out
    tr, va, te = take("train", MAX_PER_CLASS), take("val", EVAL_PER_CLASS), take("test", EVAL_PER_CLASS)
    print(f"{len(classes)} classes · train {len(tr)} · validation {len(va)} · test {len(te)} photos", flush=True)

    aug = T.Compose([T.RandomResizedCrop(SIZE, scale=(0.55, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                     T.RandomRotation(20), T.ColorJitter(0.3, 0.3, 0.3, 0.03), T.ToTensor(), T.Normalize(MEAN, STD)])
    ev = eval_transform()

    class Photos(Dataset):
        def __init__(self, rs, tf): self.rs, self.tf = rs, tf
        def __len__(self): return len(self.rs)
        def __getitem__(self, i):
            r = self.rs[i]
            return self.tf(Image.open(D / r["path"]).convert("RGB")), cid[r["label"]]

    nw = int(os.environ.get("DISEASE_WORKERS", 2))
    dl_tr = DataLoader(Photos(tr, aug), batch_size=32, shuffle=True, num_workers=nw)
    dl_va = DataLoader(Photos(va, ev), batch_size=64, num_workers=nw)
    dl_te = DataLoader(Photos(te, ev), batch_size=64, num_workers=nw)
    model = build_model(len(classes))
    counts = np.bincount([cid[r["label"]] for r in tr], minlength=len(classes)).astype(float)
    cw = torch.tensor(np.sqrt(counts.mean() / np.maximum(counts, 1)), dtype=torch.float32)   # rare classes count more
    lossf = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    def evaluate(dl):
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in dl:
                P.append(model(x).argmax(1).numpy()); Y.append(y.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        return float((P == Y).mean()), P, Y

    def fit(epochs, params, lr, tag):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        best, state = -1, None
        for ep in range(epochs):
            model.train(); t0 = time.time(); n = 0
            for i, (x, y) in enumerate(dl_tr):
                opt.zero_grad(); l = lossf(model(x), y); l.backward(); opt.step(); n += len(y)
                if i % 50 == 0:
                    print(f"  [{tag}] epoch {ep + 1}/{epochs}  {n}/{len(tr)} photos  loss {l.item():.3f}", flush=True)
            acc, _, _ = evaluate(dl_va)
            print(f"  [{tag}] epoch {ep + 1}: validation accuracy {acc:.1%}  ({time.time() - t0:.0f}s)", flush=True)
            if acc > best:
                best, state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        return best

    for p in model.features.parameters():
        p.requires_grad = False
    fit(EPOCHS_HEAD, [p for p in model.parameters() if p.requires_grad], 1e-3, "head")
    for p in model.features[-4:].parameters():
        p.requires_grad = True
    best_val = fit(EPOCHS_FT, [p for p in model.parameters() if p.requires_grad], 3e-4, "fine-tune")

    acc, P, Y = evaluate(dl_te)
    src = np.array([r["source"] for r in te])
    per_class = {}
    for c, i in cid.items():
        m = Y == i
        if m.any():
            per_class[c] = {"n": int(m.sum()), "recall": float((P[m] == i).mean()),
                            "precision": float((Y[P == i] == i).mean()) if (P == i).any() else 0.0}
    by_source = {s: float((P[src == s] == Y[src == s]).mean()) for s in set(src)}
    print(f"\nTEST accuracy {acc:.1%} on {len(te)} unseen photos · " +
          " · ".join(f"{s}: {a:.1%}" for s, a in by_source.items()), flush=True)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), OUT / "disease_model.pt")
    (OUT / "disease_meta.json").write_text(json.dumps({
        "classes": classes, "arch": "mobilenet_v3_small", "size": SIZE, "test_accuracy": acc, "val_accuracy": best_val,
        "by_source": by_source, "per_class": per_class, "n_train": len(tr), "n_test": len(te),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    print("saved outputs/disease_model.pt and outputs/disease_meta.json")


if __name__ == "__main__":
    main()
