"""DISEASE STEP 3 — diagnose one photo with the trained CNN (used by /api/diagnose)."""
from pathlib import Path
import io, json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SURE = 0.70                     # below this the app says "not sure" instead of guessing
_m = {}


def ready():
    return (OUT / "disease_model.pt").exists() and (OUT / "disease_meta.json").exists()


def meta():
    return json.loads((OUT / "disease_meta.json").read_text())


def _load():
    p = OUT / "disease_model.pt"
    if _m.get("mtime") != p.stat().st_mtime:
        import torch
        from disease.train import build_model, eval_transform
        mt = meta()
        net = build_model(len(mt["classes"]), pretrained=False)
        net.load_state_dict(torch.load(p, map_location="cpu")); net.eval()
        _m.update(mtime=p.stat().st_mtime, net=net, meta=mt, tf=eval_transform())
    return _m


def diagnose(image_bytes, crop=None):
    """-> {'top': [(label, prob), ...], 'sure': bool}. If crop is given, only that crop's classes compete."""
    import torch
    from PIL import Image, ImageOps
    m = _load()
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    with torch.no_grad():
        logits = m["net"](m["tf"](img).unsqueeze(0))[0].numpy()
    classes = m["meta"]["classes"]
    keep = np.array([(not crop) or c.startswith(crop + "_") for c in classes])
    if not keep.any():
        keep[:] = True
    z = np.where(keep, logits, -np.inf); z = z - z.max()
    p = np.exp(z); p = p / p.sum()
    order = np.argsort(-p)[:3]
    top = [(classes[i], float(p[i])) for i in order if p[i] > 0]
    p_healthy = float(sum(p[i] for i, c in enumerate(classes) if c.endswith("_healthy") and keep[i]))
    return {"top": top, "sure": top[0][1] >= SURE, "p_sick": 1.0 - p_healthy}
