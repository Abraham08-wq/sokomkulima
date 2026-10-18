"""
UNIVERSAL LEAF DOCTOR — works on ANY plant photo, not only the crops the specialist model was trained on.

Model: CLIP (OpenAI/LAION, open_clip ViT-B-32 trained on 2 billion image-text pairs). CLIP puts a photo and a sentence
in the same space, so it can compare a photo with sentences it has never been trained on ("zero-shot"):
  1. IS IT A PLANT?   photo vs "a close-up photo of a plant leaf", "a selfie", "a document", ...      -> not-a-plant message
  2. WHICH PLANT?     photo vs "a photo of a <plant> leaf" for ~120 African crops, fruit, trees, weeds -> name + alternatives
                      (or Pl@ntNet, 50,000 species, when PLANTNET_API_KEY is set)
  3. HEALTHY OR SICK? photo vs "a healthy green <plant> leaf", "a <plant> leaf with orange rust pustules", ... (11 groups)
                      + a small trained layer (PROBE) on CLIP features from 15 plants' labelled photos
                      (disease/universal_train.py), tested on plants it never saw (leave-one-plant-out).
  4. ADVICE           sick -> steps for that symptom; healthy -> good practice (disease/care.py)
For maize, potato, tomato and cassava the specialist CNN (disease/predict.py) also names the exact disease.

    python disease/universal.py --warm        # download the CLIP weights (~600 MB, once) into outputs/clip
"""
from pathlib import Path
import io, json, os, sys, threading, time, urllib.request, uuid
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from disease import plants as P   # noqa: E402

# Hugging Face's newer "xet" download stalls on some connections: use the plain HTTPS download, with timeouts
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

OUT = ROOT / "outputs"
CACHE = OUT / "clip"
FLAG = OUT / "clip_ready.json"
PROBE = OUT / "leaf_probe.json"
ARCH = os.environ.get("CLIP_ARCH", "ViT-B-32")
TAG = os.environ.get("CLIP_TAG", "laion2b_s34b_b79k")
# naming model: BioCLIP (Stevens et al. 2024, Ohio State Imageomics) — CLIP trained on 10 million labelled photos of
# living things (TreeOfLife-10M), so it names plant SPECIES far better than the general CLIP. NAMER=off to disable.
NAMER = os.environ.get("NAMER", "hf-hub:imageomics/bioclip")
NAMER_FLAG = OUT / "namer_ready.json"
SCALE = 100.0                 # CLIP's own temperature
PLANT_MIN = 0.5               # below this share of "plant" prompts -> not a plant
BAND = 0.08                  # p_sick within threshold +- BAND -> "take a clearer photo"; otherwise a clear answer
GROUPS = list(P.SYMPTOMS)


def available():
    try:
        import open_clip  # noqa: F401
        return True
    except Exception:
        return False


def ready():
    return FLAG.exists()


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class Clip:
    """The real CLIP backend (needs torch + open_clip). Tests use a fake with the same two methods."""
    def __init__(self, arch=ARCH, tag=TAG):
        import torch, open_clip
        CACHE.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(max(1, int(os.environ.get("CLIP_THREADS", 4))))
        self.torch = torch
        if arch.startswith("hf-hub:"):
            self.model, _, self.pre = open_clip.create_model_and_transforms(arch, cache_dir=str(CACHE))
        else:
            self.model, _, self.pre = open_clip.create_model_and_transforms(arch, pretrained=tag, cache_dir=str(CACHE))
        self.model.eval()
        self.tok = open_clip.get_tokenizer(arch)

    def _norm(self, f):
        f = f.float().numpy()
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    def images(self, pil_list):
        with self.torch.no_grad():
            return self._norm(self.model.encode_image(self.torch.stack([self.pre(im) for im in pil_list])))

    def texts(self, prompts):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(prompts), 128):
                out.append(self._norm(self.model.encode_text(self.tok(prompts[i:i + 128]))))
        return np.concatenate(out)


class Engine:
    def __init__(self, backend, namer=None):
        self.b = backend
        self.namer = namer
        self._cache = {}
        self.probe = None
        self.load_probe()

    def load_probe(self):
        if PROBE.exists():
            j = json.loads(PROBE.read_text())
            self.probe = dict(classes=j["classes"], W=np.array(j["W"]), b=np.array(j["b"]), w=float(j.get("blend_w", 0.5)),
                              mtime=PROBE.stat().st_mtime, meta=j,
                              proto={k: (np.array(v["v"]), v["mu"], v["sd"]) for k, v in j.get("proto", {}).items()},
                              alpha=float(j.get("proto_alpha", 0.0)))

    def groups(self, groups, b=None):
        """list of prompt lists -> one unit vector per group (prompt ensembling). Cached."""
        b = b or self.b
        key = (id(b), tuple(tuple(g) for g in groups))
        if key not in self._cache:
            flat = [p for g in groups for p in g]
            T = b.texts(flat)
            out, i = [], 0
            for g in groups:
                v = T[i:i + len(g)].mean(0); out.append(v / np.linalg.norm(v)); i += len(g)
            self._cache[key] = np.stack(out)
        return self._cache[key]

    # ---- 1. plant or not
    def p_plant(self, V):
        T = self.groups([[s] for s in P.GATE_PLANT + P.GATE_OTHER])
        pr = softmax(SCALE * V @ T.T)
        return pr[:, :len(P.GATE_PLANT)].sum(1)

    # ---- 2. which plant
    def plant_logits(self, V):
        """text matching: every name of the plant (main name + ALIASES) with a few sentence forms."""
        keys = list(P.PLANTS)
        groups = []
        for k in keys:
            names = [P.PLANTS[k][0]] + P.ALIASES.get(k, [])
            groups.append([f"a photo of a {names[0]} leaf", f"a close-up photo of {names[0]} leaves"] +
                          [f"a photo of a {n} plant" for n in names])
        return keys, SCALE * V @ self.groups(groups).T

    def proto_boost(self, V, keys, alpha=None, proto=None):
        """learned from real photos (15 plants): a photo that looks much more like the cassava photos than other plants'
        photos usually do (z-score above 2) gets extra weight for cassava."""
        proto = proto if proto is not None else (self.probe or {}).get("proto", {})
        alpha = alpha if alpha is not None else (self.probe or {}).get("alpha", 0.0)
        B = np.zeros((len(V), len(keys)))
        if alpha <= 0:
            return B
        for k, (v, mu, sd) in proto.items():
            if k in keys:
                z = (V @ v - mu) / max(sd, 1e-6)
                B[:, keys.index(k)] = alpha * np.maximum(0.0, z - 2.0)
        return B

    def plant_probs(self, V):
        keys, L = self.plant_logits(V)
        return keys, softmax(L + self.proto_boost(V, keys))

    def namer_logits(self, Vn):
        """species model: scientific name + common names, as BioCLIP was trained ("a photo of Manihot esculenta.")."""
        keys = list(P.PLANTS)
        groups = []
        for k in keys:
            names = [P.SCI[k], P.PLANTS[k][0].split(" (")[0]] + P.ALIASES.get(k, [])
            groups.append([f"a photo of {n}." for n in names])
        return keys, SCALE * Vn @ self.groups(groups, self.namer).T

    def name_plant(self, V, img):
        """-> keys, probabilities. The species model when installed, else CLIP (+ photo prototypes)."""
        if self.namer is not None and img is not None:
            keys, L = self.namer_logits(self.namer.images([img]))
            return keys, softmax(L)
        return self.plant_probs(V)

    # ---- 3. healthy or sick: zero-shot, probe and blend. Returns (n, len(GROUPS)) probabilities
    def zero_shot(self, V, plant_name="plant"):
        T = self.groups([[s.format(p=plant_name) for s in P.SYMPTOMS[g]] for g in GROUPS])
        return rebalance(softmax(SCALE * V @ T.T), GROUPS)

    def probe_probs(self, V):
        if not self.probe:
            return None
        pr = rebalance(softmax(V @ self.probe["W"].T + self.probe["b"]), self.probe["classes"])
        full = np.zeros((len(V), len(GROUPS)))
        for j, c in enumerate(self.probe["classes"]):
            full[:, GROUPS.index(c)] = pr[:, j]
        return full

    def health(self, V, plant_name="plant", w=None):
        zs = self.zero_shot(V, plant_name)
        pb = self.probe_probs(V)
        if pb is None:
            return zs, zs
        w = self.probe["w"] if w is None else w
        return w * pb + (1 - w) * zs, zs


def rebalance(pr, classes):
    """Equal odds for 'not sick' and 'sick' BEFORE looking at the photo. Without this, 1 healthy description against
    10 sick ones means a photo that matches nothing clearly comes out ~90% 'sick' (the healthy-cassava mistake)."""
    ok = [c in P.NOT_SICK for c in classes]
    n_ok, n_bad = sum(ok), len(classes) - sum(ok)
    w = np.array([(0.5 / n_ok if o else 0.5 / n_bad) * len(classes) for o in ok])
    q = pr * w
    return q / q.sum(axis=1, keepdims=True)


def threshold():
    try:
        return float(json.loads(PROBE.read_text()).get("threshold", 0.5))
    except Exception:
        return 0.5


def blend_sick(probs):
    """chance of a disease or pest; natural ageing / autumn colour (old_leaf) is not counted as sick."""
    return 1 - sum(probs[:, GROUPS.index(g)] for g in P.NOT_SICK)


# ---------------------------------------------------------------- service (one shared engine, loaded on first use)
_E, _lock = {}, threading.Lock()


def engine():
    with _lock:
        if "e" not in _E:
            _E["e"] = Engine(Clip())
            FLAG.write_text(json.dumps({"arch": ARCH, "tag": TAG, "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        e = _E["e"]
        if e.namer is None and NAMER != "off" and NAMER_FLAG.exists() and not _E.get("namer_failed"):
            try:
                e.namer = Clip(NAMER)
            except Exception as ex:
                _E["namer_failed"] = True
                print(f"species model not loaded: {type(ex).__name__}: {ex}", flush=True)
        if PROBE.exists() and (not e.probe or e.probe["mtime"] != PROBE.stat().st_mtime):
            e.load_probe()
        return e


def load_image(image_bytes):
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")


def colour_stats(img):
    """Share of the leaf that is green vs yellow/orange/red (HSV). Background = pale or dark pixels.
    A leaf with almost no green left (autumn / old leaf) gets no confident "sick" verdict: on 135 PlantVillage test
    leaves (healthy and 8 diseases) only 2 fully yellowed diseased leaves were flagged, and those get "not sure"."""
    hsv = np.asarray(img.convert("RGB").resize((128, 128)).convert("HSV")).astype(float) / 255.0
    h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    leaf = (s > 0.25) & (v > 0.15)
    n = int(leaf.sum())
    if n < 0.05 * leaf.size:
        return {"leaf": n / leaf.size, "green": None, "warm": None}
    green = leaf & (h >= 65) & (h <= 170)
    warm = leaf & ((h < 55) | (h > 330))
    return {"leaf": round(n / leaf.size, 3), "green": round(float(green.sum()) / n, 3), "warm": round(float(warm.sum()) / n, 3)}


def is_autumn(c):
    return c["green"] is not None and c["green"] < 0.08 and c["warm"] > 0.7


def plantnet(image_bytes, lang="en"):
    """Optional: Pl@ntNet identification API (free key at my.plantnet.org, 500 photos/day). None if not set or fails."""
    key = os.environ.get("PLANTNET_API_KEY", "").strip()
    if not key:
        return None
    try:
        b = uuid.uuid4().hex
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"organs\"\r\n\r\nauto\r\n"
                f"--{b}\r\nContent-Disposition: form-data; name=\"images\"; filename=\"leaf.jpg\"\r\n"
                f"Content-Type: image/jpeg\r\n\r\n").encode() + image_bytes + f"\r\n--{b}--\r\n".encode()
        url = f"https://my-api.plantnet.org/v2/identify/all?api-key={key}&lang={'fr' if lang == 'fr' else 'en'}&nb-results=3"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            res = json.loads(r.read()).get("results", [])
        out = []
        for x in res[:3]:
            sp = x.get("species", {})
            common = (sp.get("commonNames") or [None])[0]
            out.append({"scientific": sp.get("scientificNameWithoutAuthor"), "common": common, "prob": float(x.get("score", 0))})
        return out or None
    except Exception as e:
        print(f"plantnet failed: {type(e).__name__}: {e}", flush=True)
        return None


def match_key(common, scientific=""):
    """Pl@ntNet common/scientific name -> our PLANTS key (for advice), or None."""
    words = f"{common or ''} {scientific or ''}".lower()
    sci = {"zea mays": "maize", "manihot esculenta": "cassava", "solanum lycopersicum": "tomato", "solanum tuberosum": "potato",
           "phaseolus vulgaris": "bean", "coffea": "coffee", "camellia sinensis": "tea", "musa": "banana", "mangifera": "mango",
           "persea americana": "avocado", "citrus": "orange", "sorghum": "sorghum", "oryza": "rice", "arachis": "groundnut",
           "ipomoea batatas": "sweet_potato", "brassica oleracea": "kale", "vigna unguiculata": "cowpea", "cajanus": "pigeon_pea",
           "theobroma": "cocoa", "gossypium": "cotton", "saccharum": "sugarcane", "helianthus": "sunflower"}
    for s, k in sci.items():
        if s in words:
            return k
    for k, (en, _) in P.PLANTS.items():
        if en.split(" (")[0].lower() in words:
            return k
    return None


def analyse(image_bytes, lang="en", crop=None):
    """-> dict with is_plant, plant (name, key, prob, alternatives, source), health (status, p_sick, symptom probs)."""
    e = engine()
    img = load_image(image_bytes)
    V = e.b.images([img])
    col = colour_stats(img)
    autumn = is_autumn(col)
    pp = float(e.p_plant(V)[0])
    keys, pr = e.name_plant(V, img)
    order = np.argsort(-pr[0])[:4]
    alts = [{"key": keys[i], "name": P.name(keys[i], lang), "prob": round(float(pr[0, i]), 3)} for i in order]
    plant = dict(alts[0], source="clip")
    pn = plantnet(image_bytes, lang)
    if pn and pn[0]["prob"] >= 0.2:
        k = match_key(pn[0]["common"], pn[0]["scientific"])
        label = pn[0]["common"] or pn[0]["scientific"]
        plant = {"key": k, "name": f"{label} ({pn[0]['scientific']})" if pn[0]["common"] else label,
                 "prob": round(pn[0]["prob"], 3), "source": "plantnet"}
        alts = [{"key": match_key(x["common"], x["scientific"]), "name": x["common"] or x["scientific"],
                 "prob": round(x["prob"], 3)} for x in pn]
    if crop:   # the farmer told us the plant
        plant = {"key": crop, "name": P.name(crop, lang), "prob": 1.0, "source": "you"}
        alts = [plant]
    pname = P.PLANTS[plant["key"]][0] if plant.get("key") in P.PLANTS else (plant["name"] if plant["prob"] >= 0.3 else "plant")
    probs, zs = e.health(V, pname)
    if autumn:   # no green left: natural colour change is the most likely story; keep the disease signs as a minority
        probs = 0.4 * probs
        probs[0, GROUPS.index("old_leaf")] += 0.6
    p_sick = float(blend_sick(probs)[0])
    sym = {g: float(probs[0, i]) for i, g in enumerate(GROUPS) if g != "healthy"}
    old = float(probs[0, GROUPS.index("old_leaf")])
    tot = sum(sym.values()) or 1.0
    sym = sorted(((g, v / tot) for g, v in sym.items()), key=lambda x: -x[1])
    alts = [a for a in alts if a["prob"] >= 0.03]
    return {"is_plant": pp >= PLANT_MIN, "p_plant": round(pp, 3), "plant": plant, "alternatives": alts[1:],
            "autumn": autumn, "colour": col,
            "p_sick": round(p_sick, 3), "p_old": round(old, 3), "symptoms": [(g, round(v, 3)) for g, v in sym[:3]],
            "probe": bool(e.probe), "probe_weight": e.probe["w"] if e.probe else 0.0}


def status_of(p_sick, t=0.5):
    return "sick" if p_sick >= t + BAND else "healthy" if p_sick <= t - BAND else "unsure"


def meta():
    j = json.loads(PROBE.read_text()) if PROBE.exists() else {}
    return {"available": available(), "ready": ready(), "probe": bool(j), "loso": j.get("loso"), "naming": j.get("naming"),
            "gate": j.get("gate"), "n_plants": len(P.PLANTS), "namer": NAMER_FLAG.exists(), "plantnet": bool(os.environ.get("PLANTNET_API_KEY", "").strip()),
            "trained_at": j.get("trained_at")}


if __name__ == "__main__":
    if "--warm" in sys.argv:
        t = time.time()
        for f in CACHE.rglob("*.incomplete"):      # leftovers of a stalled download
            f.unlink()
        e = engine()
        e.plant_probs(np.zeros((1, e.b.texts(["x"]).shape[1])))
        e.p_plant(np.zeros((1, e.b.texts(["x"]).shape[1])))
        print(f"CLIP {ARCH}/{TAG} ready in {time.time() - t:.0f}s (weights in outputs/clip)", flush=True)
        if NAMER != "off" and not NAMER_FLAG.exists():
            t = time.time()
            n = Clip(NAMER)
            n.texts(["a photo of Manihot esculenta."])
            NAMER_FLAG.write_text(json.dumps({"model": NAMER, "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
            print(f"species model {NAMER} ready in {time.time() - t:.0f}s", flush=True)
