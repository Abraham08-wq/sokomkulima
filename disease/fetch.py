"""
DISEASE STEP 1 — leaf photos with known diseases.

  PlantVillage (Hughes & Salathé 2015, github.com/spMohanty/PlantVillage-Dataset): ~54,000 leaf photos taken on a plain
     background. We download only the crops we need (git sparse checkout): maize, potato, tomato.
  iCassava 2019 (Mwebaze et al., Makerere University AI Lab / NaCRRI Uganda): 9,430 photos of cassava leaves taken IN FARMERS'
     FIELDS in Uganda: cassava mosaic, brown streak, bacterial blight, green mite, healthy.

    python disease/fetch.py            # DISEASE_CROPS=maize,potato,tomato,cassava (default)
Writes data/disease/index.csv: path, label, crop, source, split.
"""
from pathlib import Path
import csv, os, random, subprocess, sys, urllib.request, zipfile

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "disease"
PV_REPO = "https://github.com/spMohanty/PlantVillage-Dataset"
CASSAVA_URL = os.environ.get("CASSAVA_URL", "https://storage.googleapis.com/emcassavadata/cassavaleafdata.zip")

PV = {  # PlantVillage folder -> our label
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "maize_gray_leaf_spot",
    "Corn_(maize)___Common_rust_": "maize_common_rust",
    "Corn_(maize)___Northern_Leaf_Blight": "maize_northern_leaf_blight",
    "Corn_(maize)___healthy": "maize_healthy",
    "Potato___Early_blight": "potato_early_blight", "Potato___Late_blight": "potato_late_blight",
    "Potato___healthy": "potato_healthy",
    "Tomato___Bacterial_spot": "tomato_bacterial_spot", "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Late_blight": "tomato_late_blight", "Tomato___Leaf_Mold": "tomato_leaf_mold",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato_spider_mites", "Tomato___Target_Spot": "tomato_target_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl_virus",
    "Tomato___Tomato_mosaic_virus": "tomato_mosaic_virus", "Tomato___healthy": "tomato_healthy",
}
CASSAVA = {"cbb": "cassava_bacterial_blight", "cbsd": "cassava_brown_streak", "cgm": "cassava_green_mite",
           "cmd": "cassava_mosaic", "healthy": "cassava_healthy"}


def run(cmd, cwd=None, stdin=None):
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, input=stdin, text=True, check=True)


def plantvillage(crops):
    folders = [f for f, lab in PV.items() if lab.split("_")[0] in crops]
    repo = D / "plantvillage"
    if not (repo / ".git").exists():
        D.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", PV_REPO, str(repo)])
    run(["git", "sparse-checkout", "set", "--no-cone", "--stdin"], cwd=repo,
        stdin="".join(f"/raw/color/{f}/\n" for f in folders))
    run(["git", "checkout", "-q"], cwd=repo)
    rows = []
    rng = random.Random(42)
    for f in folders:
        files = sorted((repo / "raw" / "color" / f).glob("*"))
        rng.shuffle(files)
        n = len(files)
        for i, p in enumerate(files):
            split = "test" if i < 0.1 * n else "val" if i < 0.2 * n else "train"
            rows.append((str(p.relative_to(D)), PV[f], PV[f].split("_")[0], "plantvillage", split))
    print(f"  PlantVillage: {len(rows):,} photos in {len(folders)} classes")
    return rows


def cassava():
    z, out = D / "cassava.zip", D / "cassava"
    if not out.exists():
        if not z.exists():
            print(f"  downloading iCassava (large, ~1 GB) from {CASSAVA_URL} ...", flush=True)
            tmp = z.with_suffix(".part")
            urllib.request.urlretrieve(CASSAVA_URL, tmp); tmp.rename(z)
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
    base = next(out.rglob("train")).parent                      # .../cassavaleafdata
    rows = []
    for split, name in (("train", "train"), ("validation", "val"), ("test", "test")):
        for k, lab in CASSAVA.items():
            for p in sorted((base / split / k).glob("*.jpg")):
                rows.append((str(p.relative_to(D)), lab, "cassava", "icassava", name))
    print(f"  iCassava: {len(rows):,} field photos in {len(CASSAVA)} classes")
    return rows


def main():
    crops = [c.strip() for c in os.environ.get("DISEASE_CROPS", "maize,potato,tomato,cassava").split(",") if c.strip()]
    rows = []
    pv_crops = [c for c in crops if c in ("maize", "potato", "tomato")]
    if pv_crops:
        rows += plantvillage(pv_crops)
    if "cassava" in crops:
        try:
            rows += cassava()
        except Exception as e:
            print(f"  cassava photos unavailable ({type(e).__name__}: {e}); continuing without cassava")
    if not rows:
        sys.exit("no photos downloaded")
    with open(D / "index.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "label", "crop", "source", "split"]); w.writerows(rows)
    print(f"index: {len(rows):,} photos -> data/disease/index.csv")


if __name__ == "__main__":
    main()
