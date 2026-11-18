"""Step 0 (run once, needs the Kaggle PetFinder images): photo features for every training image.

    python src/extract_photos.py --kaggle_dir /path/to/petfinder-adoption-prediction

Writes data/image_features.csv, data/sentiment_features.csv, data/metadata_features.csv,
data/clip_main_photos.npz and copies train.csv and the label files into data/.
Pixel features take about 4 minutes on 4 CPU cores; CLIP embeddings of the 14,652 main photos about 10 minutes.
"""
import argparse, glob, json, os, shutil
import numpy as np, pandas as pd
from multiprocessing import Pool
from pathlib import Path
from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data"
CLIP_MODEL, CLIP_WEIGHTS = "ViT-B-32", "laion2b_s34b_b79k"
PROMPTS = {"pos": ["a professional, well-lit close-up portrait photo of a cute pet looking at the camera",
                   "a high quality bright photo of a happy dog", "a high quality bright photo of a happy cat",
                   "an adorable pet photographed by a professional photographer"],
           "neg": ["a dark, blurry, low quality photo of an animal", "a photo of a dog or cat behind cage bars in a shelter",
                   "a cluttered amateur snapshot of an animal far away", "a poster or screenshot with lots of text"]}


def img_feats(path):
    """Pixel metrics on a copy downsized to at most 512 px (keeps original width and height)."""
    try:
        with Image.open(path) as img:
            w, h = img.size
            img.draft("RGB", (512, 512))
            im = img.convert("RGB")
            if max(im.size) > 512:
                im.thumbnail((512, 512))
            arr = np.asarray(im, dtype=np.float32)
        gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gy, gx = np.gradient(gray)
        lap = gray[1:-1, 2:] + gray[1:-1, :-2] + gray[2:, 1:-1] + gray[:-2, 1:-1] - 4 * gray[1:-1, 1:-1]
        R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
        rg, yb = np.abs(R - G), np.abs(0.5 * (R + G) - B)
        colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        pet_id, idx = os.path.basename(path)[:-4].rsplit("-", 1)
        return [pet_id, int(idx), w, h, float(gray.mean()), float(gray.std()),
                float(np.sqrt(gx ** 2 + gy ** 2).var()), float(lap.var()), float(colorful)]
    except Exception:
        return None


def clip_embeddings(paths, batch=64):
    import torch, open_clip
    m, _, pre = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_WEIGHTS)
    tok = open_clip.get_tokenizer(CLIP_MODEL)
    m.eval(); embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
            f = m.encode_image(x); embs.append((f / f.norm(dim=-1, keepdim=True)).numpy().astype(np.float16))
        T = {k: (lambda f: (f / f.norm(dim=-1, keepdim=True)).numpy())(m.encode_text(tok(v))) for k, v in PROMPTS.items()}
    return np.concatenate(embs), T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kaggle_dir", required=True); ap.add_argument("--skip_clip", action="store_true")
    k = Path(ap.parse_args().kaggle_dir); args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    shutil.copy(k / "train" / "train.csv", DATA / "train.csv")
    for f in ["breed_labels.csv", "color_labels.csv", "state_labels.csv"]:
        shutil.copy(k / f, DATA / f)
    paths = sorted(glob.glob(str(k / "train_images" / "*.jpg")))
    with Pool() as p:
        rows = [r for r in p.imap_unordered(img_feats, paths, chunksize=64) if r]
    pd.DataFrame(rows, columns=["PetID", "ImageIdx", "Width", "Height", "Brightness", "Contrast", "Sharpness_Grad",
                                "Sharpness_Lap", "Colorfulness"]).to_csv(DATA / "image_features.csv", index=False)
    sent = []
    for f in glob.glob(str(k / "train_sentiment" / "*.json")):
        d = json.load(open(f)); ds = d.get("documentSentiment", {})
        sent.append([os.path.basename(f)[:-5], ds.get("score"), ds.get("magnitude"), len(d.get("sentences", [])), d.get("language")])
    pd.DataFrame(sent, columns=["PetID", "SentScore", "SentMagnitude", "SentSentences", "Language"]).to_csv(DATA / "sentiment_features.csv", index=False)
    meta = []
    for f in glob.glob(str(k / "train_metadata" / "*-1.json")):
        d = json.load(open(f)); labels = d.get("labelAnnotations", [])
        crop = d.get("cropHintsAnnotation", {}).get("cropHints") or [{}]
        meta.append([os.path.basename(f)[:-7], labels[0]["description"] if labels else None,
                     labels[0]["score"] if labels else None, crop[0].get("confidence"), crop[0].get("importanceFraction")])
    pd.DataFrame(meta, columns=["PetID", "TopLabel", "TopLabelScore", "CropConfidence", "CropImportance"]).to_csv(DATA / "metadata_features.csv", index=False)
    if not args.skip_clip:
        main_paths = sorted(glob.glob(str(k / "train_images" / "*-1.jpg")))
        emb, T = clip_embeddings(main_paths)
        ids = np.array([os.path.basename(p)[:-6] for p in main_paths])
        np.savez_compressed(DATA / "clip_main_photos.npz", emb=emb, ids=ids, text_pos=T["pos"], text_neg=T["neg"], prompts=json.dumps(PROMPTS))
    print("done")


if __name__ == "__main__":
    main()
