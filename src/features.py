"""Build the pet-level modeling table: profile fields, photo features (technical and CLIP), and listing text features."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DATA, ARTIFACTS = ROOT / "data", ROOT / "artifacts"

# Representative days for each AdoptionSpeed class. Class 4 means "not adopted after 100 days";
# 120 is an assumption that the sensitivity analysis varies (100, 150).
CLASS_DAYS = {0: 0.0, 1: 4.0, 2: 19.0, 3: 60.0, 4: 120.0}

# Health status recoding so that "higher = better" for the vet actions (1 yes, 3 not sure, 2 no)
YES_UNSURE_NO = {1: 2, 3: 1, 2: 0}

# ---------- technical photo score (the team's original idea, kept as a benchmark) ----------
TECH_WEIGHTS = {"Sharpness_Lap": 0.30, "Contrast": 0.20, "Colorfulness": 0.20, "BrightBalance": 0.15, "Pixels": 0.15}


def _tech_table(img):
    img = img.copy()
    img["BrightBalance"] = -np.abs(img["Brightness"] - 127.5)
    img["Pixels"] = img["Width"] * img["Height"]
    img["Sharpness_Lap"] = np.log1p(img["Sharpness_Lap"])
    return img


def tech_score(img, percentiles):
    img = _tech_table(img)
    s = np.zeros(len(img))
    for m, w in TECH_WEIGHTS.items():
        s += w * np.interp(img[m].values, percentiles[m], np.arange(101))
    return pd.Series(s, index=img.index)


# ---------- CLIP photo appeal score (the actionable photo feature) ----------
def clip_quality(emb, text_pos, text_neg):
    """Zero-shot photo quality: mean similarity to 'professional, well-lit pet portrait' prompts
    minus mean similarity to 'dark, blurry, caged, cluttered' prompts."""
    return (emb @ text_pos.T).mean(1) - (emb @ text_neg.T).mean(1)


def to_percentile(x, ref_percentiles):
    return np.interp(x, ref_percentiles, np.arange(101))


def build_table(n_pcs: int = 10, save: bool = True) -> pd.DataFrame:
    t = pd.read_csv(DATA / "train.csv")
    img = pd.read_csv(DATA / "image_features.csv")
    sent = pd.read_csv(DATA / "sentiment_features.csv")
    meta = pd.read_csv(DATA / "metadata_features.csv")
    z = np.load(DATA / "clip_main_photos.npz")
    emb = z["emb"].astype(np.float32)

    # Technical score, scaled with percentiles fitted once on all training photos
    tech_pct = {m: np.percentile(_tech_table(img)[m].dropna(), np.arange(101)).tolist() for m in TECH_WEIGHTS}
    img["TechPhotoScore"] = tech_score(img, tech_pct)

    # CLIP appeal, scaled to a 0 to 100 percentile fitted once on the training photos
    q = clip_quality(emb, z["text_pos"], z["text_neg"])
    q_pct = np.percentile(q, np.arange(101))
    pca = PCA(n_components=n_pcs, random_state=0).fit(emb)
    pcs = pca.transform(emb)
    clip = pd.DataFrame(pcs, columns=[f"ClipPC{i+1}" for i in range(n_pcs)], index=z["ids"])
    clip["PhotoAppeal"] = to_percentile(q, q_pct)
    clip["PhotoAppealRaw"] = q

    if save:
        ARTIFACTS.mkdir(exist_ok=True)
        json.dump({"tech_weights": TECH_WEIGHTS, "tech_percentiles": tech_pct,
                   "appeal_percentiles": q_pct.tolist(), "prompts": json.loads(str(z["prompts"]))},
                  open(ARTIFACTS / "photo_scalers.json", "w"))
        np.savez(ARTIFACTS / "clip_pca.npz", components=pca.components_, mean=pca.mean_)

    main = img[img.ImageIdx == 1].set_index("PetID")
    df = t.set_index("PetID")
    df["TechPhotoScore"] = main["TechPhotoScore"]
    for c in ["Sharpness_Lap", "Contrast", "Colorfulness", "Brightness"]:
        df[f"Main_{c}"] = main[c]
    df = df.join(clip)
    df = df.join(sent.set_index("PetID")[["SentScore", "SentMagnitude"]])
    df = df.join(meta.set_index("PetID")[["TopLabelScore", "CropConfidence"]])

    for c in ["Vaccinated", "Dewormed", "Sterilized"]:
        df[c + "Ord"] = df[c].map(YES_UNSURE_NO)
    name = df["Name"].fillna("").str.strip().str.lower()
    df["HasName"] = (~name.isin(["", "no name", "no name yet", "unknown", "nameless", "none", "noname"])).astype(int)
    df["DescWords"] = df["Description"].fillna("").str.split().str.len().fillna(0)
    df["RescuerListings"] = df.groupby("RescuerID")["Type"].transform("size")
    df["IsMixedBreed"] = ((df.Breed1 == 307) | (df.Breed2 == 307)).astype(int)
    df["Days"] = df["AdoptionSpeed"].map(CLASS_DAYS)
    return df.reset_index()


PROFILE = ["Type", "Age", "Breed1", "Breed2", "IsMixedBreed", "Gender", "Color1", "Color2", "Color3",
           "MaturitySize", "FurLength", "VaccinatedOrd", "DewormedOrd", "SterilizedOrd", "Health", "Quantity",
           "Fee", "State", "RescuerListings"]
TECH_PHOTO = ["TechPhotoScore", "Main_Sharpness_Lap", "Main_Contrast", "Main_Colorfulness", "Main_Brightness", "PhotoAmt"]
CLIP_PHOTO = ["PhotoAppeal", "PhotoAmt"] + [f"ClipPC{i+1}" for i in range(10)]
LISTING = ["VideoAmt", "HasName", "DescWords", "SentScore", "SentMagnitude"]
CATEGORICAL = ["Breed1", "Breed2", "State", "Color1", "Color2", "Color3", "Gender"]

# Direction each actionable feature is allowed to move P(slow adoption): -1 = more of it can only help, +1 = can only hurt
MONOTONE = {"VaccinatedOrd": -1, "DewormedOrd": -1, "SterilizedOrd": -1, "Health": 1, "Fee": 1,
            "PhotoAppeal": -1, "PhotoAmt": -1, "VideoAmt": -1, "HasName": -1, "DescWords": -1}

if __name__ == "__main__":
    df = build_table()
    df.to_csv(DATA / "pets_model_table.csv", index=False)
    print(df.shape)
