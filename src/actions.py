"""Action menu: what a Malaysian shelter can do, which profile fields each action changes, and what it costs (RM)."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COSTS = ROOT / "costs" / "malaysia_costs_latest.csv"

PHOTO_TARGET_APPEAL = 85      # assumption: a professional shoot lifts the main photo to the 85th percentile of appeal
PHOTO_TARGET_COUNT = 6        # and delivers at least 6 photos
PHOTOS_PER_SESSION = 15       # assumption: one session covers up to 15 listings
STAFF_PHOTO_COUNT = 5         # staff phone photos: bring every listing to at least 5 photos
BIO_MIN_WORDS = 60            # AI-written bio: at least 60 words
MIN_STERILIZE_AGE = 6         # months

# Free "quick wins" use staff time only; paid actions go through the budget optimizer
FREE_ACTIONS = ["more_photos", "add_video", "give_name", "write_bio"]
PAID_ACTIONS = ["vaccinate", "deworm", "sterilize", "treat_minor_injury", "pro_photos", "waive_fee"]

LABELS = {"more_photos": "Staff phone photos (to 5)", "add_video": "Short phone video", "give_name": "Give a name",
          "write_bio": "AI-written bio (60+ words)", "vaccinate": "Vaccinate", "deworm": "Deworm",
          "sterilize": "Spay / neuter", "treat_minor_injury": "Treat minor injury", "pro_photos": "Professional photos",
          "waive_fee": "Waive adoption fee"}


def load_costs(path=DEFAULT_COSTS):
    c = pd.read_csv(path)
    return {(r.action, r.animal): r.price_mid_rm for r in c.itertuples()}


def eligible(df: pd.DataFrame, action: str) -> np.ndarray:
    if action == "more_photos":
        return (df.PhotoAmt < STAFF_PHOTO_COUNT).values
    if action == "add_video":
        return (df.VideoAmt == 0).values
    if action == "give_name":
        return (df.HasName == 0).values
    if action == "write_bio":
        return (df.DescWords < BIO_MIN_WORDS).values
    if action == "vaccinate":
        return (df.VaccinatedOrd < 2).values
    if action == "deworm":
        return (df.DewormedOrd < 2).values
    if action == "sterilize":
        return ((df.SterilizedOrd < 2) & (df.Age >= MIN_STERILIZE_AGE)).values
    if action == "treat_minor_injury":
        return (df.Health == 2).values
    if action == "pro_photos":
        return ((df.PhotoAppeal.fillna(0) < PHOTO_TARGET_APPEAL) | (df.PhotoAmt < PHOTO_TARGET_COUNT)).values
    if action == "waive_fee":
        return (df.Fee > 0).values
    raise KeyError(action)


def apply(df: pd.DataFrame, action: str) -> pd.DataFrame:
    """Return a copy of the profiles as they would look after the action (only on eligible rows)."""
    d = df.copy()
    m = eligible(df, action)
    if action == "more_photos":
        d.loc[m, "PhotoAmt"] = STAFF_PHOTO_COUNT
    elif action == "add_video":
        d.loc[m, "VideoAmt"] = 1
    elif action == "give_name":
        d.loc[m, "HasName"] = 1
    elif action == "write_bio":
        d.loc[m, "DescWords"] = BIO_MIN_WORDS
    elif action == "vaccinate":
        d.loc[m, "VaccinatedOrd"] = 2
    elif action == "deworm":
        d.loc[m, "DewormedOrd"] = 2
    elif action == "sterilize":
        d.loc[m, "SterilizedOrd"] = 2
    elif action == "treat_minor_injury":
        d.loc[m, "Health"] = 1
    elif action == "pro_photos":
        d.loc[m, "PhotoAppeal"] = np.maximum(d.loc[m, "PhotoAppeal"].fillna(0), PHOTO_TARGET_APPEAL)
        d.loc[m, "PhotoAmt"] = np.maximum(d.loc[m, "PhotoAmt"], PHOTO_TARGET_COUNT)
    elif action == "waive_fee":
        d.loc[m, "Fee"] = 0
    return d


def unit_cost(df: pd.DataFrame, action: str, costs: dict) -> np.ndarray:
    """RM cost per listing (per-animal prices times Quantity). Professional photos are handled as a session cost."""
    q = df.Quantity.values
    sp = np.where(df.Type == 1, "dog", "cat")
    if action in FREE_ACTIONS:
        return np.zeros(len(df))
    if action == "vaccinate":
        return np.array([costs[("vaccinate", s)] for s in sp]) * q
    if action == "deworm":
        return np.array([costs[("deworm", s)] for s in sp]) * q
    if action == "sterilize":
        # Gender 1 male, 2 female, 3 mixed group: price mixed groups at the average of male and female
        out = []
        for s, g in zip(sp, df.Gender.values):
            male, female = costs[("sterilize", f"{s}_male")], costs[("sterilize", f"{s}_female")]
            out.append(male if g == 1 else female if g == 2 else (male + female) / 2)
        return np.array(out) * q
    if action == "treat_minor_injury":
        return np.full(len(df), costs[("treat_minor_injury", "any")]) * q
    if action == "pro_photos":
        return np.zeros(len(df))  # paid per session, see PHOTOS_PER_SESSION
    if action == "waive_fee":
        return df.Fee.values.astype(float)  # lost adoption revenue
    raise KeyError(action)
