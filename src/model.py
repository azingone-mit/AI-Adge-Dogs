"""Train and benchmark days-to-adoption models with RescuerID-grouped cross-validation."""
import json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr
from features import (DATA, ARTIFACTS, CLASS_DAYS, PROFILE, TECH_PHOTO, CLIP_PHOTO, LISTING,
                      CATEGORICAL, MONOTONE)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DAYS = np.array([CLASS_DAYS[k] for k in range(5)])
FULL = PROFILE + CLIP_PHOTO + LISTING
SEED = 42

LGB_PARAMS = dict(learning_rate=0.03, num_leaves=31, min_child_samples=40, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED)


def folds(df, k=5):
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
                .split(df, df.AdoptionSpeed, groups=df.RescuerID))


def inner_split(tr_idx, df):
    """Hold out 15% of training rescuers for early stopping, so the test fold is never touched."""
    rescuers = df.RescuerID.iloc[tr_idx].unique()
    rng = np.random.default_rng(SEED)
    es = set(rng.choice(rescuers, int(0.15 * len(rescuers)), replace=False))
    mask = df.RescuerID.iloc[tr_idx].isin(es).values
    return tr_idx[~mask], tr_idx[mask]


def cat_cols(feats):
    return [c for c in CATEGORICAL if c in feats]


def fit_multiclass(df, feats, tr, es):
    p = dict(LGB_PARAMS, objective="multiclass", num_class=5)
    dtr = lgb.Dataset(df.iloc[tr][feats], df.AdoptionSpeed.iloc[tr], categorical_feature=cat_cols(feats))
    des = lgb.Dataset(df.iloc[es][feats], df.AdoptionSpeed.iloc[es], reference=dtr)
    return lgb.train(p, dtr, 3000, valid_sets=[des], callbacks=[lgb.early_stopping(100, verbose=False)])


def ordinal_params(feats):
    mono = [MONOTONE.get(f, 0) for f in feats]
    return dict(LGB_PARAMS, objective="binary", monotone_constraints=mono, monotone_constraints_method="advanced")


def fit_ordinal(df, feats, tr, es):
    """Four binary models for P(AdoptionSpeed > k), k = 0..3, with monotone constraints on actionable features."""
    models = []
    for k in range(4):
        y = (df.AdoptionSpeed > k).astype(int)
        dtr = lgb.Dataset(df.iloc[tr][feats], y.iloc[tr], categorical_feature=cat_cols(feats))
        des = lgb.Dataset(df.iloc[es][feats], y.iloc[es], reference=dtr)
        models.append(lgb.train(ordinal_params(feats), dtr, 3000, valid_sets=[des],
                                callbacks=[lgb.early_stopping(100, verbose=False)]))
    return models


def ordinal_exceed(models, X):
    p = np.column_stack([m.predict(X) for m in models])
    return np.minimum.accumulate(p, axis=1)  # P(Y>0) >= P(Y>1) >= ...


def exceed_to_probs(ex):
    cum = np.column_stack([np.ones(len(ex)), ex, np.zeros(len(ex))])
    return np.clip(cum[:, :-1] - cum[:, 1:], 1e-6, 1)


def expected_days_from_exceed(ex, days=DAYS):
    return days[0] + (ex * np.diff(days)).sum(1)


def distribution_match(score, y_train):
    """Turn a continuous score into classes that match the training class mix (standard approach for this competition)."""
    cuts = np.cumsum(np.bincount(y_train, minlength=5) / len(y_train))[:-1]
    thr = np.quantile(score, cuts)
    return np.digitize(score, thr)


def metrics(y, probs, name):
    probs = probs / probs.sum(1, keepdims=True)
    exp_days = probs @ DAYS
    pred_cls = probs.argmax(1)
    dm_cls = distribution_match(probs @ np.arange(5), y)
    return dict(model=name,
                accuracy=accuracy_score(y, pred_cls),
                qwk=cohen_kappa_score(y, dm_cls, weights="quadratic"),
                log_loss=log_loss(y, probs, labels=range(5)),
                mae_days=np.mean(np.abs(exp_days - DAYS[y])),
                spearman=spearmanr(exp_days, DAYS[y]).correlation)


def run_cv(df):
    y = df.AdoptionSpeed.values
    F = folds(df)
    oof = {}

    # 1. Naive baseline: every animal gets the overall class mix
    oof["Baseline: class average"] = np.zeros((len(df), 5))
    for tr, te in F:
        oof["Baseline: class average"][te] = np.bincount(y[tr], minlength=5) / len(tr)

    # 2. Business rule: young animals go fast (class mix by age band)
    bands = pd.cut(df.Age, [-1, 2, 6, 12, 36, 1000]).cat.codes.values
    oof["Rule: class mix by age band"] = np.zeros((len(df), 5))
    for tr, te in F:
        for b in np.unique(bands):
            mix = np.bincount(y[tr][bands[tr] == b], minlength=5) + 1
            oof["Rule: class mix by age band"][te[bands[te] == b]] = mix / mix.sum()

    # 3. Multinomial logistic regression on profile fields
    num = [c for c in PROFILE if c not in CATEGORICAL]
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL)])
    oof["Logistic regression (profile)"] = np.zeros((len(df), 5))
    for tr, te in F:
        lr = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))
        lr.fit(df.iloc[tr][PROFILE], y[tr])
        oof["Logistic regression (profile)"][te] = lr.predict_proba(df.iloc[te][PROFILE])

    # 4 to 7. LightGBM ablations
    sets = {"LightGBM: profile only": PROFILE,
            "LightGBM: profile + pixel photo metrics": PROFILE + TECH_PHOTO,
            "LightGBM: profile + CLIP photo": PROFILE + CLIP_PHOTO,
            "LightGBM: full (profile + CLIP photo + listing)": FULL}
    best_iters = []
    for name, feats in sets.items():
        oof[name] = np.zeros((len(df), 5))
        for tr, te in F:
            tr2, es = inner_split(tr, df)
            m = fit_multiclass(df, feats, tr2, es)
            oof[name][te] = m.predict(df.iloc[te][feats], num_iteration=m.best_iteration)

    # 8. Deployed model: ordinal with monotone constraints (full features)
    name = "Deployed: ordinal LightGBM, monotone (full)"
    oof[name] = np.zeros((len(df), 5))
    oof_exceed = np.zeros((len(df), 4))
    for tr, te in F:
        tr2, es = inner_split(tr, df)
        ms = fit_ordinal(df, FULL, tr2, es)
        best_iters.append([m.best_iteration for m in ms])
        ex = ordinal_exceed(ms, df.iloc[te][FULL])
        oof_exceed[te] = ex
        oof[name][te] = exceed_to_probs(ex)

    res = pd.DataFrame([metrics(y, p, n) for n, p in oof.items()])
    return res, oof, oof_exceed, np.array(best_iters)


def train_final(df, best_iters):
    """Refit the deployed ordinal model on all data with the average CV iteration counts."""
    iters = np.maximum(50, best_iters.mean(0).astype(int))
    models = []
    for k in range(4):
        yk = (df.AdoptionSpeed > k).astype(int)
        d = lgb.Dataset(df[FULL], yk, categorical_feature=cat_cols(FULL))
        models.append(lgb.train(ordinal_params(FULL), d, int(iters[k])))
    ARTIFACTS.mkdir(exist_ok=True)
    for k, m in enumerate(models):
        m.save_model(str(ARTIFACTS / f"ordinal_gt{k}.txt"))
    json.dump({"features": FULL, "iterations": iters.tolist(), "class_days": CLASS_DAYS},
              open(ARTIFACTS / "model_meta.json", "w"), indent=2)
    return models


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "pets_model_table.csv")
    res, oof, oof_exceed, best_iters = run_cv(df)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    res.to_csv(OUT / "benchmark_results.csv", index=False)
    np.save(OUT / "oof_exceed_deployed.npy", oof_exceed)
    train_final(df, best_iters)
    print("best iters per threshold:", best_iters.mean(0).round())
