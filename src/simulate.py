"""Counterfactual simulator: expected days for each animal under every combination of actions,
using out-of-fold models so no kennel is scored by a model that saw it."""
import itertools, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import DATA, ARTIFACTS, CLASS_DAYS
from model import (FULL, folds, inner_split, fit_ordinal, ordinal_exceed, expected_days_from_exceed, DAYS)
import actions as A

lgb.basic._log_warning = lambda *a, **k: None


def fold_models(df):
    """Fit the deployed ordinal model once per CV fold; return {fold_index: models} and the fold of each row."""
    F = folds(df)
    fold_of = np.zeros(len(df), dtype=int)
    models = {}
    for i, (tr, te) in enumerate(F):
        tr2, es = inner_split(tr, df)
        models[i] = fit_ordinal(df, FULL, tr2, es)
        fold_of[te] = i
    return models, fold_of


def load_final_models():
    meta = json.load(open(ARTIFACTS / "model_meta.json"))
    return [lgb.Booster(model_file=str(ARTIFACTS / f"ordinal_gt{k}.txt")) for k in range(4)]


def predict_exceed(models, X):
    return ordinal_exceed(models, X[FULL])


def action_effects(df, models, fold_of, actions, days=DAYS):
    """Average expected days saved per eligible listing for each single action (out of fold)."""
    rows = []
    base_ex = np.zeros((len(df), 4))
    for f, ms in models.items():
        idx = fold_of == f
        base_ex[idx] = predict_exceed(ms, df[idx])
    base = expected_days_from_exceed(base_ex, days)
    for a in actions:
        elig = A.eligible(df, a)
        d2 = A.apply(df, a)
        ex = np.zeros((len(df), 4))
        for f, ms in models.items():
            idx = fold_of == f
            ex[idx] = predict_exceed(ms, d2[idx])
        new = expected_days_from_exceed(ex, days)
        saved = (base - new)[elig]
        rows.append(dict(action=a, label=A.LABELS[a], eligible_listings=int(elig.sum()),
                         share_eligible=elig.mean(), mean_days_saved=saved.mean(),
                         median_days_saved=np.median(saved), p90_days_saved=np.percentile(saved, 90)))
    return pd.DataFrame(rows), base


def combo_table(kennel, models, paid_actions, days=DAYS, free_first=True):
    """For one kennel: expected days for every combination of paid actions (after free quick wins)."""
    k = kennel.copy()
    if free_first:
        for a in A.FREE_ACTIONS:
            k = A.apply(k, a)
    combos = list(itertools.product([0, 1], repeat=len(paid_actions)))
    out = np.zeros((len(k), len(combos)))
    for j, c in enumerate(combos):
        d = k
        for on, a in zip(c, paid_actions):
            if on:
                d = A.apply(d, a)
        out[:, j] = expected_days_from_exceed(predict_exceed(models, d), days)
    return k, combos, out
