"""Robustness: what an unconstrained model says about each paid action (shows why monotone constraints are needed)."""
import numpy as np, pandas as pd
import lightgbm as lgb
import model as M
from features import DATA
import actions as A
df = pd.read_csv(DATA / "pets_model_table.csv")
F = M.folds(df)
orig = M.ordinal_params
M.ordinal_params = lambda feats: {k: v for k, v in orig(feats).items() if not k.startswith("monotone")}
rows = []
base = np.zeros(len(df)); new = {a: np.zeros(len(df)) for a in A.PAID_ACTIONS + A.FREE_ACTIONS}
for tr, te in F:
    tr2, es = M.inner_split(tr, df)
    ms = M.fit_ordinal(df, M.FULL, tr2, es)
    t = df.iloc[te]
    base[te] = M.expected_days_from_exceed(M.ordinal_exceed(ms, t[M.FULL]))
    for a in new:
        new[a][te] = M.expected_days_from_exceed(M.ordinal_exceed(ms, A.apply(t, a)[M.FULL]))
for a in new:
    el = A.eligible(df, a)
    rows.append(dict(action=a, unconstrained_mean_days_saved=(base - new[a])[el].mean()))
r = pd.DataFrame(rows); print(r.round(2).to_string(index=False))
r.to_csv(DATA.parent / "outputs" / "action_effects_unconstrained.csv", index=False)
