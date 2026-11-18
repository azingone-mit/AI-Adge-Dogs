import numpy as np, pandas as pd, pickle
from features import DATA
from model import DAYS
from simulate import fold_models, action_effects
import actions as A
df = pd.read_csv(DATA / "pets_model_table.csv")
models, fold_of = fold_models(df)
pickle.dump((models, fold_of), open(DATA.parent / "artifacts" / "fold_models.pkl", "wb"))
eff, base = action_effects(df, models, fold_of, A.FREE_ACTIONS + A.PAID_ACTIONS)
pd.set_option("display.width", 200)
print(eff.round(2).to_string(index=False))
eff.to_csv(DATA.parent / "outputs" / "action_effects.csv", index=False)
for ceil in [100, 150]:
    d = DAYS.copy(); d[4] = ceil
    e2, _ = action_effects(df, models, fold_of, A.FREE_ACTIONS + A.PAID_ACTIONS, d)
    e2.to_csv(DATA.parent / "outputs" / f"action_effects_ceiling{ceil}.csv", index=False)
    print(ceil, e2[["action", "mean_days_saved"]].round(2).values.tolist())
