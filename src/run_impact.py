"""Impact study across the 50 largest real rescuers (30+ listings each), scored out of fold."""
import pickle, time
import numpy as np
import pandas as pd
from features import DATA
from model import DAYS, expected_days_from_exceed
from simulate import predict_exceed
from optimize import candidate_table, optimize, rule_vet_first, rule_cheapest_first
import actions as A

OUT = DATA.parent / "outputs"
import sys
BUDGETS_PER_LISTING = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 5, 10, 15, 25, 50, 100, 200]

df = pd.read_csv(DATA / "pets_model_table.csv")
models, fold_of = pickle.load(open(DATA.parent / "artifacts" / "fold_models.pkl", "rb"))
costs = A.load_costs()
session_cost = costs[("pro_photography", "any")]
care = {1: costs[("daily_care", "dog")], 2: costs[("daily_care", "cat")]}

vc = df.RescuerID.value_counts()
kennels = vc[vc >= 30].index
rows = []
t0 = time.time()
for n, rid in enumerate(kennels):
    idx = np.where(df.RescuerID == rid)[0]
    k0 = df.iloc[idx].reset_index(drop=True)
    ms = models[fold_of[idx[0]]]
    q = k0.Quantity.values
    rate = np.array([care[t] for t in k0.Type])
    base_days = expected_days_from_exceed(predict_exceed(ms, k0), DAYS)
    k, cand = candidate_table(k0, ms, costs)
    free = cand[cand.actions == ()].set_index("i").loc[range(len(k))]
    base_ad = (base_days * q).sum()
    for b in BUDGETS_PER_LISTING:
        budget = b * len(k)
        res = {"kennel": rid, "listings": len(k), "animals": int(q.sum()), "budget_per_listing": b, "budget": budget,
               "baseline_animal_days": base_ad, "free_wins_animal_days": free.animal_days.sum()}
        chosen, spend, sessions = optimize(cand, budget, session_cost)
        res.update(opt_animal_days=chosen.animal_days.sum(), opt_spend=spend, opt_sessions=sessions)
        res["opt_rm_care_saved"] = ((base_days - chosen.days.values) * q * rate).sum()
        vet, s1 = rule_vet_first(k, cand, budget, costs)
        res.update(vet_first_animal_days=vet.animal_days.sum(), vet_first_spend=s1)
        ch, s2 = rule_cheapest_first(k, cand, budget, costs, session_cost)
        res.update(cheapest_first_animal_days=ch.animal_days.sum(), cheapest_first_spend=s2)
        rows.append(res)
    print(n, rid[:8], len(k), round(time.time() - t0), flush=True)

r = pd.DataFrame(rows)
r.to_csv(OUT / (sys.argv[2] if len(sys.argv) > 2 else "impact_by_kennel.csv"), index=False)

# Portfolio summary across kennels
g = r.groupby("budget_per_listing")[["animals", "baseline_animal_days", "free_wins_animal_days", "opt_animal_days",
                                     "vet_first_animal_days", "cheapest_first_animal_days", "opt_spend",
                                     "opt_rm_care_saved"]].sum()
for c in ["free_wins", "opt", "vet_first", "cheapest_first"]:
    g[c + "_saved"] = g.baseline_animal_days - g[c + "_animal_days"]
g["paid_extra_days"] = g.opt_saved - g.free_wins_saved
g["rm_per_extra_day"] = g.opt_spend / g.paid_extra_days
g["paid_care_saved"] = g.opt_rm_care_saved - g.loc[0, "opt_rm_care_saved"]
g.to_csv(OUT / "impact_summary.csv")
print(g.round(1).to_string())
