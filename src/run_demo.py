"""Demo kennel: one real Penang rescuer (65 listings), scored by the fold model that never saw it."""
import pickle
import numpy as np
import pandas as pd
from features import DATA
from model import DAYS, expected_days_from_exceed
from simulate import predict_exceed
from optimize import candidate_table, optimize, rule_vet_first, rule_cheapest_first
import actions as A

OUT = DATA.parent / "outputs"
DEMO = "cccb18b8f8b81862f9a1ebc65d651d22"
BUDGET = 3000

df = pd.read_csv(DATA / "pets_model_table.csv")
models, fold_of = pickle.load(open(DATA.parent / "artifacts" / "fold_models.pkl", "rb"))
costs = A.load_costs(); session_cost = costs[("pro_photography", "any")]
care = {1: costs[("daily_care", "dog")], 2: costs[("daily_care", "cat")]}
idx = np.where(df.RescuerID == DEMO)[0]
k0 = df.iloc[idx].reset_index(drop=True)
ms = models[fold_of[idx[0]]]
q = k0.Quantity.values
base = expected_days_from_exceed(predict_exceed(ms, k0), DAYS)
k, cand = candidate_table(k0, ms, costs)
free = cand[cand.actions == ()].set_index("i").loc[range(len(k))]

chosen, spend, sessions = optimize(cand, BUDGET, session_cost)
plan = pd.DataFrame({
    "PetID": k0.PetID, "Name": k0.Name.fillna("(no name)"), "Species": np.where(k0.Type == 1, "Dog", "Cat"),
    "AgeMonths": k0.Age, "Animals": q, "PredictedDaysNow": base.round(1),
    "FreeQuickWins": [", ".join(A.LABELS[a] for a in A.FREE_ACTIONS if A.eligible(k0.iloc[[i]], a)[0]) for i in range(len(k0))],
    "DaysAfterFreeWins": free.days.values.round(1),
    "PaidActions": [", ".join(A.LABELS[a] for a in acts) for acts in chosen.actions],
    "PaidCostRM": chosen.cost.values.round(0),
    "DaysAfterPlan": chosen.days.values.round(1)})
plan["DaysSaved"] = (plan.PredictedDaysNow - plan.DaysAfterPlan).round(1)
plan.sort_values("DaysSaved", ascending=False).to_csv(OUT / "demo_kennel_plan.csv", index=False)

def totals(rows):
    return (rows.animal_days.sum())
summary = dict(kennel=DEMO, state="Pulau Pinang", listings=len(k0), animals=int(q.sum()), budget=BUDGET,
               baseline_animal_days=float((base * q).sum()), after_free=float(free.animal_days.sum()),
               after_plan=float(chosen.animal_days.sum()), spend=float(spend), sessions=sessions,
               actual_mean_days=float(k0.Days.mean()), predicted_mean_days=float(base.mean()))
print(summary)
mix = {}
for acts, c in zip(chosen.actions, chosen.cost):
    for a in acts: mix[a] = mix.get(a, 0) + 1
print("action counts:", mix)
pd.Series(summary).to_csv(OUT / "demo_kennel_summary.csv")

# Budget curve for the demo kennel
curve = []
for b in range(0, 10001, 500):
    c1, s1, _ = optimize(cand, b, session_cost)
    v, _ = rule_vet_first(k, cand, b, costs)
    ch, _ = rule_cheapest_first(k, cand, b, costs, session_cost)
    curve.append(dict(budget=b, optimizer=summary["baseline_animal_days"] - c1.animal_days.sum(),
                      vet_first=summary["baseline_animal_days"] - v.animal_days.sum(),
                      cheapest_first=summary["baseline_animal_days"] - ch.animal_days.sum()))
pd.DataFrame(curve).to_csv(OUT / "demo_budget_curve.csv", index=False)
print(pd.DataFrame(curve).round(0).to_string(index=False))
