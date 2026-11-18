"""Budget optimizer: choose, for each listing, which paid actions to take so total expected animal-days is lowest."""
import numpy as np
import pandas as pd
import pulp
import actions as A
from simulate import combo_table


def candidate_table(kennel, models, costs, paid=A.PAID_ACTIONS):
    """Rows = (listing, action combination) with total RM cost and expected animal-days. Free quick wins applied first."""
    k, combos, days = combo_table(kennel, models, paid)
    unit = {a: A.unit_cost(k, a, costs) for a in paid}
    elig = {a: A.eligible(k, a) for a in paid}
    rows = []
    for i in range(len(k)):
        q = k.Quantity.iloc[i]
        for j, c in enumerate(combos):
            active = [a for on, a in zip(c, paid) if on]
            if any(not elig[a][i] for a in active):
                continue  # skip combinations that include an action this animal doesn't need
            rows.append(dict(i=i, combo=j, actions=tuple(active), cost=sum(unit[a][i] for a in active),
                             pro=int("pro_photos" in active), days=days[i, j], animal_days=days[i, j] * q))
    return k, pd.DataFrame(rows)


def optimize(cand, budget, session_cost, per_session=A.PHOTOS_PER_SESSION, time_limit=60):
    prob = pulp.LpProblem("kennel", pulp.LpMinimize)
    x = {r: pulp.LpVariable(f"x{r}", cat="Binary") for r in cand.index}
    s = pulp.LpVariable("sessions", lowBound=0, cat="Integer")
    prob += pulp.lpSum(cand.animal_days[r] * x[r] for r in cand.index)
    for i, g in cand.groupby("i"):
        prob += pulp.lpSum(x[r] for r in g.index) == 1
    pro_rows = cand.index[cand.pro == 1]
    prob += per_session * s >= pulp.lpSum(x[r] for r in pro_rows)
    prob += pulp.lpSum(cand.cost[r] * x[r] for r in cand.index) + session_cost * s <= budget
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    chosen = cand.loc[[r for r in cand.index if x[r].value() > 0.5]].sort_values("i")
    spend = chosen.cost.sum() + session_cost * (s.value() or 0)
    return chosen, spend, int(round(s.value() or 0))


def _evaluate(cand, picks):
    """picks: dict listing -> tuple of actions. Returns rows of cand matching."""
    key = {(r.i, r.actions): idx for idx, r in cand.iterrows()}
    return cand.loc[[key[(i, tuple(sorted(a, key=A.PAID_ACTIONS.index)))] for i, a in picks.items()]]


def rule_vet_first(k, cand, budget, costs, seed=0):
    """Common practice: vaccinate everyone, then deworm, then sterilize, in random animal order, until money runs out."""
    picks = {i: [] for i in range(len(k))}
    order = np.random.default_rng(seed).permutation(len(k))
    spent = 0.0
    for a in ["vaccinate", "deworm", "sterilize"]:
        unit, el = A.unit_cost(k, a, costs), A.eligible(k, a)
        for i in order:
            if el[i] and spent + unit[i] <= budget:
                picks[i].append(a); spent += unit[i]
    return _evaluate(cand, picks), spent


def rule_cheapest_first(k, cand, budget, costs, session_cost):
    """Do the cheapest available action across all animals first, then the next cheapest, until money runs out."""
    picks = {i: [] for i in range(len(k))}
    items = []
    for a in A.PAID_ACTIONS:
        if a == "pro_photos":
            continue
        unit, el = A.unit_cost(k, a, costs), A.eligible(k, a)
        items += [(unit[i], i, a) for i in range(len(k)) if el[i]]
    items.sort()
    spent = 0.0
    for c, i, a in items:
        if spent + c <= budget:
            picks[i].append(a); spent += c
    # leftover money: photographer sessions for the listings with the weakest photos
    el = A.eligible(k, "pro_photos")
    weakest = [i for i in np.argsort(k.PhotoAppeal.fillna(0).values) if el[i]]
    while spent + session_cost <= budget and weakest:
        for i in weakest[:A.PHOTOS_PER_SESSION]:
            picks[i].append("pro_photos")
        weakest = weakest[A.PHOTOS_PER_SESSION:]; spent += session_cost
    return _evaluate(cand, picks), spent
