# Kennel Optimizer: predicting days to adoption and spending a shelter's budget where it matters

**Team:** Andrew Zingone and Michael Naber

**15.071 The AI Edge, Project 1 (Predictive AI).** Audience: a data scientist at a Malaysian animal shelter or rescue network.

## Problem
Malaysian shelters run near capacity on tight budgets. Every day an animal waits costs about RM4 (dog) or RM2.70 (cat) in care, and it also blocks a kennel space that another rescue needs. Shelters already spend money on vet care, photos and fee discounts, but they choose by habit. This project predicts how long each listed animal will wait and then picks the cheapest set of actions that shortens the total wait for the whole shelter.

## Data
Kaggle *PetFinder.my Adoption Prediction*: 14,993 dog and cat listings from 5,595 Malaysian rescuers, with 58,311 photos, the description text and Google NL sentiment. The target `AdoptionSpeed` is binned (0 same day, 1 within 7 days, 2 within 30, 3 within 90, 4 not adopted after 100 days). The model predicts class probabilities, and expected days are computed from them using class midpoints of 0, 4, 19, 60 and 120 days. The 120 days for class 4 is an assumption, tested at 100 and 150.

## Pipeline (`src/`)
| Step | File | What it does |
|---|---|---|
| 0 | `extract_photos.py` | Pixel metrics for all 58k photos, CLIP ViT-B/32 embeddings of each main photo, and sentiment and metadata features |
| 1 | `features.py` | Builds the pet-level table. **Photo Appeal** (0 to 100) is CLIP similarity to "professional, well-lit pet portrait" prompts minus similarity to "dark, blurry, caged, cluttered" prompts, scaled with percentiles fitted once so a new shelter gets the same scale |
| 2 | `model.py` | 5-fold CV **grouped by RescuerID**, so no rescuer is in both train and test. Benchmarks plus the deployed model: an ordinal LightGBM (four models for P(speed > k)) with **monotone constraints** on actionable fields |
| 3 | `actions.py`, `simulate.py` | Ten actions, each mapped to the profile fields it changes and its RM cost. Every combination of actions is re-scored per animal |
| 4 | `optimize.py` | Mixed-integer program (PuLP/CBC): one action set per listing, photographer sessions charged per 15 listings, total cost within budget, minimizing total animal-days |
| 5 | `price_agent.py` | Live price agent: an LLM with web search returns current RM prices and sources, validates them against the cached table, and writes `costs/malaysia_costs_<date>.csv` |
| 6 | `run_effects.py`, `run_impact.py`, `run_demo.py`, `charts.py` | Effects, portfolio impact, demo shelter and charts |

## Results (all out of fold)
| Model | QWK | Accuracy | MAE (days) |
|---|---|---|---|
| Class average baseline | 0.000 | 0.280 | 41.0 |
| Rule: class mix by age band | 0.182 | 0.341 | 39.7 |
| Logistic regression, profile | 0.320 | 0.370 | 37.8 |
| LightGBM, profile only | 0.379 | 0.399 | 37.1 |
| LightGBM + pixel photo metrics (original approach) | 0.380 | 0.398 | 37.0 |
| LightGBM + CLIP photo | 0.415 | 0.413 | 35.8 |
| **Deployed: ordinal, monotone, full** | **0.417** | 0.400 | **35.3** |

* **AI vision is what makes photos count.** Pixel sharpness, contrast and color add nothing, and better pixel scores even go with *slower* adoption, likely because busy, cluttered shots score as "sharp." CLIP appeal reverses that: listings with the best fifth of photos average 47.5 days, against 56.2 for the worst fifth.
* **Calibrated.** Predicted and actual days agree within about 2 days in every decile (`outputs/charts/calibration.png`).
* **What moves adoption** (average days saved per eligible listing): waive fee 5.3, professional photos 2.8, AI-written bio 2.6, treat minor injury 1.6, five phone photos 1.4, phone video 0.7, name 0.3, deworm 0.2, **vaccinate 0.0, sterilize 0.0**. Without constraints, the model says sterilizing *adds* 4.0 days and vaccinating adds 1.0, which is confounding: older, owner-surrendered animals are the ones already treated. We constrain those effects to be zero or helpful, and report the result as "no measurable speed-up," not as "no value." Vet care still matters for welfare and the law.
* **Portfolio test: 50 real rescuers, 4,730 animals.** Free quick wins (phone photos, video, a name, an AI-written bio) save 13,287 animal-days, 6.1% of total waiting time. That frees about 287 kennel places at no cash cost and saves about RM49,600 of care. At RM25 per listing, the optimizer saves another 4,544 animal-days, against 711 for "cheapest actions first" and 13 for "vet care first" with the same money.
* **The economics are honest.** Paid actions cost RM12 to RM48 per extra animal-day saved, more than the RM2.70 to RM4 daily care cost. Paid spend makes sense to free kennel capacity (about RM570 per extra animal place at small budgets), not to save on food.

## Reproduce
```bash
pip install -r requirements.txt
python src/extract_photos.py --kaggle_dir /path/to/petfinder-adoption-prediction   # once, about 15 min on CPU
./run_all.sh                                                                          # about 5 min, every number above
export ANTHROPIC_API_KEY=...; python src/price_agent.py --state "Pulau Pinang"       # optional: refresh prices
```
Seeds are fixed. `data/` is not committed, because Kaggle competition data shouldn't be redistributed. Step 0 rebuilds it from your own Kaggle download, and after that `run_all.sh` needs no images. Price sources are in `costs/`, benchmark tables in `outputs/`, and the deck's charts in `outputs/charts/`.

## Limitations
These are model estimates from observational 2018 to 2019 data, not experiments. Action effects are associations controlled for profile, kennel size and photo content, so a pilot A/B test should confirm them before large spend. The photographer effect assumes a shoot lifts the main photo to the 85th percentile of appeal. Costs are indicative Malaysian prices with sources in `costs/`, and the minor injury treatment price is an assumption. Individual shelters can differ from the model: the Penang demo shelter actually averaged 86 days against 54 predicted, so per-shelter recalibration is a natural next step.
