#!/usr/bin/env bash
# Reproduces every number in the deck from the prepared data/ folder (about 10 minutes on a laptop CPU).
set -e
cd "$(dirname "$0")/src"
python features.py          # pet table, photo appeal score, scalers
python model.py             # benchmarks + final model (artifacts/)
python run_effects.py       # days saved per action, sensitivity to the 120-day assumption
python check_unconstrained.py
python run_impact.py        # 50 real rescuers: optimizer vs rules at 8 budgets
python run_demo.py          # Penang demo kennel plan and budget curve
python charts.py
