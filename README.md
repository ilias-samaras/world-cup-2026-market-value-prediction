A machine-learning pipeline that learns from the **2014, 2018 and 2022** World Cups to predict how a player's transfer-market value will change after the **2026** World Cup — i.e. *which players are positioned to break out*.

> University project (Business Administration / Information Systems). The novelty is predicting event-driven value **changes** with an event-study framing, rather than value **levels** (which is what most ML literature does).

---

## Problem framing

For every `(player, tournament)` we measure market value **just before** the tournament and **~3 months after** it, and model the change. Two target definitions are explored:

- **Target A — log-ratio:** `log(value_after / value_before)` → "who has the biggest *percentage* upside" (surfaces young, pre-discovered talent).
- **Target B — absolute Δ€ (signed-log):** `sign(Δ)·log(1+|Δ|)` → "who adds the most value in *euros*" (surfaces established stars / big breakouts).

The 2026 tournament is the prediction target: we only use **pre-tournament** features (age, value, value trend, prior WC experience, position), since in-tournament performance is unknown ahead of time.

## Data

- **[Fjelstul World Cup Database](https://github.com/jfjelstul/worldcup)** — squads, appearances, goals, team results (who played in which WC).
- **Transfermarkt** — `players_transf.csv`, `player_valuations.csv` (market-value time series), `transfers.csv`.
- **`mundial_squads.csv`** — a custom-built dataset of the projected 2026 squads (48 nations).

### The core challenge: entity resolution
Fjelstul (`P-#####` IDs) and Transfermarkt (integer IDs) share **no common key**. Players are linked by **name + date-of-birth** with a 3-tier matcher (exact → first+last → fuzzy via `rapidfuzz`, all anchored on DOB to avoid false matches):

- Historical squads (2014/18/22): **~83%** matched → **1,710** usable training rows.
- 2026 squads: **780 / 1,248** matched → **384** players with a market value (the prediction set).

## Pipeline

| Stage | What it does |
|------|---------------|
| 1–4  | Setup, Transfermarkt bridge, entity resolution, market-value extraction (`value_at`) |
| 5–7  | Build training rows (2014/18/22), 2026 rows, merge + target + features |
| 8–10 | Temporal evaluation harness (train 2014+2018, **test 2022**) + baselines + Keras MLP |
| 11   | 2026 predictions (Target A) |
| 12   | Permutation feature importance |
| 13–15| Target B (absolute Δ€) — full re-run + 2026 list |

## Models & evaluation

Evaluated with a **temporal split** (train on 2014+2018, test on 2022 — the honest "predict the next World Cup" scenario). Models: `DummyRegressor` (floor), `Ridge`, `RandomForest`, `GradientBoosting`, and a **Keras MLP** (primary). Metrics: MAE, RMSE, R², Spearman (ranking quality) and Precision@20 (of the top-20 predicted risers, how many really rose).

**Key results (test = 2022):**
- All models beat the mean-baseline → real signal exists.
- MLP / Ridge lead; tree ensembles overfit on Target A. On Target B the models agree more (R² ≈ 0.20–0.22), suggesting absolute-€ change is more stably learnable.
- **Age dominates** (corr(prediction, age) ≈ −0.86), with **value momentum** (`value_trend_1y`) the surprising second-strongest feature, ahead of current value. Position and prior-WC experience are near-noise.

## Limitations (discussed in the report)

- **Non-random missingness:** unmatched players skew toward smaller federations (Arabic-name transliteration, lower Transfermarkt coverage) → a mild bias toward larger footballing nations.
- **Regression to the mean:** pre-tournament features can't predict the rare +500% explosions (those are driven by in-tournament performance, deliberately excluded for 2026).
- `prior_wc` only counts 2014/18/22 (pre-2014 WCs ignored — negligible for 2026).

## How to run

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook market_value_after_wc_prediction.ipynb   # Run All, top to bottom
```

All CSVs must sit in the same folder as the notebook (`UP = "."`).

## Repository structure

```
market_value_after_wc_prediction.ipynb   # the full pipeline (15 stages)
data CSVs (Fjelstul + Transfermarkt + mundial_squads)
predictions_2026.csv          # Target A — % upside ranking
predictions_2026_targetB.csv  # Target B — absolute €-gain ranking
feature_importance.png        # permutation importance (MLP)
requirements.txt
```
## Data sources & licenses

- **Fjelstul World Cup Database** — © 2023 Joshua C. Fjelstul, Ph.D.,
  licensed under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode).
  Source: https://github.com/jfjelstul/worldcup
  Files used: squads.csv, players.csv, goals.csv, player_appearances.csv, qualified_teams.csv.
  Modifications: filtered to the 2014/2018/2022 tournaments and linked to Transfermarkt
  data via name + date-of-birth matching.

- **Football Data from Transfermarkt** (David Cariboo) —
  licensed [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain).
  Source: https://www.kaggle.com/datasets/davidcariboo/player-scores
  Files used: players_transf.csv, player_valuations.csv, transfers.csv.

- **mundial_squads.csv** — compiled by us from official FIFA sources (2026 squads).

Because this project redistributes data from the Fjelstul World Cup Database (CC-BY-SA 4.0),
the dataset files in this repository are released under the same **CC-BY-SA 4.0** license.
The source code is © the authors.
