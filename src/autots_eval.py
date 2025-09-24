# run_autots_eval.py
# Evaluate standalone models (NO ensembles) on existing data only.
# Models: Naive baselines + ARIMA/SARIMA/ETS/GLM/XGBoost
# Exogenous: month dummies, weekday dummies, holiday (0/1) as-is
# Outputs: per-horizon leaderboards, best-per-model summaries, combined summaries

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from autots import AutoTS

# ----------------------------
# Config
# ----------------------------
INPUT_CSV = "./dummy_timeseries.csv"   # expects: date, value, month_name, weekday_name, holiday
SERIES_ID = "series_A"
HORIZONS = [7, 14, 30]

# Baselines + your chosen models (all tested INDIVIDUALLY)
MODEL_LIST = [
    "AverageValueNaive",
    "SeasonalNaive",
    "ARIMA",
    "ETS",
    "GLM",
]

ENSEMBLE = None          # no mixing: pure head-to-head
VERBOSE = 1
MAX_GENERATIONS = 5      # increase for deeper searches
NUM_VALIDATIONS = 3      # more folds -> more robust scores

# Lean, non-redundant with your exogenous dummies
TRANSFORMERS = [
    "DifferencedTransformer",
    "StandardScaler",
]

# ----------------------------
# Load & prep
# ----------------------------
df = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values("date")
df["series_id"] = SERIES_ID

# Build exogenous: dummyize month & weekday, keep holiday as-is (already 0/1)
month_dummies = pd.get_dummies(df["month_name"], prefix="month", drop_first=True)
weekday_dummies = pd.get_dummies(df["weekday_name"], prefix="weekday", drop_first=True)
exo = pd.concat([month_dummies, weekday_dummies, df[["holiday"]].reset_index(drop=True)], axis=1)

# AutoTS minimum columns
train_long = df[["date", "series_id", "value"]].copy()

all_best_rows = []
all_leaderboards = []

for horizon in HORIZONS:
    print(f"\n=== Evaluating horizon (backtest fold size): {horizon} days ===")

    model = AutoTS(
        forecast_length=horizon,      # fold size for backtesting
        frequency="D",
        model_list=MODEL_LIST,
        ensemble=ENSEMBLE,            # no ensembling
        transformer_list=TRANSFORMERS,
        max_generations=MAX_GENERATIONS,
        num_validations=NUM_VALIDATIONS,
        verbose=VERBOSE,
    )

    # Fit on existing data; AutoTS will perform rolling-origin holdout backtests
    model = model.fit(
        train_long,
        date_col="date",
        value_col="value",
        id_col="series_id",
        future_regressor=exo,   # exogenous regressors aligned to existing data only
    )

    # Leaderboard of all individual runs (across folds), averaged metrics
    results = model.results()
    keep_cols = ["model_name", "model_parameters", "coverage", "smape", "mae", "rmse"]
    leaderboard = results[keep_cols].copy()
    leaderboard["horizon"] = horizon

    # Console preview
    print("\nTop 5 models by sMAPE for horizon", horizon)
    print(leaderboard.sort_values("smape", ascending=True).head(5).to_string(index=False))

    # Save full leaderboard
    leader_path = f"leaderboard_eval_{horizon}d.csv"
    leaderboard.to_csv(leader_path, index=False)
    print(f"Saved leaderboard → {leader_path}")

    # Best row per model family (lowest sMAPE) at this horizon
    best_per_model = (
        leaderboard.sort_values("smape", ascending=True)
                  .groupby("model_name", as_index=False)
                  .first()
    )
    best_per_model["horizon"] = horizon
    best_path = f"best_per_model_eval_{horizon}d.csv"
    best_per_model.to_csv(best_path, index=False)
    print(f"Saved best-per-model summary → {best_path}")

    all_best_rows.append(best_per_model)
    all_leaderboards.append(leaderboard)

# ----------------------------
# Combined summaries across horizons
# ----------------------------
combined_best = pd.concat(all_best_rows, ignore_index=True)
combined_best = combined_best[["horizon", "model_name", "model_parameters", "coverage", "smape", "mae", "rmse"]]
combined_best.sort_values(["horizon", "smape"], inplace=True)
combined_best.to_csv("model_summary_eval_all_horizons.csv", index=False)
print("\nSaved combined best-per-model summary → model_summary_eval_all_horizons.csv")

combined_leader = pd.concat(all_leaderboards, ignore_index=True)
combined_leader.to_csv("model_leaderboards_eval_all_horizons.csv", index=False)
print("Saved combined full leaderboards → model_leaderboards_eval_all_horizons.csv")
