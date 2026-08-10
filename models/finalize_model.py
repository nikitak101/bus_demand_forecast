"""
Step 10a — Finalize the production model.

WHY RETRAIN ON 100% OF THE DATA:
train_models.py already gave us an honest, unbiased performance estimate
(XGBoost: MAE 4.03, RMSE 5.35, R2 0.904 on a held-out 20% of trips never
seen during training). That estimate stands on its own and is what you'd
quote as "expected real-world performance."

But once a model type is CHOSEN, holding back 20% of data from the actual
deployed model just throws away signal for no further benefit -- there's no
more model selection happening, so no more need for a held-out set. Standard
practice: retrain the same architecture/hyperparameters on ALL available
data for the artifact you actually ship. The test metrics remain valid as
your expected-performance number; the final .pkl is a strictly-more-informed
version of the model that produced them.

This script also saves the two things predict.py needs to reproduce Gold
features for brand-new trips:
    feature_columns.json   -- exact column order/set the model expects
    route_avg_lookup.json  -- per-route avg final_occupancy_pct (+ global
                               mean fallback), for the route_avg_occupancy
                               feature on NEW trips (no leave-one-out needed
                               here, since these aren't trips being scored
                               against their own historical outcome)
"""

import pandas as pd
import json
import pickle
from xgboost import XGBRegressor

GOLD_INPUT = "bus_features_gold.csv"
TRIPS_INPUT = "bus_trips_master.csv"

FINAL_MODEL_OUTPUT = "model_xgboost_final.pkl"
FEATURE_COLUMNS_OUTPUT = "feature_columns.json"
ROUTE_LOOKUP_OUTPUT = "route_avg_lookup.json"

NON_FEATURE_COLS = ["trip_id", "route", "final_occupancy_pct"]

# same hyperparameters validated in train_models.py -- not re-tuned here,
# just retrained on more data
XGB_PARAMS = dict(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
)


def main():
    gold = pd.read_csv(GOLD_INPUT)
    trips = pd.read_csv(TRIPS_INPUT)

    feature_cols = [c for c in gold.columns if c not in NON_FEATURE_COLS]
    X = gold[feature_cols]
    y = gold["final_occupancy_pct"]

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)

    with open(FINAL_MODEL_OUTPUT, "wb") as f:
        pickle.dump(model, f)

    with open(FEATURE_COLUMNS_OUTPUT, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # route lookup for scoring brand-new trips (not leave-one-out --
    # these new trips have no "own outcome" to exclude)
    route_avg = trips.groupby("route")["final_occupancy_pct"].mean().round(2).to_dict()
    global_mean = round(trips["final_occupancy_pct"].mean(), 2)
    route_lookup = {"routes": route_avg, "global_mean": global_mean}
    with open(ROUTE_LOOKUP_OUTPUT, "w") as f:
        json.dump(route_lookup, f, indent=2)

    print(f"Final model trained on {len(gold)} rows ({gold['trip_id'].nunique()} trips)")
    print(f"Saved: {FINAL_MODEL_OUTPUT}")
    print(f"Saved: {FEATURE_COLUMNS_OUTPUT} ({len(feature_cols)} features)")
    print(f"Saved: {ROUTE_LOOKUP_OUTPUT} ({len(route_avg)} routes + global mean {global_mean})")


if __name__ == "__main__":
    main()