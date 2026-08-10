"""
Step 9 — Model training: compare Linear Regression, Random Forest, and
XGBoost on the Gold feature table.

CRITICAL SPLIT DECISION: split by trip_id, NOT by row.
Each trip contributes several snapshot rows (one per days_left checkpoint).
A random row-level split would let different snapshots of the SAME trip
land in both train and test, so the model could partially "recognize" a
trip it's supposedly being tested on -- leakage. Splitting on trip_id first,
then keeping all of a trip's rows together, avoids this.

MODELS COMPARED
1. Linear Regression       -- simple baseline, tells us how much of the
                               signal is linear
2. Random Forest Regressor -- bagged trees, robust to outliers/noise
3. XGBoost Regressor       -- boosted trees, usually the strongest of the
                               three, gives feature importances

METRICS
- MAE, RMSE, R^2 on the held-out test trips
- Error broken out by days_left bucket -- error should SHRINK as departure
  approaches (more booking history available), which is a good sanity check
  that the model is actually using the time-varying features meaningfully
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

GOLD_INPUT = "bus_features_gold.csv"
RANDOM_STATE = 42

NON_FEATURE_COLS = ["trip_id", "route", "final_occupancy_pct"]


def trip_level_split(gold: pd.DataFrame, test_size=0.2):
    unique_trips = gold["trip_id"].unique()
    train_ids, test_ids = train_test_split(
        unique_trips, test_size=test_size, random_state=RANDOM_STATE
    )
    train = gold[gold["trip_id"].isin(train_ids)].copy()
    test = gold[gold["trip_id"].isin(test_ids)].copy()
    return train, test


def evaluate(name, y_true, y_pred, results):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    results.append(dict(model=name, MAE=round(mae, 3), RMSE=round(rmse, 3), R2=round(r2, 4)))
    print(f"{name:20s}  MAE={mae:6.3f}  RMSE={rmse:6.3f}  R2={r2:.4f}")
    return mae, rmse, r2


def error_by_days_left_bucket(name, test_df, y_pred):
    df = test_df.copy()
    df["abs_error"] = np.abs(df["final_occupancy_pct"] - y_pred)
    bins = [0, 3, 7, 15, 30, 100]
    labels = ["0-3d", "4-7d", "8-15d", "16-30d", "30d+"]
    df["days_left_bucket"] = pd.cut(df["days_left"], bins=bins, labels=labels)
    bucket_mae = df.groupby("days_left_bucket", observed=True)["abs_error"].mean().round(3)
    print(f"\n{name} -- MAE by days-to-departure bucket (should shrink toward 0-3d):")
    print(bucket_mae.to_string())
    return bucket_mae


def main():
    gold = pd.read_csv(GOLD_INPUT)
    train, test = trip_level_split(gold)

    print(f"Total rows: {len(gold)} | Unique trips: {gold['trip_id'].nunique()}")
    print(f"Train rows: {len(train)} ({train['trip_id'].nunique()} trips)")
    print(f"Test rows:  {len(test)} ({test['trip_id'].nunique()} trips)")
    assert set(train["trip_id"]).isdisjoint(set(test["trip_id"])), "trip leakage between train/test!"
    print("Confirmed: zero trip_id overlap between train and test.\n")

    feature_cols = [c for c in gold.columns if c not in NON_FEATURE_COLS]
    X_train, y_train = train[feature_cols], train["final_occupancy_pct"]
    X_test, y_test = test[feature_cols], test["final_occupancy_pct"]

    results = []
    bucket_tables = {}

    # ---- 1. Linear Regression baseline ----
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    evaluate("Linear Regression", y_test, pred_lr, results)
    bucket_tables["Linear Regression"] = error_by_days_left_bucket("Linear Regression", test, pred_lr)

    # ---- 2. Random Forest ----
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    evaluate("Random Forest", y_test, pred_rf, results)
    bucket_tables["Random Forest"] = error_by_days_left_bucket("Random Forest", test, pred_rf)

    # ---- 3. XGBoost ----
    xgb = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    evaluate("XGBoost", y_test, pred_xgb, results)
    bucket_tables["XGBoost"] = error_by_days_left_bucket("XGBoost", test, pred_xgb)

    # ---- Save comparison table ----
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison.csv", index=False)
    print("\n=== Final comparison ===")
    print(results_df.to_string(index=False))

    # ---- Save per-bucket error table ----
    bucket_df = pd.DataFrame(bucket_tables)
    bucket_df.to_csv("model_error_by_days_left.csv")

    # ---- Feature importances (XGBoost) ----
    importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    importances.to_csv("xgboost_feature_importance.csv", header=["importance"])
    print("\nTop 10 XGBoost feature importances:")
    print(importances.head(10).to_string())

    # ---- Save trained models ----
    import pickle
    with open("model_linear_regression.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open("model_random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open("model_xgboost.pkl", "wb") as f:
        pickle.dump(xgb, f)
    print("\nSaved: model_comparison.csv, model_error_by_days_left.csv, "
          "xgboost_feature_importance.csv, 3x .pkl model files")

    return results_df, bucket_df, importances


if __name__ == "__main__":
    main()