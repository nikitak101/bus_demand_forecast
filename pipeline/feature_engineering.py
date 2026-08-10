"""
Step 8 — Silver -> Gold: feature engineering.

INPUT (Silver + dimension table):
    bus_booking_silver.csv  -- trip_id, booking_date, days_left, seats_booked,
                                capacity, occupancy_pct
    bus_trips_master.csv    -- trip dimension table

OUTPUT (Gold):
    bus_features_gold.csv       -- one row per (trip_id, snapshot days_left),
                                    ML-ready features + target
    gold_feature_dictionary.md  -- what every column means and why it's safe

DESIGN PRINCIPLE: SNAPSHOT-BASED, LEAKAGE-SAFE FEATURES
A single trip's full daily history isn't one training example — it's a
sequence. To turn it into a supervised-learning table, we sample each trip
at several "snapshot" points (e.g. 30, 20, 10, 5, 1 days before departure)
and ask: "given everything knowable as of THIS day, what will the final
occupancy be?" Every feature below is computed using ONLY data at-or-before
its snapshot day — nothing from later in the same trip's timeline.

FEATURE GROUPS
1. Static (known at booking-open, non-leaky by construction):
   distance_km, capacity, base_price_inr, price_ratio, popularity_tier,
   bus_type (one-hot), departure_weekday (one-hot), is_holiday_period,
   booking_window_days, route_avg_occupancy_loo (route popularity signal,
   computed via LEAVE-ONE-OUT so a trip's own outcome never informs its own
   feature)

2. Time-varying / booking-progress (as-of the snapshot day only):
   days_left, days_left_ratio, current_seats_booked, current_occupancy_pct,
   seats_remaining, occupancy_lag1/3/7 (via groupby shift -- strictly past
   data), booking_velocity_3d/7d

TARGET:
   final_occupancy_pct (from the trips master table)
"""

import pandas as pd
import numpy as np

SILVER_INPUT = "bus_booking_silver.csv"
TRIPS_INPUT = "bus_trips_master.csv"

GOLD_OUTPUT = "bus_features_gold.csv"
DICT_OUTPUT = "gold_feature_dictionary.md"

# Snapshot checkpoints to sample per trip (only kept if the trip's booking
# window actually reaches that far back)
SNAPSHOT_DAYS_LEFT = [45, 40, 35, 30, 25, 20, 15, 12, 10, 7, 5, 3, 1]


def add_time_varying_features(silver: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    df = silver.merge(trips[["trip_id", "booking_window_days"]], on="trip_id", how="left")
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    df = df.sort_values(["trip_id", "booking_date"]).reset_index(drop=True)

    g = df.groupby("trip_id")

    # Lags -- strictly PAST values only (shift moves data forward in time,
    # so row t's lag1 is row t-1's value, never row t's own or future value)
    df["occupancy_lag1"] = g["occupancy_pct"].shift(1)
    df["occupancy_lag3"] = g["occupancy_pct"].shift(3)
    df["occupancy_lag7"] = g["occupancy_pct"].shift(7)
    df["seats_lag3"] = g["seats_booked"].shift(3)
    df["seats_lag7"] = g["seats_booked"].shift(7)

    df["booking_velocity_3d"] = (df["seats_booked"] - df["seats_lag3"]) / 3
    df["booking_velocity_7d"] = (df["seats_booked"] - df["seats_lag7"]) / 7

    df["seats_remaining"] = df["capacity"] - df["seats_booked"]
    df["days_since_open"] = df["booking_window_days"] - df["days_left"]
    df["days_left_ratio"] = df["days_left"] / df["booking_window_days"]

    return df


def leave_one_out_route_encoding(trips: pd.DataFrame) -> pd.DataFrame:
    """
    Route popularity as a numeric feature, computed WITHOUT letting a trip's
    own outcome leak into its own feature value (leave-one-out mean).
    route_avg_occupancy_loo = (sum of final_occupancy_pct for this route,
    excluding this trip) / (count of trips on this route, excluding this trip)
    """
    route_sum = trips.groupby("route")["final_occupancy_pct"].transform("sum")
    route_count = trips.groupby("route")["final_occupancy_pct"].transform("count")
    loo = (route_sum - trips["final_occupancy_pct"]) / (route_count - 1)
    # routes with only 1 trip have no valid LOO value -> fall back to global mean
    global_mean = trips["final_occupancy_pct"].mean()
    trips = trips.copy()
    trips["route_avg_occupancy_loo"] = loo.fillna(global_mean)
    return trips


def build_gold():
    silver = pd.read_csv(SILVER_INPUT)
    trips = pd.read_csv(TRIPS_INPUT)

    trips = leave_one_out_route_encoding(trips)
    daily = add_time_varying_features(silver, trips)

    # keep only sampled snapshot days_left, and only if available for that trip
    snapshot_rows = daily[daily["days_left"].isin(SNAPSHOT_DAYS_LEFT)].copy()

    # attach static features + target (booking_window_days already present
    # from add_time_varying_features -- don't pull it in again here)
    static_cols = [
        "trip_id", "route", "distance_km", "bus_type", "popularity_tier",
        "base_price_inr", "price_ratio", "departure_weekday",
        "is_holiday_period", "route_avg_occupancy_loo", "final_occupancy_pct",
    ]
    gold = snapshot_rows.merge(trips[static_cols], on="trip_id", how="left")

    gold = gold.rename(columns={
        "seats_booked": "current_seats_booked",
        "occupancy_pct": "current_occupancy_pct",
    })

    # one-hot encode categoricals
    gold = pd.get_dummies(gold, columns=["bus_type", "departure_weekday"], prefix=["bus", "wd"])
    gold["is_holiday_period"] = gold["is_holiday_period"].astype(int)

    # drop rows where lag features are undefined (very first days of a trip's
    # window -- not enough history yet for a meaningful velocity signal)
    before = len(gold)
    gold = gold.dropna(subset=["occupancy_lag3", "occupancy_lag7"])
    dropped = before - len(gold)

    # final column order: identifiers -> features -> target
    drop_cols = ["trip_id", "route", "booking_date", "final_occupancy_pct"]
    feature_cols = [c for c in gold.columns if c not in drop_cols]
    gold = gold[["trip_id", "route"] + feature_cols + ["final_occupancy_pct"]]

    gold.to_csv(GOLD_OUTPUT, index=False)

    print(f"Silver rows: {len(silver)}")
    print(f"Snapshot rows sampled: {len(snapshot_rows)}")
    print(f"Dropped (insufficient lag history): {dropped}")
    print(f"Gold rows: {len(gold)}, columns: {len(gold.columns)}")
    print(f"Saved: {GOLD_OUTPUT}")

    return gold


if __name__ == "__main__":
    build_gold()