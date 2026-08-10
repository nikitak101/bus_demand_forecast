"""
Step 10b - Prediction pipeline for brand-new trips.

    NEW USER INPUT
         v
    Build the SAME Gold-style features used in training
         v
    Load model_xgboost_final.pkl
         v
    Predicted final occupancy %
         v
    Demand tier + business recommendation (bonus layer)

CRITICAL RULE: predict.py must derive features EXACTLY the same way
feature_engineering.py did, in the EXACT same column order (feature_columns.json),
or the model will silently score garbage. This is the #1 real-world failure
mode of ML pipelines ("training/serving skew") -- so every feature below is
built with the same formulas/reference data as the training pipeline, not
re-derived from scratch.

WHAT THIS SCRIPT NEEDS FROM THE CALLER (a "live" prediction request):
    origin, destination           -- route
    bus_type                      -- one of the 5 known types
    capacity                      -- seats on the bus
    base_price_inr                -- ticket price
    departure_date                -- "YYYY-MM-DD"
    booking_open_date             -- "YYYY-MM-DD"
    as_of_date                    -- "YYYY-MM-DD", the day the prediction is
                                      being made (defaults to today)
    current_seats_booked          -- seats booked as of as_of_date
    recent_seats_booked (optional)-- dict of {"-1": n, "-3": n, "-7": n}
                                      seats booked 1/3/7 days before as_of_date,
                                      for real lag/velocity features. If
                                      omitted, lags are ESTIMATED from the
                                      average booking pace so far (flagged
                                      in the output as estimated, not exact).

WHAT IT DOES NOT DO: retrain, backfill Silver/Bronze, or touch the training
data. It's a pure function: new trip in -> Gold-shaped feature row ->
prediction out.
"""

import json
import pickle
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from reference_data import ROUTES, BUS_TYPES, HOLIDAYS

# Resolve paths relative to THIS FILE's location, not the terminal's current
# working directory. Without this, running the script via a full path (e.g.
# `python C:\...\api\predict.py` from a different folder, or an IDE "Run"
# button that sets cwd to the project root) fails with FileNotFoundError
# even though the files are right there next to the script.
_HERE = Path(__file__).resolve().parent
MODEL_PATH = _HERE / "model_xgboost_final.pkl"
FEATURE_COLUMNS_PATH = _HERE / "feature_columns.json"
ROUTE_LOOKUP_PATH = _HERE / "route_avg_lookup.json"


@lru_cache(maxsize=1)
def _load_artifacts():
    """
    Loads the model + supporting JSON files from disk ONCE per process and
    caches the result (lru_cache with maxsize=1 acts as a simple memoized
    singleton here). Every subsequent call returns the same in-memory
    objects instead of re-reading from disk -- this matters once a FastAPI
    server is handling many requests per second; reloading a pickle file on
    every single call would add unnecessary disk I/O and latency to every
    prediction.
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_cols = json.load(f)
    with open(ROUTE_LOOKUP_PATH) as f:
        route_lookup = json.load(f)
    return model, feature_cols, route_lookup


def _is_holiday_window(d: datetime) -> bool:
    for h in HOLIDAYS:
        hd = datetime.strptime(h, "%Y-%m-%d")
        if abs((d - hd).days) <= 2:
            return True
    return False


def _lookup_route(origin: str, destination: str):
    """Find popularity tier + distance from the reference skeleton.
    Falls back to sensible defaults (with a warning) for unseen routes."""
    for o, d, dist, tier in ROUTES:
        if o.lower() == origin.lower() and d.lower() == destination.lower():
            return dist, tier, True
    return None, 2, False  # unseen route -> default medium tier, no distance


def build_feature_row(trip_input: dict, feature_cols: list, route_lookup: dict):
    warnings = []

    origin = trip_input["origin"]
    destination = trip_input["destination"]
    bus_type = trip_input["bus_type"]
    capacity = trip_input["capacity"]
    base_price = trip_input["base_price_inr"]
    departure_date = datetime.strptime(trip_input["departure_date"], "%Y-%m-%d")
    booking_open_date = datetime.strptime(trip_input["booking_open_date"], "%Y-%m-%d")
    as_of_date = datetime.strptime(
        trip_input.get("as_of_date", datetime.today().strftime("%Y-%m-%d")), "%Y-%m-%d"
    )
    current_seats_booked = trip_input["current_seats_booked"]

    if bus_type not in BUS_TYPES:
        raise ValueError(f"Unknown bus_type '{bus_type}'. Must be one of {list(BUS_TYPES.keys())}")

    distance, tier, route_known = _lookup_route(origin, destination)
    if not route_known:
        warnings.append(
            f"Route '{origin} -> {destination}' not in reference skeleton; "
            f"using default popularity_tier=2 and provided/estimated distance."
        )
        distance = trip_input.get("distance_km", 300)  # rough fallback

    booking_window_days = (departure_date - booking_open_date).days
    days_since_open = (as_of_date - booking_open_date).days
    days_left = (departure_date - as_of_date).days
    if days_left < 0:
        raise ValueError("as_of_date is after departure_date.")

    is_holiday = _is_holiday_window(departure_date)
    departure_weekday = departure_date.strftime("%A")

    fare_lo, fare_hi = BUS_TYPES[bus_type]["fare_per_km"]
    route_median_price = distance * (fare_lo + fare_hi) / 2
    price_ratio = base_price / route_median_price if route_median_price else 1.0

    route_key = f"{origin} -> {destination}"
    route_avg_occupancy = route_lookup["routes"].get(route_key, route_lookup["global_mean"])
    if route_key not in route_lookup["routes"]:
        warnings.append(f"No historical data for route '{route_key}'; using global average occupancy.")

    current_occupancy_pct = 100 * current_seats_booked / capacity
    seats_remaining = capacity - current_seats_booked
    days_left_ratio = days_left / booking_window_days if booking_window_days else 0.0

    recent = trip_input.get("recent_seats_booked")
    if recent and all(k in recent for k in ("-1", "-3", "-7")):
        seats_lag1, seats_lag3, seats_lag7 = recent["-1"], recent["-3"], recent["-7"]
        lag_source = "provided"
    else:
        avg_daily_rate = current_seats_booked / max(days_since_open, 1)
        seats_lag1 = max(0, current_seats_booked - avg_daily_rate * 1)
        seats_lag3 = max(0, current_seats_booked - avg_daily_rate * 3)
        seats_lag7 = max(0, current_seats_booked - avg_daily_rate * 7)
        lag_source = "estimated"
        warnings.append(
            "recent_seats_booked not provided -- lag/velocity features "
            "estimated from average booking pace so far (less accurate "
            "than real history)."
        )

    occupancy_lag1 = 100 * seats_lag1 / capacity
    occupancy_lag3 = 100 * seats_lag3 / capacity
    occupancy_lag7 = 100 * seats_lag7 / capacity
    booking_velocity_3d = (current_seats_booked - seats_lag3) / 3
    booking_velocity_7d = (current_seats_booked - seats_lag7) / 7

    row = {
        "days_left": days_left,
        "current_seats_booked": current_seats_booked,
        "capacity": capacity,
        "current_occupancy_pct": current_occupancy_pct,
        "booking_window_days": booking_window_days,
        "occupancy_lag1": occupancy_lag1,
        "occupancy_lag3": occupancy_lag3,
        "occupancy_lag7": occupancy_lag7,
        "seats_lag3": seats_lag3,
        "seats_lag7": seats_lag7,
        "booking_velocity_3d": booking_velocity_3d,
        "booking_velocity_7d": booking_velocity_7d,
        "seats_remaining": seats_remaining,
        "days_since_open": days_since_open,
        "days_left_ratio": days_left_ratio,
        "distance_km": distance,
        "popularity_tier": tier,
        "base_price_inr": base_price,
        "price_ratio": price_ratio,
        "is_holiday_period": int(is_holiday),
        "route_avg_occupancy_loo": route_avg_occupancy,
    }
    for bt in BUS_TYPES.keys():
        row[f"bus_{bt}"] = 1 if bt == bus_type else 0
    for wd in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        row[f"wd_{wd}"] = 1 if wd == departure_weekday else 0

    missing = [c for c in feature_cols if c not in row]
    if missing:
        raise KeyError(f"Feature builder is missing columns the model expects: {missing}")
    ordered = {c: row[c] for c in feature_cols}

    meta = dict(
        distance_km=distance, popularity_tier=tier, route_known=route_known,
        departure_weekday=departure_weekday, is_holiday_period=is_holiday,
        booking_window_days=booking_window_days, days_left=days_left,
        lag_source=lag_source, warnings=warnings,
    )
    return pd.DataFrame([ordered]), meta


def demand_tier_and_recommendation(predicted_pct: float):
    if predicted_pct >= 80:
        return "HIGH", "Consider adding frequency on this route -- demand is likely to exceed capacity."
    elif predicted_pct >= 50:
        return "MEDIUM", "Demand looks healthy -- no action needed, monitor as departure approaches."
    else:
        return "LOW", "Consider promotional pricing or a discount to lift bookings before departure."


def predict(trip_input: dict) -> dict:
    model, feature_cols, route_lookup = _load_artifacts()
    X, meta = build_feature_row(trip_input, feature_cols, route_lookup)
    predicted_pct = float(model.predict(X)[0])
    predicted_pct = float(np.clip(predicted_pct, 0, 100))
    predicted_seats = round(predicted_pct / 100 * trip_input["capacity"])
    tier, recommendation = demand_tier_and_recommendation(predicted_pct)

    return dict(
        predicted_final_occupancy_pct=round(predicted_pct, 2),
        predicted_final_seats=predicted_seats,
        capacity=trip_input["capacity"],
        demand_tier=tier,
        recommendation=recommendation,
        days_left=meta["days_left"],
        popularity_tier_used=meta["popularity_tier"],
        route_known_in_reference_data=meta["route_known"],
        lag_feature_source=meta["lag_source"],
        warnings=meta["warnings"],
    )


if __name__ == "__main__":
    example_input = dict(
        origin="Mumbai",
        destination="Pune",
        bus_type="AC Sleeper",
        capacity=32,
        base_price_inr=800,
        departure_date="2026-08-30",
        booking_open_date="2026-08-10",
        as_of_date="2026-08-20",
        current_seats_booked=18,
    )
    result = predict(example_input)
    print(json.dumps(result, indent=2))