"""
Bus Demand & Booking Forecasting - Synthetic-but-grounded dataset generator.

METHOD SUMMARY (for your README / interview explanation):
1. Trip-level attributes (route, distance, bus type, capacity, base price,
   departure date) are drawn from the REAL reference skeleton in
   reference_data.py.
2. For each trip, we simulate a daily booking trajectory using a logistic
   (S-curve) function of "days since booking opened":

       booked(t) = FinalOccupancy * capacity / (1 + exp(-k * (t - midpoint)))

   - FinalOccupancy is the trip's terminal demand level, driven by:
       * popularity tier of the route
       * weekday of departure (Fri/Sat/Sun/holiday = higher)
       * price relative to the route's median price (elasticity)
       * random trip-level noise (unobserved demand shocks)
   - midpoint and k (steepness) vary per trip to avoid every booking curve
     looking identical (some trips fill early, some are last-minute).
3. Day-to-day multiplicative noise is added so the curve isn't perfectly
   smooth (real booking counts are noisy/lumpy, esp. for low-capacity buses).
4. A small cancellation process is applied in the last few days before
   departure (some booked seats get released).
5. Everything is clipped to [0, capacity] and rounded to whole seats.

This gives you a dataset shaped exactly like real booking-curve data
(one row per trip per day-before-departure) without needing access to a
platform's private transaction database.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from reference_data import ROUTES, BUS_TYPES, HOLIDAYS

RNG = np.random.default_rng(42)

N_TRIPS = 4000
DEPARTURE_DATE_START = datetime(2025, 8, 1)
DEPARTURE_DATE_END = datetime(2026, 6, 30)

TIER_DEMAND_BASE = {1: 0.85, 2: 0.68, 3: 0.50}   # avg final occupancy by popularity tier


def is_holiday_window(d: datetime) -> bool:
    # flag if date is within 2 days of a listed holiday (holiday travel rush)
    for h in HOLIDAYS:
        hd = datetime.strptime(h, "%Y-%m-%d")
        if abs((d - hd).days) <= 2:
            return True
    return False


def weekday_multiplier(d: datetime) -> float:
    wd = d.weekday()  # 0=Mon ... 6=Sun
    mult = {0: 0.85, 1: 0.80, 2: 0.82, 3: 0.95, 4: 1.15, 5: 1.25, 6: 1.10}
    return mult[wd]


def make_trip(trip_id: int) -> dict:
    origin, dest, distance, tier = ROUTES[RNG.integers(0, len(ROUTES))]
    bus_type = RNG.choice(list(BUS_TYPES.keys()))
    cap_lo, cap_hi = BUS_TYPES[bus_type]["capacity"]
    capacity = int(RNG.integers(cap_lo, cap_hi + 1))
    fare_lo, fare_hi = BUS_TYPES[bus_type]["fare_per_km"]
    fare_per_km = RNG.uniform(fare_lo, fare_hi)
    base_price = round(distance * fare_per_km, -1)  # round to nearest 10 INR

    days_offset = int(RNG.integers(0, (DEPARTURE_DATE_END - DEPARTURE_DATE_START).days))
    departure_date = DEPARTURE_DATE_START + timedelta(days=days_offset)
    booking_open_days = int(RNG.integers(14, 46))  # booking opens 14-45 days before
    booking_open_date = departure_date - timedelta(days=booking_open_days)

    holiday_flag = is_holiday_window(departure_date)
    wd_mult = weekday_multiplier(departure_date)

    # price relative to route's typical price (elasticity effect applied later using this)
    route_median_price = distance * (fare_lo + fare_hi) / 2
    price_ratio = base_price / route_median_price  # >1 = pricier than typical

    return dict(
        trip_id=trip_id,
        origin=origin,
        destination=dest,
        route=f"{origin} -> {dest}",
        distance_km=distance,
        popularity_tier=tier,
        bus_type=bus_type,
        capacity=capacity,
        base_price_inr=base_price,
        departure_date=departure_date,
        departure_weekday=departure_date.strftime("%A"),
        booking_open_date=booking_open_date,
        booking_window_days=booking_open_days,
        is_holiday_period=holiday_flag,
        weekday_multiplier=round(wd_mult, 3),
        price_ratio=round(price_ratio, 3),
    )


def simulate_final_occupancy(trip: dict) -> float:
    base = TIER_DEMAND_BASE[trip["popularity_tier"]]
    demand = base * trip["weekday_multiplier"]
    if trip["is_holiday_period"]:
        demand *= 1.30
    # price elasticity: pricier-than-median trips see reduced demand, cheaper see a lift
    elasticity = -0.6
    demand *= (trip["price_ratio"] ** elasticity)
    # unobserved trip-level demand shock
    demand *= RNG.normal(1.0, 0.12)
    return float(np.clip(demand, 0.05, 1.0))


def simulate_booking_curve(trip: dict) -> pd.DataFrame:
    capacity = trip["capacity"]
    window = trip["booking_window_days"]
    final_occ = simulate_final_occupancy(trip)
    final_seats = final_occ * capacity

    # per-trip logistic shape params: some trips fill early, some late
    midpoint = RNG.uniform(window * 0.35, window * 0.75)
    steepness = RNG.uniform(0.15, 0.35)

    rows = []
    prev_booked = 0
    for t in range(0, window + 1):  # t = days since booking opened
        days_to_departure = window - t
        logistic = 1 / (1 + np.exp(-steepness * (t - midpoint)))
        target = final_seats * logistic

        # day-to-day multiplicative noise (bookings arrive lumpily)
        noisy = target * RNG.normal(1.0, 0.06)
        booked = max(prev_booked, noisy)  # cumulative bookings shouldn't go down (pre-cancellation)

        # cancellations kick in mainly in the last 5 days before departure
        cancels = 0
        if days_to_departure <= 5:
            cancel_rate = RNG.uniform(0.0, 0.04)
            cancels = booked * cancel_rate

        net_booked = int(np.clip(round(booked - cancels), 0, capacity))
        prev_booked = max(prev_booked, net_booked)

        current_date = trip["booking_open_date"] + timedelta(days=t)
        rows.append(dict(
            trip_id=trip["trip_id"],
            date=current_date,
            days_to_departure=days_to_departure,
            days_since_booking_open=t,
            seats_booked=net_booked,
            cancellations=int(round(cancels)),
            occupancy_pct=round(100 * net_booked / capacity, 2),
        ))

    return pd.DataFrame(rows)


def main():
    trips = [make_trip(i) for i in range(1, N_TRIPS + 1)]
    trips_df = pd.DataFrame(trips)

    daily_frames = []
    for trip in trips:
        daily_frames.append(simulate_booking_curve(trip))
    daily_df = pd.concat(daily_frames, ignore_index=True)

    # final summary per trip (useful as a simpler ML target: predict final occupancy)
    final_summary = (
        daily_df.sort_values("days_to_departure")
        .groupby("trip_id")
        .first()[["seats_booked", "occupancy_pct"]]
        .rename(columns={"seats_booked": "final_seats_booked", "occupancy_pct": "final_occupancy_pct"})
        .reset_index()
    )
    trips_full = trips_df.merge(final_summary, on="trip_id")

    trips_full.to_csv("bus_trips_master.csv", index=False)
    daily_df.to_csv("bus_booking_daily.csv", index=False)

    print("Trips generated:", len(trips_full))
    print("Daily booking rows generated:", len(daily_df))
    print(trips_full.head(3).to_string())
    print(daily_df.head(5).to_string())


if __name__ == "__main__":
    main()


''' Step 2 — Generate the basic booking curve

For every trip, generate the expected booking pattern using the logistic/S-curve.

Days before departure
        ↓
Logistic function
        ↓
Base expected bookings

Example:

20 days → 3 seats
15 days → 8 seats
10 days → 17 seats
5 days  → 29 seats
0 days  → 37 seats

This gives you the basic demand pattern.

Step 3 — Apply factors affecting demand

Now modify that base curve based on things like:

Route popularity
Price
Holiday
Weekend
Bus type

For example:

Base bookings = 20

Popular route
20 × 1.20

Holiday
24 × 1.15

Higher price
↓ demand

Noise
± small random variation

So now you get realistic synthetic bookings.

Step 4 — Apply cancellations

Near departure, simulate some cancellations.

Before cancellation = 35
Cancellations = 2

Final bookings = 33

Also enforce:

Final bookings ≤ Bus capacity
Step 5 — NOW you have your raw synthetic booking data

For example:

T001 | Mumbai-Pune | 10 Aug | 20 days | 3 seats
T001 | Mumbai-Pune | 11 Aug | 19 days | 4 seats
T001 | Mumbai-Pune | 12 Aug | 18 days | 5 seats
...

This is essentially your raw booking event data.

Step 6 — Store raw data in Bronze

You keep the raw/generated booking events.

Bronze
   ↓
Raw booking data

Don't do heavy transformations here. '''