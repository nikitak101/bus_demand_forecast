"""
Step 7 — Bronze -> Silver: clean, validate, and transform the raw synthetic
booking events into a trustworthy daily table.

INPUT (Bronze):
    bus_trips_master.csv    -- trip dimension table (route, capacity, price...)
    bus_booking_daily.csv   -- raw daily booking events, as generated

OUTPUT (Silver):
    bus_booking_silver.csv  -- cleaned daily table with columns:
        trip_id, booking_date, days_left, seats_booked, capacity, occupancy_pct
    silver_validation_report.txt -- what was checked, what was found/fixed

VALIDATION STEPS (in order):
    1. Correct data types      -- parse dates, cast numeric columns
    2. Remove invalid records  -- nulls in required fields, negative values
    3. Check duplicate records -- duplicate (trip_id, booking_date) rows
    4. Check bookings <= capacity -- seats_booked can never exceed bus capacity
    5. Check valid trip IDs    -- every daily row must reference a real trip
       in the master (dimension) table -- referential integrity
    6. Calculate occupancy %   -- recompute independently from seats_booked
       and capacity, rather than trusting the Bronze-layer value, and flag
       any mismatch (this is the "don't trust upstream math" check)

Each step prints/logs how many rows were affected, so the Silver output is
provably clean rather than just re-saved under a new name.
"""

import pandas as pd
import numpy as np

BRONZE_TRIPS = "bus_trips_master.csv"
BRONZE_DAILY = "bus_booking_daily.csv"

SILVER_OUTPUT = "bus_booking_silver.csv"
REPORT_OUTPUT = "silver_validation_report.txt"


def log(report_lines, msg):
    print(msg)
    report_lines.append(msg)


def transform():
    report = []
    log(report, "=== Bronze -> Silver transformation ===\n")

    # ---- Load Bronze ----
    trips = pd.read_csv(BRONZE_TRIPS)
    daily = pd.read_csv(BRONZE_DAILY)
    log(report, f"Loaded Bronze: {len(trips)} trips, {len(daily)} daily booking rows")

    starting_rows = len(daily)

    # ---- 1. Correct data types ----
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    for col in ["days_to_departure", "days_since_booking_open", "seats_booked", "cancellations"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    trips["capacity"] = pd.to_numeric(trips["capacity"], errors="coerce")
    trips["trip_id"] = pd.to_numeric(trips["trip_id"], errors="coerce").astype("Int64")
    daily["trip_id"] = pd.to_numeric(daily["trip_id"], errors="coerce").astype("Int64")

    bad_dtype = daily[daily["date"].isna() | daily["seats_booked"].isna() | daily["trip_id"].isna()]
    log(report, f"1. Data type coercion: {len(bad_dtype)} rows had unparseable date/seats_booked/trip_id")
    daily = daily.dropna(subset=["date", "seats_booked", "trip_id"])

    # ---- 2. Remove invalid records ----
    before = len(daily)
    daily = daily[daily["seats_booked"] >= 0]
    daily = daily[daily["days_to_departure"] >= 0]
    invalid_removed = before - len(daily)
    log(report, f"2. Removed invalid records (negative seats_booked / days_to_departure): {invalid_removed} rows")

    # ---- 3. Check duplicate records ----
    before = len(daily)
    dupe_mask = daily.duplicated(subset=["trip_id", "date"], keep="first")
    n_dupes = int(dupe_mask.sum())
    daily = daily[~dupe_mask]
    log(report, f"3. Duplicate (trip_id, date) rows found and dropped: {n_dupes}")

    # ---- 4. Check bookings <= capacity ----
    daily = daily.merge(trips[["trip_id", "capacity"]], on="trip_id", how="left", suffixes=("", "_ref"))
    over_capacity = daily[daily["seats_booked"] > daily["capacity"]]
    n_over = len(over_capacity)
    if n_over:
        daily.loc[daily["seats_booked"] > daily["capacity"], "seats_booked"] = daily["capacity"]
    log(report, f"4. Rows where seats_booked > capacity: {n_over} (clipped to capacity)")

    # ---- 5. Check valid trip IDs (referential integrity) ----
    before = len(daily)
    valid_ids = set(trips["trip_id"].dropna().astype(int))
    orphan_mask = ~daily["trip_id"].astype(int).isin(valid_ids)
    n_orphans = int(orphan_mask.sum())
    daily = daily[~orphan_mask]
    log(report, f"5. Orphan rows (trip_id not in master trips table): {n_orphans} dropped")

    # ---- 6. Recalculate occupancy % independently, flag Bronze mismatches ----
    daily["occupancy_pct_recalculated"] = round(100 * daily["seats_booked"] / daily["capacity"], 2)
    if "occupancy_pct" in daily.columns:
        mismatch = (daily["occupancy_pct"] - daily["occupancy_pct_recalculated"]).abs() > 0.5
        n_mismatch = int(mismatch.sum())
        log(report, f"6. Rows where Bronze occupancy_pct disagreed with recalculated value (>0.5pt): {n_mismatch}")
    else:
        log(report, "6. No occupancy_pct present in Bronze -- calculated fresh, nothing to compare")

    # ---- Assemble final Silver schema ----
    silver = daily.rename(columns={
        "date": "booking_date",
        "days_to_departure": "days_left",
    })[["trip_id", "booking_date", "days_left", "seats_booked", "capacity", "occupancy_pct_recalculated"]]
    silver = silver.rename(columns={"occupancy_pct_recalculated": "occupancy_pct"})
    silver = silver.sort_values(["trip_id", "booking_date"]).reset_index(drop=True)

    ending_rows = len(silver)
    log(report, f"\nBronze rows: {starting_rows} -> Silver rows: {ending_rows} "
                 f"({starting_rows - ending_rows} rows removed total, "
                 f"{100*(starting_rows - ending_rows)/starting_rows:.2f}% of Bronze)")

    # ---- Final schema sanity check ----
    assert (silver["seats_booked"] <= silver["capacity"]).all(), "seats_booked must never exceed capacity"
    assert silver["trip_id"].isin(valid_ids).all(), "all trip_ids must exist in master table"
    assert not silver.duplicated(subset=["trip_id", "booking_date"]).any(), "no duplicate trip/date rows allowed"
    log(report, "\nAll Silver-layer assertions passed: bookings<=capacity, valid trip_ids, no duplicates.")

    silver.to_csv(SILVER_OUTPUT, index=False)
    with open(REPORT_OUTPUT, "w") as f:
        f.write("\n".join(report))

    log(report, f"\nSaved: {SILVER_OUTPUT} ({len(silver)} rows)")
    log(report, f"Saved: {REPORT_OUTPUT}")

    return silver


if __name__ == "__main__":
    transform()