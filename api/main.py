"""
Step 11 - FastAPI backend for the Bus Demand & Booking Forecasting model.

    React frontend (Step 12, not built yet)
         v  HTTP POST /predict
    FastAPI  ---->  predict.py  ---->  model_xgboost_final.pkl
         ^
         |  GET /routes, /bus-types  (populate dropdowns)
         |  GET /health              (liveness check)

This file is a thin HTTP wrapper around predict.py -- it does NOT duplicate
any feature-engineering or modeling logic. That separation matters: the
prediction logic can be unit-tested and reused (e.g. from a batch script)
independently of the web layer.

Run locally:
    uvicorn main:app --reload --port 8000

Docs auto-generated at:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from datetime import date
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict

from predict import predict as run_prediction, _load_artifacts
from reference_data import ROUTES, BUS_TYPES


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern replacement for the deprecated @app.on_event("startup") pattern.
    Code before `yield` runs once at server startup (here: warm the model
    cache so the first real request doesn't pay the disk-load cost); code
    after `yield` would run at shutdown (nothing needed here).
    """
    _load_artifacts()
    print("Model artifacts loaded and cached.")
    yield


app = FastAPI(
    title="Bus Demand & Booking Forecasting API",
    description="Predicts final occupancy for a bus trip given its route, "
                 "pricing, and current booking progress.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow a local React dev server (Step 12) to call this API from the browser.
# Tighten this to your real frontend's origin before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
       "http://localhost:3000", 
       "http://localhost:5173",
       "https://bus-demand-forecast.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class RecentSeatsBooked(BaseModel):
    """Optional real booking history for accurate lag/velocity features."""
    model_config = ConfigDict(populate_by_name=True)

    minus_1: int = Field(..., alias="-1", description="Seats booked 1 day ago")
    minus_3: int = Field(..., alias="-3", description="Seats booked 3 days ago")
    minus_7: int = Field(..., alias="-7", description="Seats booked 7 days ago")


class TripPredictionRequest(BaseModel):
    origin: str = Field(..., examples=["Mumbai"])
    destination: str = Field(..., examples=["Pune"])
    bus_type: str = Field(..., examples=["AC Sleeper"])
    capacity: int = Field(..., gt=0, le=80, examples=[32])
    base_price_inr: float = Field(..., gt=0, examples=[800])
    departure_date: date = Field(..., examples=["2026-08-30"])
    booking_open_date: date = Field(..., examples=["2026-08-10"])
    as_of_date: Optional[date] = Field(
        None, description="Defaults to today if omitted", examples=["2026-08-20"]
    )
    current_seats_booked: int = Field(..., ge=0, examples=[18])
    recent_seats_booked: Optional[Dict[str, int]] = Field(
        None, description='Optional real history, e.g. {"-1": 21, "-3": 18, "-7": 12}'
    )

    @field_validator("bus_type")
    @classmethod
    def bus_type_must_be_known(cls, v):
        if v not in BUS_TYPES:
            raise ValueError(f"bus_type must be one of {list(BUS_TYPES.keys())}")
        return v

    @field_validator("booking_open_date")
    @classmethod
    def booking_open_before_departure(cls, v, info):
        departure = info.data.get("departure_date")
        if departure and v >= departure:
            raise ValueError("booking_open_date must be before departure_date")
        return v

    @field_validator("current_seats_booked")
    @classmethod
    def seats_within_capacity(cls, v, info):
        capacity = info.data.get("capacity")
        if capacity is not None and v > capacity:
            raise ValueError("current_seats_booked cannot exceed capacity")
        return v


class TripPredictionResponse(BaseModel):
    predicted_final_occupancy_pct: float
    predicted_final_seats: int
    capacity: int
    demand_tier: str
    recommendation: str
    days_left: int
    popularity_tier_used: int
    route_known_in_reference_data: bool
    lag_feature_source: str
    warnings: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/routes")
def list_routes():
    """Real route skeleton -- for populating a frontend dropdown."""
    return [
        {"origin": o, "destination": d, "distance_km": dist, "popularity_tier": tier}
        for o, d, dist, tier in ROUTES
    ]


@app.get("/bus-types")
def list_bus_types():
    return [
        {"bus_type": name, "capacity_range": info["capacity"],
         "fare_per_km_range": info["fare_per_km"]}
        for name, info in BUS_TYPES.items()
    ]


@app.post("/predict", response_model=TripPredictionResponse)
def predict_occupancy(req: TripPredictionRequest):
    trip_input = req.model_dump(exclude_none=True, mode="json")
    if "as_of_date" in trip_input and trip_input["as_of_date"] is None:
        del trip_input["as_of_date"]

    try:
        result = run_prediction(trip_input)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        # a feature the model expects wasn't built -- this is a bug in the
        # feature pipeline, not bad user input, so 500 not 400
        raise HTTPException(status_code=500, detail=f"Internal feature-building error: {e}")

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)