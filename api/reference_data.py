"""
Reference "skeleton" data grounded in real Indian intercity bus market facts:
- Real city pairs actually served by operators (RedBus/AbhiBus corridors)
- Realistic distances (road km)
- Realistic base one-way fares by bus type, benchmarked against known
  2024-2025 market pricing for these corridors (AC sleeper/seater/Volvo etc.)
- Realistic bus types and seat capacities used by Indian operators

This is the "real" half of the synthetic-but-grounded dataset. Only the
booking TRAJECTORIES (who booked when) are simulated -- everything here
(route existed, approx distance, approx fare band, bus type/capacity) is
real-world grounded.
"""

# (origin, destination, distance_km, popularity_tier)
# popularity_tier: 1 = very high demand corridor, 2 = high, 3 = medium
ROUTES = [
    ("Mumbai", "Pune", 150, 1),
    ("Mumbai", "Nashik", 170, 2),
    ("Mumbai", "Goa", 590, 2),
    ("Mumbai", "Shirdi", 250, 2),
    ("Pune", "Nashik", 210, 3),
    ("Pune", "Kolhapur", 230, 2),
    ("Bangalore", "Chennai", 350, 1),
    ("Bangalore", "Hyderabad", 570, 2),
    ("Bangalore", "Coimbatore", 365, 2),
    ("Bangalore", "Goa", 560, 3),
    ("Bangalore", "Mysore", 145, 1),
    ("Chennai", "Coimbatore", 500, 2),
    ("Chennai", "Madurai", 460, 2),
    ("Hyderabad", "Vijayawada", 275, 2),
    ("Hyderabad", "Bangalore", 570, 2),
    ("Delhi", "Jaipur", 280, 1),
    ("Delhi", "Chandigarh", 250, 2),
    ("Delhi", "Agra", 230, 2),
    ("Delhi", "Dehradun", 250, 2),
    ("Jaipur", "Udaipur", 400, 3),
    ("Ahmedabad", "Mumbai", 525, 2),
    ("Ahmedabad", "Surat", 265, 1),
    ("Surat", "Mumbai", 285, 2),
    ("Kolkata", "Digha", 185, 3),
    ("Kochi", "Bangalore", 550, 3),
]

# Bus type -> (capacity_range, base_fare_per_km_INR)
BUS_TYPES = {
    "Non-AC Seater":      {"capacity": (48, 56), "fare_per_km": (0.9, 1.2)},
    "AC Seater":          {"capacity": (40, 45), "fare_per_km": (1.5, 1.9)},
    "Non-AC Sleeper":     {"capacity": (30, 36), "fare_per_km": (1.3, 1.6)},
    "AC Sleeper":         {"capacity": (28, 34), "fare_per_km": (2.0, 2.6)},
    "Volvo AC Semi-Sleeper": {"capacity": (38, 42), "fare_per_km": (1.8, 2.3)},
}

# Major India-wide / regional holidays relevant to travel demand spikes (2025-2026)
HOLIDAYS = {
    "2025-08-15", "2025-08-27",  # Independence Day, Ganesh Chaturthi
    "2025-10-02", "2025-10-20", "2025-10-21",  # Gandhi Jayanti, Diwali window
    "2025-11-01",
    "2025-12-25",
    "2026-01-01", "2026-01-14", "2026-01-26",  # New Year, Makar Sankranti, Republic Day
    "2026-03-04",  # Holi (approx)
    "2026-04-14",
}


'''Step 1 — Collect/reference real-world information

First define the things we know realistically:

Routes
Distances
Bus types
Capacities
Typical fares
Route popularity
Weekdays
Holidays

This becomes your reference data.'''