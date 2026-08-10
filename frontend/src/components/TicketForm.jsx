import { useEffect, useMemo, useState } from 'react'

const inputClass =
  'w-full rounded-md border border-indigo-night/15 bg-white px-3 py-2 font-body text-sm text-indigo-night placeholder:text-indigo-night/30 focus:border-amber-board focus:outline-none'

const labelClass = 'font-body text-xs font-medium uppercase tracking-wide text-indigo-night/60'

function Field({ label, children }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className={labelClass}>{label}</span>
      {children}
    </label>
  )
}

export default function TicketForm({ routes, busTypes, onSubmit, submitting }) {
  const today = new Date().toISOString().slice(0, 10)

  const [routeKey, setRouteKey] = useState('')
  const [busType, setBusType] = useState('')
  const [capacity, setCapacity] = useState('')
  const [price, setPrice] = useState('')
  const [departureDate, setDepartureDate] = useState('')
  const [bookingOpenDate, setBookingOpenDate] = useState('')
  const [asOfDate, setAsOfDate] = useState(today)
  const [seatsBooked, setSeatsBooked] = useState('')
  const [formError, setFormError] = useState('')

  const selectedRoute = useMemo(
    () => routes.find((r) => `${r.origin}__${r.destination}` === routeKey),
    [routes, routeKey],
  )
  const selectedBusType = useMemo(
    () => busTypes.find((b) => b.bus_type === busType),
    [busTypes, busType],
  )

  useEffect(() => {
    if (selectedBusType) {
      const [lo, hi] = selectedBusType.capacity_range
      setCapacity(String(Math.round((lo + hi) / 2)))
    }
  }, [selectedBusType])

  function validate() {
    if (!selectedRoute) return 'Pick a route.'
    if (!selectedBusType) return 'Pick a bus type.'
    if (!capacity || Number(capacity) <= 0) return 'Capacity must be a positive number.'
    if (!price || Number(price) <= 0) return 'Enter a valid ticket price.'
    if (!departureDate) return 'Pick a departure date.'
    if (!bookingOpenDate) return 'Pick a booking-open date.'
    if (bookingOpenDate >= departureDate) return 'Booking must open before departure.'
    if (seatsBooked === '' || Number(seatsBooked) < 0) return 'Enter seats booked so far (0 or more).'
    if (Number(seatsBooked) > Number(capacity)) return 'Seats booked cannot exceed capacity.'
    return ''
  }

  function handleSubmit(e) {
    e.preventDefault()
    const err = validate()
    if (err) {
      setFormError(err)
      return
    }
    setFormError('')
    onSubmit({
      origin: selectedRoute.origin,
      destination: selectedRoute.destination,
      bus_type: busType,
      capacity: Number(capacity),
      base_price_inr: Number(price),
      departure_date: departureDate,
      booking_open_date: bookingOpenDate,
      as_of_date: asOfDate,
      current_seats_booked: Number(seatsBooked),
    })
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Field label="Route">
          <select
            className={inputClass}
            value={routeKey}
            onChange={(e) => setRouteKey(e.target.value)}
          >
            <option value="">Select a route</option>
            {routes.map((r) => (
              <option key={`${r.origin}__${r.destination}`} value={`${r.origin}__${r.destination}`}>
                {r.origin} → {r.destination} · {r.distance_km} km
              </option>
            ))}
          </select>
        </Field>

        <Field label="Bus type">
          <select className={inputClass} value={busType} onChange={(e) => setBusType(e.target.value)}>
            <option value="">Select a bus type</option>
            {busTypes.map((b) => (
              <option key={b.bus_type} value={b.bus_type}>
                {b.bus_type}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Capacity (seats)">
          <input
            type="number"
            min="1"
            className={inputClass}
            value={capacity}
            onChange={(e) => setCapacity(e.target.value)}
            placeholder="e.g. 32"
          />
        </Field>

        <Field label="Ticket price (Rs)">
          <input
            type="number"
            min="1"
            className={inputClass}
            value={price}
            onChange={(e) => setPrice(e.target.value)}
            placeholder="e.g. 800"
          />
        </Field>

        <Field label="Departure date">
          <input
            type="date"
            className={inputClass}
            value={departureDate}
            onChange={(e) => setDepartureDate(e.target.value)}
          />
        </Field>

        <Field label="Booking opens">
          <input
            type="date"
            className={inputClass}
            value={bookingOpenDate}
            onChange={(e) => setBookingOpenDate(e.target.value)}
          />
        </Field>

        <Field label="As of date (today)">
          <input
            type="date"
            className={inputClass}
            value={asOfDate}
            onChange={(e) => setAsOfDate(e.target.value)}
          />
        </Field>

        <Field label="Seats booked so far">
          <input
            type="number"
            min="0"
            className={inputClass}
            value={seatsBooked}
            onChange={(e) => setSeatsBooked(e.target.value)}
            placeholder="e.g. 18"
          />
        </Field>
      </div>

      {formError && (
        <p className="rounded-md bg-stamp-red/10 px-3 py-2 font-body text-sm text-stamp-red">
          {formError}
        </p>
      )}

      <button
        type="submit"
        disabled={submitting}
        className="mt-1 w-full rounded-md bg-indigo-night px-4 py-3 font-display text-sm font-semibold uppercase tracking-wide text-paper transition hover:bg-indigo-deep disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
      >
        {submitting ? 'Punching ticket...' : 'Forecast this trip'}
      </button>
    </form>
  )
}
