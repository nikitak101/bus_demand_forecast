const TIER_STYLES = {
  HIGH: { text: 'text-stamp-red', bg: 'bg-stamp-red/10', ring: 'ring-stamp-red/30' },
  MEDIUM: { text: 'text-amber-board', bg: 'bg-amber-board/10', ring: 'ring-amber-board/40' },
  LOW: { text: 'text-teal-promo', bg: 'bg-teal-promo/10', ring: 'ring-teal-promo/30' },
}

function DigitBlock({ value, label }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <span
        key={value}
        className="board-flicker font-mono text-3xl font-semibold text-amber-bright sm:text-4xl"
      >
        {value}
      </span>
      <span className="font-body text-[11px] uppercase tracking-wide text-paper/50">{label}</span>
    </div>
  )
}

export default function DepartureBoard({ result, loading, error }) {
  if (loading) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 rounded-lg bg-indigo-night px-6 py-14 text-center">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-amber-board border-t-transparent" />
        <p className="font-body text-sm text-paper/60">Checking the booking board...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full flex-col justify-center gap-2 rounded-lg bg-indigo-night px-6 py-10">
        <p className="font-display text-sm font-semibold uppercase tracking-wide text-stamp-red">
          Couldn't reach the counter
        </p>
        <p className="font-body text-sm text-paper/70">{error}</p>
        <p className="font-body text-xs text-paper/40">
          Make sure the FastAPI backend is running at the configured API URL.
        </p>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 rounded-lg bg-indigo-night px-6 py-14 text-center">
        <p className="font-display text-lg font-semibold text-paper/80">Board is empty</p>
        <p className="max-w-[24ch] font-body text-sm text-paper/45">
          Fill in a trip on the left and punch "Forecast this trip" to see the predicted occupancy.
        </p>
      </div>
    )
  }

  const tier = TIER_STYLES[result.demand_tier] || TIER_STYLES.MEDIUM

  return (
    <div className="flex h-full flex-col gap-6 rounded-lg bg-indigo-night px-6 py-8">
      <div className="flex items-center justify-between">
        <span className="font-body text-xs uppercase tracking-widest text-paper/40">
          Predicted final occupancy
        </span>
        <span
          className={`rounded-full px-3 py-1 font-display text-xs font-semibold uppercase tracking-wide ring-1 ${tier.text} ${tier.bg} ${tier.ring}`}
        >
          {result.demand_tier} demand
        </span>
      </div>

      <div className="flex items-baseline gap-2">
        <span
          key={result.predicted_final_occupancy_pct}
          className="board-flicker font-mono text-6xl font-semibold text-amber-board sm:text-7xl"
        >
          {result.predicted_final_occupancy_pct.toFixed(1)}
        </span>
        <span className="font-mono text-2xl text-amber-board/70">%</span>
      </div>

      <div className="grid grid-cols-3 gap-2 border-t border-paper/10 pt-5">
        <DigitBlock value={result.predicted_final_seats} label="Seats filled" />
        <DigitBlock value={result.capacity} label="Capacity" />
        <DigitBlock value={result.days_left} label="Days left" />
      </div>

      <div className="rounded-md bg-paper/5 px-4 py-3">
        <p className="font-body text-sm leading-relaxed text-paper/80">{result.recommendation}</p>
      </div>

      {result.warnings?.length > 0 && (
        <ul className="flex flex-col gap-1.5 border-t border-paper/10 pt-4">
          {result.warnings.map((w, i) => (
            <li key={i} className="flex gap-2 font-body text-xs text-paper/45">
              <span className="text-amber-board/60">.</span>
              <span>{w}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
