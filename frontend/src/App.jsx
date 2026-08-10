import { useEffect, useState } from 'react'
import TicketForm from './components/TicketForm.jsx'
import DepartureBoard from './components/DepartureBoard.jsx'
import { fetchRoutes, fetchBusTypes, predictOccupancy } from './api.js'

export default function App() {
  const [routes, setRoutes] = useState([])
  const [busTypes, setBusTypes] = useState([])
  const [loadError, setLoadError] = useState('')

  const [result, setResult] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [predictError, setPredictError] = useState('')

  useEffect(() => {
    Promise.all([fetchRoutes(), fetchBusTypes()])
      .then(([r, b]) => {
        setRoutes(r)
        setBusTypes(b)
      })
      .catch((err) => setLoadError(err.message))
  }, [])

  async function handleSubmit(payload) {
    setPredicting(true)
    setPredictError('')
    setResult(null)
    try {
      const prediction = await predictOccupancy(payload)
      setResult(prediction)
    } catch (err) {
      setPredictError(err.message)
    } finally {
      setPredicting(false)
    }
  }

  return (
    <div className="min-h-screen px-4 py-10 sm:px-8 lg:px-16">
      <header className="mx-auto mb-10 max-w-5xl">
        <p className="font-body text-xs uppercase tracking-[0.2em] text-indigo-night/50">
          Bus Demand &amp; Booking Forecasting
        </p>
        <h1 className="mt-1 font-display text-3xl font-semibold text-indigo-night sm:text-4xl">
          Booking Counter
        </h1>
        <p className="mt-2 max-w-xl font-body text-sm text-indigo-night/60">
          Enter a trip's details to forecast how full it will be by departure —
          same logic a booking clerk would use, run through the trained model.
        </p>
      </header>

      {loadError && (
        <div className="mx-auto mb-6 max-w-5xl rounded-md bg-stamp-red/10 px-4 py-3 font-body text-sm text-stamp-red">
          Couldn't load routes/bus types from the API: {loadError}. Is the FastAPI
          backend running?
        </div>
      )}

      <main className="mx-auto grid max-w-5xl grid-cols-1 overflow-hidden rounded-xl shadow-lg shadow-indigo-night/10 lg:grid-cols-[1.1fr_1px_1fr]">
        <section className="bg-paper-card px-6 py-8 sm:px-8">
          <h2 className="mb-5 font-display text-sm font-semibold uppercase tracking-wide text-indigo-night/70">
            Ticket Counter
          </h2>
          <TicketForm
            routes={routes}
            busTypes={busTypes}
            onSubmit={handleSubmit}
            submitting={predicting}
          />
        </section>

        <div className="perforation-h lg:hidden" />
        <div className="perforation-v hidden lg:block" />

        <section className="bg-indigo-night px-6 py-8 sm:px-8">
          <h2 className="mb-5 font-display text-sm font-semibold uppercase tracking-wide text-paper/60">
            Departure Board
          </h2>
          <DepartureBoard result={result} loading={predicting} error={predictError} />
        </section>
      </main>

      <footer className="mx-auto mt-6 max-w-5xl">
        <p className="font-body text-xs text-indigo-night/40">
          Predictions come from an XGBoost model trained on synthetic-but-grounded
          booking data. Not a substitute for real historical performance.
        </p>
      </footer>
    </div>
  )
}
