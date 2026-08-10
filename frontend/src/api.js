const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function handleResponse(res) {
  if (res.ok) return res.json()
  const body = await res.json().catch(() => null)
  let message = `Request failed (${res.status})`
  if (body?.detail) {
    if (Array.isArray(body.detail)) {
      message = body.detail
        .map((d) => `${(d.loc || []).slice(-1)[0] || 'field'}: ${d.msg}`)
        .join('; ')
    } else {
      message = body.detail
    }
  }
  const err = new Error(message)
  err.status = res.status
  throw err
}

export async function fetchRoutes() {
  const res = await fetch(`${API_BASE}/routes`)
  return handleResponse(res)
}

export async function fetchBusTypes() {
  const res = await fetch(`${API_BASE}/bus-types`)
  return handleResponse(res)
}

export async function predictOccupancy(payload) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return handleResponse(res)
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`)
  return handleResponse(res)
}
