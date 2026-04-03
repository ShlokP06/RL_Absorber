import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = 'ws://localhost:8000/stream'
const API_URL = 'http://localhost:8000'

export function useWebSocket() {
  const [state, setState] = useState(null)
  const [history, setHistory] = useState([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const reconnectRef = useRef(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      if (reconnectRef.current) {
        clearTimeout(reconnectRef.current)
        reconnectRef.current = null
      }
    }

    ws.onmessage = (e) => {
      try {
        const snap = JSON.parse(e.data)
        setState(snap)
        setHistory((prev) => {
          const next = [...prev, snap]
          return next.slice(-120)  // keep 60s @ 500ms
        })
      } catch (_) {}
    }

    ws.onerror = () => ws.close()

    ws.onclose = () => {
      setConnected(false)
      reconnectRef.current = setTimeout(connect, 2000)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectRef.current) clearTimeout(reconnectRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  // ── API helpers ──────────────────────────────────────────────────────────

  const post = useCallback(async (path, body = {}) => {
    try {
      const res = await fetch(`${API_URL}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      return res.ok ? await res.json() : null
    } catch (_) {
      return null
    }
  }, [])

  const reset = useCallback(async () => {
    const snap = await post('/reset')
    if (snap) {
      setState(snap)
      setHistory([snap])
    }
  }, [post])

  const setDisturbance = useCallback(
    (G_gas, y_CO2_in) => post('/set_disturbance', { G_gas, y_CO2_in }),
    [post]
  )

  const clearDisturbance = useCallback(() => post('/clear_disturbance'), [post])

  const attack = useCallback(() => post('/attack'), [post])

  const setLambda = useCallback(
    (lambda_energy) => post('/set_lambda', { lambda_energy }),
    [post]
  )

  const freeze = useCallback(
    (frozen) => post('/freeze', { frozen }),
    [post]
  )

  const resetImpact = useCallback(() => post('/reset_impact'), [post])

  return {
    state,
    history,
    connected,
    reset,
    setDisturbance,
    clearDisturbance,
    attack,
    setLambda,
    freeze,
    resetImpact,
  }
}
