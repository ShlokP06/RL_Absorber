/**
 * App — root layout for "The Plant Under Attack" demo.
 *
 * Layout (dark fullscreen):
 *   ┌─ Header ──────────────────────────────────────────────────────────┐
 *   │ PANEL 1      │ PANEL 2 (chart)    │ PANEL 3 (dist) │ PANEL 4 (act) │
 *   │ MEA Column   │                    │ controls       │ agent actions  │
 *   ├─ Controls bar ────────────────────────────────────────────────────┤
 *   └─ Bottom bar (impact counters) ────────────────────────────────────┘
 */

import { useState, useCallback } from 'react'
import { useWebSocket }           from './hooks/useWebSocket'
import MEAColumn                  from './components/MEAColumn'
import TimeSeriesChart            from './components/TimeSeriesChart'
import DisturbanceControls        from './components/DisturbanceControls'
import AgentActionsPanel          from './components/AgentActionsPanel'
import BottomBar                  from './components/BottomBar'

export default function App() {
  const {
    state, history, connected,
    reset, setDisturbance, clearDisturbance, attack, setLambda, freeze, resetImpact,
  } = useWebSocket()

  const [frozen,     setFrozen]     = useState(false)
  const [lambdaVal,  setLambdaVal]  = useState(0.05)
  const [attacking,  setAttacking]  = useState(false)

  // ── Attack button ──────────────────────────────────────────────────
  const handleAttack = useCallback(async () => {
    setAttacking(true)
    await attack()
    setTimeout(() => setAttacking(false), 3000)
  }, [attack])

  // ── Freeze toggle ──────────────────────────────────────────────────
  const handleFreeze = useCallback(async () => {
    const next = !frozen
    setFrozen(next)
    await freeze(next)
  }, [frozen, freeze])

  // ── Lambda slider ──────────────────────────────────────────────────
  const handleLambda = useCallback((e) => {
    const v = parseFloat(e.target.value)
    setLambdaVal(v)
    setLambda(v)
  }, [setLambda])

  // ── Reset ──────────────────────────────────────────────────────────
  const handleReset = useCallback(async () => {
    setFrozen(false)
    await reset()
  }, [reset])

  const isFrozen = frozen || state?.frozen

  return (
    <div className="app">

      {/* ── Header ──────────────────────────────────────────────────── */}
      <div className="header">
        <h1>⚡ The Plant Under Attack — MEA CO₂ Capture RL vs PID</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {isFrozen && (
            <span style={{ fontSize: 10, color: '#facc15', letterSpacing: '0.08em',
              background: 'rgba(250,204,21,0.1)', border: '1px solid rgba(250,204,21,0.3)',
              padding: '2px 8px', borderRadius: 3 }}>
              ⚠ AGENT FROZEN
            </span>
          )}
          <span style={{ fontSize: 10 }}>
            <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`} />
            {connected ? 'LIVE' : 'RECONNECTING…'}
          </span>
          <span style={{ fontSize: 10, color: '#475569' }}>
            λ={lambdaVal.toFixed(3)}
          </span>
        </div>
      </div>

      {/* ── 4-Panel grid ────────────────────────────────────────────── */}
      <div className="panels">

        {/* Panel 1: MEA Column */}
        <div className="panel">
          <div className="panel-title">Absorber Column</div>
          <div className="panel-body">
            <MEAColumn state={state} />
          </div>
        </div>

        {/* Panel 2: Live time-series */}
        <div className="panel">
          <div className="panel-title">
            <span style={{ color: '#22c55e' }}>■ RL</span>
            &nbsp;vs&nbsp;
            <span style={{ color: '#ef4444' }}>■ PID</span>
            &nbsp;— Capture & Energy (live 60 s window)
          </div>
          <div className="panel-body">
            <TimeSeriesChart history={history} />
          </div>
        </div>

        {/* Panel 3: Disturbance controls */}
        <div className="panel">
          <div className="panel-title">Disturbances</div>
          <div className="panel-body">
            <DisturbanceControls
              state={state}
              onDisturbance={setDisturbance}
              onClear={clearDisturbance}
            />
          </div>
        </div>

        {/* Panel 4: Agent actions */}
        <div className="panel">
          <div className="panel-title">Agent Actions (RL solid / PID ghost)</div>
          <div className="panel-body" style={{ position: 'relative' }}>
            {isFrozen && <div className="frozen-badge">FROZEN</div>}
            <AgentActionsPanel state={state} />
          </div>
        </div>

      </div>

      {/* ── Controls bar ────────────────────────────────────────────── */}
      <div className="controls-bar">

        {/* Attack button */}
        <button
          className={`btn danger ${attacking ? 'active' : ''}`}
          onClick={handleAttack}
          disabled={attacking}
        >
          {attacking ? '💥 ATTACKING…' : '⚡ ATTACK THE PLANT'}
        </button>

        {/* Freeze toggle */}
        <button
          className={`btn ${isFrozen ? 'active' : ''}`}
          onClick={handleFreeze}
        >
          {isFrozen ? '▶ UNFREEZE AGENT' : '⏸ FREEZE AGENT'}
        </button>

        {/* Reset */}
        <button className="btn reset" onClick={handleReset}>
          ↺ RESET SIM
        </button>

        {/* Reset meters */}
        <button className="btn reset" onClick={resetImpact}>
          ⟳ RESET METERS
        </button>

        <div style={{ width: 1, height: 24, background: '#1e2d4a', flexShrink: 0 }} />

        {/* Lambda slider */}
        <div className="lambda-group">
          <div className="lambda-label">
            <span>Pareto λ (energy weight)</span>
            <span>{lambdaVal.toFixed(3)}</span>
          </div>
          <input
            type="range"
            className="lambda-range"
            min={0} max={0.08} step={0.005}
            value={lambdaVal}
            onChange={handleLambda}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#334155' }}>
            <span>capture ←</span><span>→ efficient</span>
          </div>
        </div>

        {/* Pareto frontier mini scatter */}
        <ParetoMini history={history} lambda={lambdaVal} />

      </div>

      {/* ── Bottom bar ──────────────────────────────────────────────── */}
      <BottomBar state={state} />

    </div>
  )
}

// ── Pareto frontier mini-scatter (SVG) ────────────────────────────────────

function ParetoMini({ history, lambda }) {
  if (!history?.length) return null

  const W = 120; const H = 52
  const padX = 20; const padY = 8
  const plotW = W - padX - 4
  const plotH = H - padY - 12

  // Last 60 points
  const pts = history.slice(-60)

  const capLo = 60, capHi = 100
  const engLo = 2.5, engHi = 7.0

  const cx = (cap) => padX + ((cap - capLo) / (capHi - capLo)) * plotW
  const cy = (eng) => padY + plotH - ((eng - engLo) / (engHi - engLo)) * plotH

  return (
    <div style={{ flexShrink: 0 }}>
      <div style={{ fontSize: 9, color: '#a855f7', marginBottom: 2, letterSpacing: '0.06em' }}>
        PARETO FRONTIER
      </div>
      <svg width={W} height={H} style={{ background: '#0f1729', borderRadius: 4, border: '1px solid #1e2d4a' }}>
        {/* Axes */}
        <line x1={padX} y1={padY} x2={padX} y2={padY + plotH} stroke="#1e2d4a" />
        <line x1={padX} y1={padY + plotH} x2={W - 4} y2={padY + plotH} stroke="#1e2d4a" />

        {/* Points */}
        {pts.map((s, i) => (
          <g key={i}>
            <circle cx={cx(s.rl?.cap ?? 85)} cy={cy(s.rl?.eng ?? 4)} r={1.5} fill="#22c55e" opacity={0.6} />
            <circle cx={cx(s.pid?.cap ?? 85)} cy={cy(s.pid?.eng ?? 4)} r={1.5} fill="#ef4444" opacity={0.4} />
          </g>
        ))}

        {/* Labels */}
        <text x={padX} y={H - 2} fontSize={7} fill="#475569">Cap %</text>
        <text x={2} y={padY + 4} fontSize={7} fill="#475569" transform={`rotate(-90, 6, ${padY + plotH / 2})`}>GJ/t</text>
      </svg>
    </div>
  )
}
