/**
 * BottomBar — live impact counter with animated ticking numbers.
 *
 * Numbers smoothly interpolate toward their target value via
 * a spring-like approach (lerp each render frame) so they visually
 * "tick up" rather than jumping.
 */

import { useRef, useState, useEffect } from 'react'

function useAnimatedValue(target, speed = 0.12) {
  const [display, setDisplay] = useState(target ?? 0)
  const current = useRef(target ?? 0)
  const rafRef  = useRef(null)

  useEffect(() => {
    if (target === null || target === undefined) return

    const animate = () => {
      const diff = target - current.current
      if (Math.abs(diff) < 0.001) {
        current.current = target
        setDisplay(target)
        return
      }
      current.current += diff * speed
      setDisplay(current.current)
      rafRef.current = requestAnimationFrame(animate)
    }

    cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(rafRef.current)
  }, [target, speed])

  return display
}

function AnimatedMetric({ rawVal, decimals = 1, suffix = '', label, className = 'neutral' }) {
  const val = useAnimatedValue(rawVal)

  const formatted = (() => {
    if (rawVal === undefined || rawVal === null) return '—'
    const abs = Math.abs(val)
    if (abs >= 1000) return `${(val / 1000).toFixed(1)}k`
    return val.toFixed(decimals)
  })()

  return (
    <div className={`metric ${className}`}>
      <div className="val">{formatted}{suffix}</div>
      <div className="label">{label}</div>
    </div>
  )
}

export default function BottomBar({ state }) {
  const imp = state?.impact ?? {}

  const co2Rl      = imp.co2_captured_rl_t  ?? 0
  const co2Delta   = imp.co2_delta_t        ?? 0
  const kwhSaved   = imp.energy_kwh_saved    ?? 0
  const moneySaved = imp.money_saved_usd     ?? 0
  const trees      = imp.trees_equivalent    ?? 0

  const rlCap  = state?.rl?.cap  ?? 0
  const pidCap = state?.pid?.cap ?? 0
  const rlEng  = state?.rl?.eng  ?? 0
  const pidEng = state?.pid?.eng ?? 0

  const capDelta  = rlCap - pidCap
  const engDelta  = pidEng - rlEng   // positive = RL cheaper

  return (
    <div className="bottom-bar">

      <AnimatedMetric
        rawVal={co2Rl * 1000}
        decimals={2}
        label="kg CO₂ (RL)"
        className="neutral"
      />

      <div style={{ width: 1, height: 28, background: '#1e2d4a' }} />

      <AnimatedMetric
        rawVal={co2Delta * 1000}
        decimals={2}
        suffix=" kg"
        label="extra CO₂ vs PID"
        className={co2Delta >= 0 ? 'positive' : 'negative'}
      />

      <div style={{ width: 1, height: 28, background: '#1e2d4a' }} />

      <AnimatedMetric
        rawVal={kwhSaved}
        decimals={2}
        suffix=" kWh"
        label="energy saved vs PID"
        className={kwhSaved >= 0 ? 'positive' : 'negative'}
      />

      <div style={{ width: 1, height: 28, background: '#1e2d4a' }} />

      <AnimatedMetric
        rawVal={moneySaved}
        decimals={2}
        suffix=" USD"
        label="cost saved"
        className={moneySaved >= 0 ? 'gold' : 'negative'}
      />

      <div style={{ width: 1, height: 28, background: '#1e2d4a' }} />

      <AnimatedMetric
        rawVal={trees}
        decimals={1}
        label="trees equivalent"
        className="tree"
      />

      <div style={{ width: 1, height: 28, background: '#1e2d4a' }} />

      {/* Live capture comparison with delta */}
      <div className="metric">
        <div style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
          <span style={{ color: '#22c55e', fontSize: 15, fontWeight: 800 }}>
            {rlCap.toFixed(1)}%
          </span>
          <span style={{
            fontSize: 10, fontWeight: 700,
            color: capDelta >= 0 ? '#22c55e' : '#ef4444',
          }}>
            {capDelta >= 0 ? '+' : ''}{capDelta.toFixed(1)}
          </span>
          <span style={{ color: '#ef4444', fontSize: 15, fontWeight: 800 }}>
            {pidCap.toFixed(1)}%
          </span>
        </div>
        <div className="label">capture RL vs PID</div>
      </div>

      <div style={{ width: 1, height: 28, background: '#1e2d4a' }} />

      {/* Live energy comparison with delta */}
      <div className="metric">
        <div style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
          <span style={{ color: '#86efac', fontSize: 15, fontWeight: 800 }}>
            {rlEng.toFixed(2)}
          </span>
          <span style={{
            fontSize: 10, fontWeight: 700,
            color: engDelta >= 0 ? '#22c55e' : '#ef4444',
          }}>
            {engDelta >= 0 ? '-' : '+'}{Math.abs(engDelta).toFixed(2)}
          </span>
          <span style={{ color: '#fca5a5', fontSize: 15, fontWeight: 800 }}>
            {pidEng.toFixed(2)}
          </span>
        </div>
        <div className="label">energy GJ/t RL vs PID</div>
      </div>

    </div>
  )
}
