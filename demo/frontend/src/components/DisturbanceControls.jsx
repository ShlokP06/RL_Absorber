/**
 * DisturbanceControls — sliders for G_gas and y_CO2_in.
 * On drag: immediately POSTs /set_disturbance.
 * Renders current live values from WebSocket state alongside slider value.
 */

import { useState, useCallback } from 'react'

const G_MIN = 0.40; const G_MAX = 2.50
const Y_MIN = 0.04; const Y_MAX = 0.22

function pct(v, lo, hi) {
  return ((v - lo) / (hi - lo)) * 100
}

function lerp(t, lo, hi) {
  return lo + t * (hi - lo)
}

export default function DisturbanceControls({ state, onDisturbance, onClear }) {
  const [gVal, setGVal] = useState(0.80)
  const [yVal, setYVal] = useState(0.08)
  const [active, setActive] = useState(false)

  const liveG = state?.rl?.G ?? gVal
  const liveY = state?.rl?.y ?? yVal

  const handleG = useCallback((e) => {
    const v = parseFloat(e.target.value)
    setGVal(v)
    setActive(true)
    onDisturbance(v, yVal)
  }, [yVal, onDisturbance])

  const handleY = useCallback((e) => {
    const v = parseFloat(e.target.value)
    setYVal(v)
    setActive(true)
    onDisturbance(gVal, v)
  }, [gVal, onDisturbance])

  const handleClear = useCallback(() => {
    setActive(false)
    onClear()
  }, [onClear])

  // Colour G bar by stress level
  const gStress = pct(liveG, G_MIN, G_MAX)
  const gColor  = gStress > 75 ? '#ef4444' : gStress > 50 ? '#f97316' : '#22c55e'
  const yStress = pct(liveY, Y_MIN, Y_MAX)
  const yColor  = yStress > 75 ? '#ef4444' : yStress > 50 ? '#f97316' : '#22c55e'

  return (
    <div className="dist-wrap">

      {/* G_gas */}
      <div className="dist-block">
        <div className="title">Gas Flux G_gas</div>
        <div className="dist-value" style={{ color: gColor }}>
          {liveG.toFixed(3)}
          <span style={{ fontSize: 11, color: '#64748b', fontWeight: 400 }}> kg/m²/s</span>
        </div>
        <div style={{ height: 6, background: '#151f38', borderRadius: 3, border: '1px solid #1e2d4a', overflow: 'hidden' }}>
          <div style={{ width: `${gStress}%`, height: '100%', background: gColor, transition: 'all 0.3s', borderRadius: 3 }} />
        </div>
        <input
          type="range"
          min={G_MIN} max={G_MAX} step={0.01}
          value={gVal}
          onChange={handleG}
          style={{ width: '100%', marginTop: 4, accentColor: '#f97316' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#334155' }}>
          <span>{G_MIN}</span><span>{G_MAX}</span>
        </div>
      </div>

      {/* y_CO2_in */}
      <div className="dist-block">
        <div className="title">Inlet CO₂ y_CO₂ᵢₙ</div>
        <div className="dist-value" style={{ color: yColor }}>
          {liveY.toFixed(4)}
          <span style={{ fontSize: 11, color: '#64748b', fontWeight: 400 }}> mol/mol</span>
        </div>
        <div style={{ height: 6, background: '#151f38', borderRadius: 3, border: '1px solid #1e2d4a', overflow: 'hidden' }}>
          <div style={{ width: `${yStress}%`, height: '100%', background: yColor, transition: 'all 0.3s', borderRadius: 3 }} />
        </div>
        <input
          type="range"
          min={Y_MIN} max={Y_MAX} step={0.001}
          value={yVal}
          onChange={handleY}
          style={{ width: '100%', marginTop: 4, accentColor: '#f97316' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#334155' }}>
          <span>{Y_MIN}</span><span>{Y_MAX}</span>
        </div>
      </div>

      {/* Auto override indicator */}
      {active && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#f97316', boxShadow: '0 0 6px #f97316' }} />
          <span style={{ fontSize: 9, color: '#f97316', letterSpacing: '0.08em' }}>MANUAL OVERRIDE ACTIVE</span>
          <button
            onClick={handleClear}
            style={{
              marginLeft: 'auto', padding: '2px 8px',
              background: 'transparent', border: '1px solid #334155',
              color: '#64748b', fontSize: 9, cursor: 'pointer',
              borderRadius: 3, fontFamily: 'inherit'
            }}
          >
            RELEASE
          </button>
        </div>
      )}

      {/* Live OU indication when not overriding */}
      {!active && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e', animation: 'pulse 2s infinite' }} />
          <span style={{ fontSize: 9, color: '#22c55e', letterSpacing: '0.08em' }}>OU PROCESS ACTIVE</span>
        </div>
      )}
    </div>
  )
}
