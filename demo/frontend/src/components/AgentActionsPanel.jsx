/**
 * AgentActionsPanel — four live bar indicators showing RL agent's
 * current actuator values: L_liq, α_lean, T_in, T_ic.
 *
 * Also shows PID values as ghost bars for direct comparison.
 */

const VARS = [
  { key: 'L',  pidKey: 'L',  label: 'L_liq',   unit: 'kg/m²/s', lo: 2.0,  hi: 12.0, color: '#3b82f6' },
  { key: 'al', pidKey: 'al', label: 'α_lean',   unit: 'mol/mol', lo: 0.18, hi: 0.38, color: '#a855f7' },
  { key: 'T',  pidKey: 'T',  label: 'T_L_in',   unit: '°C',      lo: 30.0, hi: 55.0, color: '#f97316' },
  { key: 'ic', pidKey: 'ic', label: 'T_ic',     unit: '°C',      lo: 25.0, hi: 50.0, color: '#06b6d4' },
]

function pct(v, lo, hi) {
  return Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100))
}

function ActionBar({ spec, rlVal, pidVal }) {
  const rlPct  = pct(rlVal ?? spec.lo, spec.lo, spec.hi)
  const pidPct = pct(pidVal ?? spec.lo, spec.lo, spec.hi)

  return (
    <div className="action-bar-row">
      <div className="action-bar-label">
        <span>{spec.label}</span>
        <span className="val">{(rlVal ?? 0).toFixed(spec.lo < 1 ? 4 : 2)}&nbsp;{spec.unit}</span>
      </div>

      {/* RL bar */}
      <div className="action-bar-track">
        <div
          className="action-bar-fill"
          style={{ width: `${rlPct}%`, background: spec.color, opacity: 0.9 }}
        />
      </div>

      {/* PID ghost bar */}
      <div style={{ height: 4, background: '#0f1729', borderRadius: 2, border: '1px solid #1e2d4a', overflow: 'hidden', marginTop: 2 }}>
        <div style={{
          height: '100%', width: `${pidPct}%`,
          background: spec.color, opacity: 0.3,
          borderRadius: 2, transition: 'width 0.4s ease',
        }} />
      </div>
    </div>
  )
}

export default function AgentActionsPanel({ state }) {
  const rl  = state?.rl  ?? {}
  const pid = state?.pid ?? {}
  const frozen = state?.frozen ?? false

  return (
    <div className="actions-wrap">
      {frozen && (
        <div style={{
          fontSize: 9, color: '#facc15', letterSpacing: '0.08em',
          background: 'rgba(250,204,21,0.08)', border: '1px solid rgba(250,204,21,0.3)',
          padding: '4px 6px', borderRadius: 3, textAlign: 'center',
        }}>
          ⚠ AGENT FROZEN — PID ACTIVE
        </div>
      )}

      {VARS.map((spec) => (
        <ActionBar
          key={spec.key}
          spec={spec}
          rlVal={rl[spec.key]}
          pidVal={pid[spec.pidKey]}
        />
      ))}

      {/* Action deltas */}
      {rl.action && (
        <div style={{ marginTop: 6, borderTop: '1px solid #1e2d4a', paddingTop: 8 }}>
          <div style={{ fontSize: 9, color: '#475569', marginBottom: 4, letterSpacing: '0.08em' }}>
            LAST Δ ACTION
          </div>
          {['ΔL', 'Δα', 'ΔT', 'ΔTᵢc'].map((label, i) => {
            const v = rl.action[i] ?? 0
            const pos = v >= 0
            return (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 3 }}>
                <span style={{ fontSize: 9, color: '#475569', width: 24 }}>{label}</span>
                <div style={{
                  flex: 1, height: 5, background: '#151f38',
                  borderRadius: 2, border: '1px solid #1e2d4a',
                  overflow: 'visible', position: 'relative',
                }}>
                  <div style={{
                    position: 'absolute',
                    left: pos ? '50%' : `${50 + v * 50}%`,
                    width: `${Math.abs(v) * 50}%`,
                    height: '100%',
                    background: pos ? '#22c55e' : '#ef4444',
                    borderRadius: 2,
                  }} />
                </div>
                <span style={{ fontSize: 9, color: pos ? '#22c55e' : '#ef4444', width: 36, textAlign: 'right' }}>
                  {v >= 0 ? '+' : ''}{v.toFixed(3)}
                </span>
              </div>
            )
          })}
        </div>
      )}

      {/* Flood fraction */}
      <div style={{ marginTop: 6, borderTop: '1px solid #1e2d4a', paddingTop: 8 }}>
        <div style={{ fontSize: 9, color: '#475569', marginBottom: 4, letterSpacing: '0.08em' }}>FLOOD FRACTION</div>
        {(() => {
          const ff = rl.ff ?? 0
          const ffPct = Math.min(100, (ff / 0.79) * 100)
          const ffColor = ff > 0.70 ? '#ef4444' : ff > 0.60 ? '#f97316' : '#3b82f6'
          return (
            <div>
              <div style={{ height: 8, background: '#151f38', borderRadius: 3, border: '1px solid #1e2d4a', overflow: 'hidden' }}>
                <div style={{ width: `${ffPct}%`, height: '100%', background: ffColor, transition: 'width 0.4s', borderRadius: 3 }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, marginTop: 2 }}>
                <span style={{ color: '#475569' }}>0</span>
                <span style={{ color: ffColor, fontWeight: 700 }}>{ff.toFixed(3)}</span>
                <span style={{ color: '#ef4444' }}>0.79 ⚠</span>
              </div>
            </div>
          )
        })()}
      </div>
    </div>
  )
}
