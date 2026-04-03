/**
 * TimeSeriesChart — live dual-axis chart: capture rate + energy consumption.
 *
 * Left Y-axis:  capture rate [%], RL (green) vs PID (red)
 * Right Y-axis: energy [GJ/t CO₂],  RL (green dashed) vs PID (red dashed)
 * Reference line at 90% capture setpoint.
 * Scrolling 60-second window (120 data points @ 500 ms).
 */

import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import { useMemo } from 'react'

const CAPTURE_DOMAIN = [50, 100]
const ENERGY_DOMAIN  = [2.5, 7.0]

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#0f1729', border: '1px solid #1e2d4a',
      padding: '8px 10px', borderRadius: 4, fontSize: 11
    }}>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color, lineHeight: '1.6' }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : '—'}
          {p.dataKey.includes('cap') ? ' %' : ' GJ/t'}
        </div>
      ))}
    </div>
  )
}

export default function TimeSeriesChart({ history }) {
  const data = useMemo(() => {
    if (!history?.length) return []
    return history.map((snap, i) => ({
      i,
      rl_cap:  snap.rl?.cap  ?? null,
      pid_cap: snap.pid?.cap ?? null,
      rl_eng:  snap.rl?.eng  ?? null,
      pid_eng: snap.pid?.eng ?? null,
      G:       snap.rl?.G    ?? null,
    }))
  }, [history])

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={data}
          margin={{ top: 8, right: 48, bottom: 4, left: 32 }}
        >
          <CartesianGrid stroke="#1e2d4a" strokeDasharray="3 3" />

          <XAxis
            dataKey="i"
            tick={{ fill: '#64748b', fontSize: 9 }}
            tickLine={false}
            label={{ value: 'steps (×0.5 s)', position: 'insideBottomRight', offset: -4, fill: '#475569', fontSize: 9 }}
          />

          {/* Left: Capture rate */}
          <YAxis
            yAxisId="cap"
            domain={CAPTURE_DOMAIN}
            tick={{ fill: '#64748b', fontSize: 9 }}
            tickLine={false}
            label={{ value: 'Capture [%]', angle: -90, position: 'insideLeft', offset: 10, fill: '#64748b', fontSize: 9 }}
          />

          {/* Right: Energy */}
          <YAxis
            yAxisId="eng"
            orientation="right"
            domain={ENERGY_DOMAIN}
            tick={{ fill: '#64748b', fontSize: 9 }}
            tickLine={false}
            label={{ value: 'Energy [GJ/t]', angle: 90, position: 'insideRight', offset: 10, fill: '#64748b', fontSize: 9 }}
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
            iconType="plainline"
          />

          {/* 90% setpoint reference line */}
          <ReferenceLine
            yAxisId="cap"
            y={90}
            stroke="#facc15"
            strokeDasharray="6 3"
            strokeWidth={1}
            label={{ value: '90% SP', position: 'insideTopRight', fill: '#facc15', fontSize: 9 }}
          />

          {/* Capture lines */}
          <Line
            yAxisId="cap"
            dataKey="rl_cap"
            name="RL Capture %"
            stroke="#22c55e"
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
            connectNulls
          />
          <Line
            yAxisId="cap"
            dataKey="pid_cap"
            name="PID Capture %"
            stroke="#ef4444"
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
            connectNulls
          />

          {/* Energy lines (dashed) */}
          <Line
            yAxisId="eng"
            dataKey="rl_eng"
            name="RL Energy GJ/t"
            stroke="#86efac"
            dot={false}
            strokeWidth={1.5}
            strokeDasharray="5 3"
            isAnimationActive={false}
            connectNulls
          />
          <Line
            yAxisId="eng"
            dataKey="pid_eng"
            name="PID Energy GJ/t"
            stroke="#fca5a5"
            dot={false}
            strokeWidth={1.5}
            strokeDasharray="5 3"
            isAnimationActive={false}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
