/**
 * MEAColumn — animated D3 absorber column.
 *
 * Visuals:
 *  - Temperature heatmap gradient (blue→red, smooth 400 ms D3 transition)
 *  - Absorption front marker (yellow dashed, animates up/down)
 *  - Rising CO₂ bubble particles (spawned continuously, speed scales with G_gas)
 *  - Capture rate badge (pulses red when cap < 85%)
 *  - Flood fraction bar on right edge
 *  - Stress glow on column border when G_gas is high
 */

import { useEffect, useRef, useCallback } from 'react'
import * as d3 from 'd3'

const STAGES   = 40
const BUBBLE_INTERVAL_MS = 600  // new bubble every N ms

function buildProfile(rl) {
  if (!rl) return Array(STAGES).fill(40)
  const { G = 0.8, y = 0.08, L = 5, T = 40, cap = 85 } = rl
  const load     = Math.min(1.0, (G * y) / (L * 0.012 + 0.001))
  const frontPos = 0.65 - load * 0.35
  const deltaT   = 12 + 8 * load + (100 - cap) * 0.15
  return Array.from({ length: STAGES }, (_, i) => {
    const z    = i / (STAGES - 1)
    const dist = Math.abs(z - frontPos)
    const bump = deltaT * Math.exp(-(dist * dist) / 0.03)
    return T + z * 5 + bump
  })
}

function frontZ(rl) {
  if (!rl) return 0.5
  const { G = 0.8, y = 0.08, L = 5 } = rl
  const load = Math.min(1.0, (G * y) / (L * 0.012 + 0.001))
  return 0.65 - load * 0.35
}

export default function MEAColumn({ state }) {
  const svgRef      = useRef(null)
  const bubblesRef  = useRef([])           // [{id, x, y, r, opacity}]
  const frameRef    = useRef(null)
  const spawnRef    = useRef(null)
  const stateRef    = useRef(state)

  // Keep a live ref to state so animation loop can read it without closure staling
  useEffect(() => { stateRef.current = state }, [state])

  // ── Bubble animation loop ─────────────────────────────────────────────────
  const animateBubbles = useCallback(() => {
    const svg  = d3.select(svgRef.current)
    const g    = svg.select('g.col-g')
    if (g.empty()) { frameRef.current = requestAnimationFrame(animateBubbles); return }

    const rl   = stateRef.current?.rl
    const W    = +svg.attr('width')  || 160
    const H    = +svg.attr('height') || 500
    const padX = 28, padT = 20, padB = 24
    const colW = W - padX * 2
    const colH = H - padT - padB

    // Move existing bubbles upward
    bubblesRef.current = bubblesRef.current
      .map(b => ({ ...b, y: b.y - b.speed }))
      .filter(b => b.y > padT - 10)

    // Render
    g.selectAll('.bubble')
      .data(bubblesRef.current, d => d.id)
      .join('circle')
      .attr('class', 'bubble')
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)
      .attr('r',  d => d.r)
      .attr('fill', 'rgba(100,180,255,0.18)')
      .attr('stroke', 'rgba(100,180,255,0.35)')
      .attr('stroke-width', 0.5)

    frameRef.current = requestAnimationFrame(animateBubbles)
  }, [])

  // Spawn bubbles on a timer
  useEffect(() => {
    let id = 0
    spawnRef.current = setInterval(() => {
      const rl    = stateRef.current?.rl
      const W     = +d3.select(svgRef.current).attr('width')  || 160
      const H     = +d3.select(svgRef.current).attr('height') || 500
      const padX  = 28, padT = 20, padB = 24
      const colW  = W - padX * 2
      const colH  = H - padT - padB
      const G     = rl?.G ?? 0.8
      const speed = 0.8 + G * 0.6    // faster bubbles at high G
      const n     = Math.max(1, Math.round(G * 2))  // more bubbles at high G

      for (let i = 0; i < n; i++) {
        bubblesRef.current.push({
          id:      ++id,
          x:       padX + 6 + Math.random() * (colW - 12),
          y:       padT + colH - 2,
          r:       1.2 + Math.random() * 2.5,
          speed:   speed * (0.7 + Math.random() * 0.6),
          opacity: 0.15 + Math.random() * 0.2,
        })
      }
    }, BUBBLE_INTERVAL_MS)

    frameRef.current = requestAnimationFrame(animateBubbles)

    return () => {
      clearInterval(spawnRef.current)
      cancelAnimationFrame(frameRef.current)
    }
  }, [animateBubbles])

  // ── Static D3 update (gradient + markers) ────────────────────────────────
  useEffect(() => {
    if (!svgRef.current) return

    const container = svgRef.current.parentElement
    const W = container.clientWidth  || 160
    const H = container.clientHeight || 500

    const svg = d3.select(svgRef.current)
      .attr('width',  W)
      .attr('height', H)

    const padX = 28, padT = 20, padB = 24
    const colW = W - padX * 2
    const colH = H - padT - padB

    // ── Defs ──────────────────────────────────────────────────────────────
    let defs = svg.select('defs')
    if (defs.empty()) defs = svg.append('defs')

    // Gradient
    let grad = defs.select('#col-grad')
    if (grad.empty()) {
      grad = defs.append('linearGradient')
        .attr('id', 'col-grad')
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '0%').attr('y2', '100%')
    }

    // Glow filter for stressed column
    if (defs.select('#col-glow').empty()) {
      const filt = defs.append('filter').attr('id', 'col-glow')
      filt.append('feGaussianBlur').attr('stdDeviation', 3).attr('result', 'blur')
      filt.append('feMerge').selectAll('feMergeNode')
        .data(['blur', 'SourceGraphic']).join('feMergeNode')
        .attr('in', d => d)
    }

    // ── Build group if needed ────────────────────────────────────────────
    let g = svg.select('g.col-g')
    if (g.empty()) g = svg.append('g').attr('class', 'col-g')

    // Ensure bubble layer is BELOW markers
    if (g.select('.bubble-layer').empty()) g.append('g').attr('class', 'bubble-layer')

    // ── Update gradient stops (with smooth transition) ───────────────────
    const profile  = buildProfile(state?.rl)
    const minT     = d3.min(profile)
    const maxT     = d3.max(profile)
    const colScale = d3.scaleSequential(d3.interpolateRdBu).domain([maxT, minT])

    const stops = grad.selectAll('stop').data(profile)
    stops.join('stop')
      .attr('offset', (_, i) => `${(i / (STAGES - 1)) * 100}%`)
      .transition().duration(400)
      .attr('stop-color', d => colScale(d))

    // ── Column body ──────────────────────────────────────────────────────
    const G       = state?.rl?.G ?? 0.8
    const stress  = Math.min(1, (G - 0.4) / 2.1)
    const glowClr = d3.interpolateOranges(stress * 0.8)

    g.selectAll('.col-body').data([1]).join('rect')
      .attr('class', 'col-body')
      .attr('x', padX).attr('y', padT)
      .attr('width', colW).attr('height', colH)
      .attr('rx', 4)
      .attr('fill', 'url(#col-grad)')
      .transition().duration(300)
      .attr('stroke', stress > 0.5 ? glowClr : '#1e2d4a')
      .attr('stroke-width', stress > 0.5 ? 2 : 1)
      .attr('filter', stress > 0.6 ? 'url(#col-glow)' : null)

    // ── Absorption front ─────────────────────────────────────────────────
    const fz   = frontZ(state?.rl)
    const fY   = padT + fz * colH

    g.selectAll('.front-line').data([1]).join('line')
      .attr('class', 'front-line')
      .attr('x1', padX - 4).attr('x2', padX + colW + 4)
      .transition().duration(400)
      .attr('y1', fY).attr('y2', fY)
      .attr('stroke', '#facc15')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0.8)

    // ── Labels ───────────────────────────────────────────────────────────
    svg.selectAll('.lbl-top').data([1]).join('text')
      .attr('class', 'lbl-top')
      .attr('x', W / 2).attr('y', padT - 6)
      .attr('text-anchor', 'middle')
      .style('fill', '#475569').style('font-size', 8).style('font-family', 'monospace')
      .text('← LEAN SOLVENT IN')

    svg.selectAll('.lbl-bot').data([1]).join('text')
      .attr('class', 'lbl-bot')
      .attr('x', W / 2).attr('y', H - 8)
      .attr('text-anchor', 'middle')
      .style('fill', '#475569').style('font-size', 8).style('font-family', 'monospace')
      .text('↑ FLUE GAS IN')

    // ── Capture badge ─────────────────────────────────────────────────────
    const cap      = state?.rl?.cap ?? 0
    const capColor = cap >= 90 ? '#22c55e' : cap >= 80 ? '#facc15' : '#ef4444'
    const pulse    = cap < 85

    const badgeBg = g.selectAll('.cap-badge-bg').data([1]).join('rect')
      .attr('class', 'cap-badge-bg')
      .attr('x', padX + colW / 2 - 30)
      .attr('y', padT + colH * 0.45)
      .attr('width', 60).attr('height', 22)
      .attr('rx', 4)
      .attr('fill', 'rgba(10,15,30,0.80)')

    badgeBg.transition().duration(300)
      .attr('stroke', capColor)
      .attr('stroke-width', pulse ? 2 : 1)

    g.selectAll('.cap-badge').data([1]).join('text')
      .attr('class', 'cap-badge')
      .attr('x', padX + colW / 2)
      .attr('y', padT + colH * 0.45 + 15)
      .attr('text-anchor', 'middle')
      .style('font-size', 13).style('font-weight', 700).style('font-family', 'monospace')
      .transition().duration(300)
      .style('fill', capColor)
      .text(`${cap.toFixed(1)}%`)

    // ── Flood fraction bar ────────────────────────────────────────────────
    const ff     = state?.rl?.ff ?? 0
    const ffFrac = Math.min(1, ff / 0.79)
    const ffH    = colH * ffFrac
    const ffClr  = ff > 0.70 ? '#ef4444' : ff > 0.60 ? '#f97316' : '#3b82f6'

    g.selectAll('.ff-track').data([1]).join('rect')
      .attr('class', 'ff-track')
      .attr('x', padX + colW + 4).attr('y', padT)
      .attr('width', 6).attr('height', colH)
      .attr('rx', 2).attr('fill', '#151f38')

    g.selectAll('.ff-fill').data([1]).join('rect')
      .attr('class', 'ff-fill')
      .attr('x', padX + colW + 4)
      .attr('width', 6).attr('rx', 2)
      .transition().duration(300)
      .attr('y', padT + colH - ffH)
      .attr('height', Math.max(0, ffH))
      .attr('fill', ffClr)

    // ── Intercooler stage marker ──────────────────────────────────────────
    const icY = padT + 0.5 * colH   // midpoint
    g.selectAll('.ic-line').data([1]).join('line')
      .attr('class', 'ic-line')
      .attr('x1', padX).attr('x2', padX + colW)
      .attr('y1', icY).attr('y2', icY)
      .attr('stroke', '#06b6d4')
      .attr('stroke-width', 0.8)
      .attr('stroke-dasharray', '2,4')
      .attr('opacity', 0.5)

    svg.selectAll('.ic-label').data([1]).join('text')
      .attr('class', 'ic-label')
      .attr('x', padX + 2).attr('y', icY - 3)
      .style('fill', '#06b6d4').style('font-size', 7).style('font-family', 'monospace')
      .text('IC')

  }, [state])

  return (
    <div className="mea-column-wrap">
      <svg ref={svgRef} style={{ display: 'block' }} />
    </div>
  )
}
