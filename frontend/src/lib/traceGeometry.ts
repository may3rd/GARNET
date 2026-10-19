/**
 * Pure trace geometry for the Gate 2 editor: simplify, straighten, erase, and merge a drawn
 * path into the existing network. No React, so `bun run src/lib/traceGeometry.test.ts` checks it.
 */

export type Segment = { x1: number; y1: number; x2: number; y2: number; length_px?: number }

export type Pt = [number, number]

/** Ramer-Douglas-Peucker: drop points within `eps` of the line through their neighbours. */
export function rdp(pts: Pt[], eps: number): Pt[] {
  if (pts.length < 3) return pts
  const [a, b] = [pts[0], pts[pts.length - 1]]
  let idx = 0
  let max = 0
  for (let i = 1; i < pts.length - 1; i++) {
    const d = segDist(pts[i][0], pts[i][1], { x1: a[0], y1: a[1], x2: b[0], y2: b[1] })
    if (d > max) [max, idx] = [d, i]
  }
  return max > eps ? [...rdp(pts.slice(0, idx + 1), eps).slice(0, -1), ...rdp(pts.slice(idx), eps)] : [a, b]
}

/** Freehand -> horizontal/vertical runs. Each leg follows its dominant axis; collinear legs merge. */
export function orthogonal(pts: Pt[]): Pt[] {
  const out: Pt[] = [[Math.round(pts[0][0]), Math.round(pts[0][1])]]
  for (const p of pts.slice(1)) {
    const cur = out[out.length - 1]
    const q: Pt = Math.abs(p[0] - cur[0]) >= Math.abs(p[1] - cur[1]) ? [Math.round(p[0]), cur[1]] : [cur[0], Math.round(p[1])]
    if (q[0] === cur[0] && q[1] === cur[1]) continue
    const prev = out[out.length - 2]
    const sameAxis = prev && ((prev[1] === cur[1] && cur[1] === q[1]) || (prev[0] === cur[0] && cur[0] === q[0]))
    if (sameAxis) out[out.length - 1] = q
    else out.push(q)
  }
  return out
}

/** Interpolate a stroke so consecutive points are at most `step` apart (brush coverage has no gaps). */
export function densify(pts: Pt[], step: number): Pt[] {
  const out: Pt[] = [pts[0]]
  for (let i = 1; i < pts.length; i++) {
    const [ax, ay] = pts[i - 1]
    const [bx, by] = pts[i]
    const n = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / step))
    for (let k = 1; k <= n; k++) out.push([ax + ((bx - ax) * k) / n, ay + ((by - ay) * k) / n])
  }
  return out
}

/**
 * Cut away the parts of `segs` within `r` of any brush point. Works on each segment's own axis:
 * every brush point blanks the chord where its circle crosses the segment; what is left (>= 3 px)
 * survives as new segments. Returns null when nothing was touched.
 */
/** Max angle between a stroke and a segment for the stroke to erase it (directional mode). */
export const ERASE_ALIGN_DEG = 30

/**
 * Unit direction of the stroke at each point, smoothed over ~one brush width either side so hand
 * wobble does not count as a turn. Null for a DAB (stroke shorter than two brush diameters): a
 * tap has no direction, so it erases everything under it.
 */
export function strokeTangents(brush: Pt[], r: number): (Pt | null)[] | null {
  let length = 0
  for (let i = 1; i < brush.length; i++) length += Math.hypot(brush[i][0] - brush[i - 1][0], brush[i][1] - brush[i - 1][1])
  if (length < 4 * r) return null
  return brush.map((_, i) => {
    // Walk out to ~r of arc length on each side.
    const reach = (dir: 1 | -1) => {
      let j = i
      let acc = 0
      while (j + dir >= 0 && j + dir < brush.length && acc < r) {
        acc += Math.hypot(brush[j + dir][0] - brush[j][0], brush[j + dir][1] - brush[j][1])
        j += dir
      }
      return brush[j]
    }
    const a = reach(-1)
    const b = reach(1)
    const n = Math.hypot(b[0] - a[0], b[1] - a[1])
    return n > 1e-6 ? [(b[0] - a[0]) / n, (b[1] - a[1]) / n] : null
  })
}

/**
 * Cut away the parts of `segs` within `r` of the brush. Each brush point blanks the chord where
 * its circle crosses a segment; what is left (>= 3 px) survives as new segments. Returns null
 * when nothing was touched.
 *
 * Directional: with `tangents` (see strokeTangents), a brush point only cuts a segment running
 * the same way as the stroke there (within ERASE_ALIGN_DEG), so wiping along one pipe leaves the
 * pipes that cross it intact. A point with no direction (null tangent) cuts anything, as a dab.
 */
export function eraseSegments(segs: Segment[], brush: Pt[], r: number, tangents: (Pt | null)[] | null = null): Segment[] | null {
  const minCos = Math.cos((ERASE_ALIGN_DEG * Math.PI) / 180)
  let touched = false
  const out: Segment[] = []
  for (const g of segs) {
    const L = Math.hypot(g.x2 - g.x1, g.y2 - g.y1)
    if (L === 0) continue
    const ux = (g.x2 - g.x1) / L
    const uy = (g.y2 - g.y1) / L
    const cut: [number, number][] = []
    for (let i = 0; i < brush.length; i++) {
      const [px, py] = brush[i]
      const tan = tangents?.[i]
      // Either direction along the segment counts: a pipe has no preferred sense here.
      if (tan && Math.abs(tan[0] * ux + tan[1] * uy) < minCos) continue
      const t = (px - g.x1) * ux + (py - g.y1) * uy
      const d = Math.abs((px - g.x1) * uy - (py - g.y1) * ux)
      if (d > r) continue
      const half = Math.sqrt(r * r - d * d)
      const lo = Math.max(0, t - half)
      const hi = Math.min(L, t + half)
      if (hi > lo) cut.push([lo, hi])
    }
    if (cut.length === 0) {
      out.push(g)
      continue
    }
    touched = true
    cut.sort((a, b) => a[0] - b[0])
    let at = 0
    const keep: [number, number][] = []
    for (const [lo, hi] of cut) {
      if (lo > at) keep.push([at, lo])
      at = Math.max(at, hi)
    }
    if (at < L) keep.push([at, L])
    for (const [a, b] of keep) {
      if (b - a < 3) continue
      out.push({
        x1: Math.round(g.x1 + ux * a),
        y1: Math.round(g.y1 + uy * a),
        x2: Math.round(g.x1 + ux * b),
        y2: Math.round(g.y1 + uy * b),
      })
    }
  }
  return touched ? out : null
}

const xy = (g: Segment, end: boolean): Pt => (end ? [g.x2, g.y2] : [g.x1, g.y1])
const startOf = (segs: Segment[]): Pt => xy(segs[0], false)
const endOf = (segs: Segment[]): Pt => xy(segs[segs.length - 1], true)
const reverseSegs = (segs: Segment[]): Segment[] =>
  [...segs].reverse().map((g) => ({ x1: g.x2, y1: g.y2, x2: g.x1, y2: g.y1 }))

/** Axis-aligned path a -> b: nothing, one run, or an L (horizontal then vertical). */
export function connector(a: Pt, b: Pt): Segment[] {
  if (a[0] === b[0] && a[1] === b[1]) return []
  if (a[0] === b[0] || a[1] === b[1]) return [{ x1: a[0], y1: a[1], x2: b[0], y2: b[1] }]
  return [
    { x1: a[0], y1: a[1], x2: b[0], y2: a[1] },
    { x1: b[0], y1: a[1], x2: b[0], y2: b[1] },
  ]
}

/** Nearest point on segment g to p, rounded to the pixel grid. */
export function foot(p: Pt, g: Segment): Pt {
  const dx = g.x2 - g.x1
  const dy = g.y2 - g.y1
  const len2 = dx * dx + dy * dy
  const t = len2 ? Math.max(0, Math.min(1, ((p[0] - g.x1) * dx + (p[1] - g.y1) * dy) / len2)) : 0
  return [Math.round(g.x1 + t * dx), Math.round(g.y1 + t * dy)]
}

export type MergeResult =
  | { kind: 'extend'; traceId: string; segments: Segment[]; joinedTo?: string }
  | { kind: 'new'; segments: Segment[]; joinedTo: string[] }

/**
 * Join a freshly drawn path to the existing network.
 *
 * 1. If an end of the stroke is within `tol` of a FREE END of a trace, the stroke becomes an
 *    extension of that trace (same id), oriented to continue from that end; the joint is closed
 *    exactly with an axis-aligned connector.
 * 2. Otherwise an end within `tol` of the BODY of a trace is pulled onto it (a tee): connected,
 *    but still its own trace.
 * The far end of an extension may still tee into another trace. Two existing traces are never
 * fused: they can carry different line numbers, and that call belongs to the reviewer.
 */
export function mergeDrawn(drawn: Segment[], traces: { id: string; segments: Segment[] }[], tol: number): MergeResult {
  const d = (a: Pt, b: Pt) => Math.hypot(a[0] - b[0], a[1] - b[1])
  type Hit = { id: string; segs: Segment[]; which: 'start' | 'end'; dist: number }
  const endHit = (p: Pt): Hit | null => {
    let best: Hit | null = null
    for (const t of traces) {
      if (!t.segments.length) continue
      for (const which of ['start', 'end'] as const) {
        const q = which === 'start' ? startOf(t.segments) : endOf(t.segments)
        const dist = d(p, q)
        if (dist <= tol && (!best || dist < best.dist)) best = { id: t.id, segs: t.segments, which, dist }
      }
    }
    return best
  }
  const bodyHit = (p: Pt, skip?: string): { id: string; at: Pt } | null => {
    let best: { id: string; at: Pt; dist: number } | null = null
    for (const t of traces) {
      if (t.id === skip) continue
      for (const g of t.segments) {
        const f = foot(p, g)
        const dist = d(p, f)
        if (dist <= tol && (!best || dist < best.dist)) best = { id: t.id, at: f, dist }
      }
    }
    return best && { id: best.id, at: best.at }
  }
  /** Pull the free end of `path` onto a nearby trace body, if any. */
  const teeEnd = (path: Segment[], skip?: string): { segs: Segment[]; to?: string } => {
    const tee = bodyHit(endOf(path), skip)
    return tee ? { segs: [...path, ...connector(endOf(path), tee.at)], to: tee.id } : { segs: path }
  }

  const atStart = endHit(startOf(drawn))
  const atEnd = endHit(endOf(drawn))
  // Prefer the closer join when both ends sit on trace ends.
  const hit = atStart && atEnd ? (atStart.dist <= atEnd.dist ? { h: atStart, end: 'start' as const } : { h: atEnd, end: 'end' as const })
    : atStart ? { h: atStart, end: 'start' as const } : atEnd ? { h: atEnd, end: 'end' as const } : null

  if (hit) {
    // Orient the stroke so it runs AWAY from the joint, then attach at the trace's matching end.
    const away = hit.end === 'start' ? drawn : reverseSegs(drawn)
    const { segs: tail, to } = teeEnd(away, hit.h.id)
    if (hit.h.which === 'end') {
      const joint = endOf(hit.h.segs)
      return { kind: 'extend', traceId: hit.h.id, segments: [...hit.h.segs, ...connector(joint, startOf(tail)), ...tail], joinedTo: to }
    }
    const joint = startOf(hit.h.segs)
    const ext = reverseSegs(tail)
    return { kind: 'extend', traceId: hit.h.id, segments: [...ext, ...connector(endOf(ext), joint), ...hit.h.segs], joinedTo: to }
  }

  // No free end nearby: tee either end onto a trace body.
  const joined: string[] = []
  let segs = drawn
  const e = teeEnd(segs)
  if (e.to) [segs, joined[0]] = [e.segs, e.to]
  const s2 = teeEnd(reverseSegs(segs))
  if (s2.to) {
    segs = reverseSegs(s2.segs)
    joined.push(s2.to)
  }
  return { kind: 'new', segments: segs, joinedTo: joined }
}


/** Remove the outer segment; a trace must retain at least two points. */
export function deleteTraceEndpoint(segs: Segment[], point: Pt): Segment[] | null {
  if (segs.length < 2) return null
  const first = segs[0], last = segs[segs.length - 1]
  if (point[0] === first.x1 && point[1] === first.y1) return segs.slice(1)
  if (point[0] === last.x2 && point[1] === last.y2) return segs.slice(0, -1)
  return null
}

/** Add one outer segment when the reviewer places a new endpoint. */
export function extendTraceEndpoint(segs: Segment[], point: Pt, to: Pt, snapOrthogonal: boolean): Segment[] | null {
  if (!segs.length) return null
  const first = segs[0], last = segs[segs.length - 1]
  const atStart = point[0] === first.x1 && point[1] === first.y1
  const atEnd = point[0] === last.x2 && point[1] === last.y2
  if (atStart === atEnd) return null
  const next = snapOrthogonal ? orthogonal([point, to])[1] : to
  if (!next || (next[0] === point[0] && next[1] === point[1])) return null
  const added = atStart
    ? { x1: next[0], y1: next[1], x2: point[0], y2: point[1] }
    : { x1: point[0], y1: point[1], x2: next[0], y2: next[1] }
  return atStart ? [added, ...segs] : [...segs, added]
}

/** Move a shared vertex; orthogonal snap inserts bends so all other vertices stay fixed. */
export function moveTracePoint(segs: Segment[], from: Pt, to: Pt, snapOrthogonal = false): Segment[] {
  const loneEndpoint = segs.filter((g) =>
    (g.x1 === from[0] && g.y1 === from[1]) || (g.x2 === from[0] && g.y2 === from[1])).length === 1
  return segs.flatMap((g) => {
    const start = g.x1 === from[0] && g.y1 === from[1]
    const end = g.x2 === from[0] && g.y2 === from[1]
    if (!start && !end) return [{ x1: g.x1, y1: g.y1, x2: g.x2, y2: g.y2 }]
    if (!snapOrthogonal || (start && end)) return [{
      x1: start ? to[0] : g.x1, y1: start ? to[1] : g.y1,
      x2: end ? to[0] : g.x2, y2: end ? to[1] : g.y2,
    }]
    const fixed: Pt = start ? [g.x2, g.y2] : [g.x1, g.y1]
    const horizontal = Math.abs(g.x2 - g.x1) >= Math.abs(g.y2 - g.y1)
    if (loneEndpoint) {
      const snapped: Pt = horizontal ? [to[0], fixed[1]] : [fixed[0], to[1]]
      return [{ x1: start ? snapped[0] : fixed[0], y1: start ? snapped[1] : fixed[1],
        x2: end ? snapped[0] : fixed[0], y2: end ? snapped[1] : fixed[1] }]
    }
    const bend: Pt = horizontal ? [to[0], fixed[1]] : [fixed[0], to[1]]
    const points = start ? [to, bend, fixed] : [fixed, bend, to]
    return points.slice(1).map((p, i) => ({ x1: points[i][0], y1: points[i][1], x2: p[0], y2: p[1] }))
      .filter((s) => s.x1 !== s.x2 || s.y1 !== s.y2)
  })
}

/** Distance from point (x, y) to segment g, in image pixels. */
export function segDist(x: number, y: number, g: Segment) {
  const dx = g.x2 - g.x1
  const dy = g.y2 - g.y1
  const len2 = dx * dx + dy * dy
  const t = len2 ? Math.max(0, Math.min(1, ((x - g.x1) * dx + (y - g.y1) * dy) / len2)) : 0
  return Math.hypot(x - (g.x1 + t * dx), y - (g.y1 + t * dy))
}
