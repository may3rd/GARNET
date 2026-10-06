/**
 * Runnable check for the trace editor geometry: `bun run src/lib/traceGeometry.test.ts`
 * Throws (non-zero exit) on any failure.
 */
import { densify, eraseSegments, mergeDrawn, orthogonal, strokeTangents, type Segment } from './traceGeometry'

let failures = 0
function check(label: string, actual: unknown, expected: unknown) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures += 1
    console.error(`FAIL ${label}\n  expected ${JSON.stringify(expected)}\n  got      ${JSON.stringify(actual)}`)
  }
}
const seg = (x1: number, y1: number, x2: number, y2: number): Segment => ({ x1, y1, x2, y2 })
// An existing horizontal trace A: (0,0) -> (100,0).
const A = { id: 'A', segments: [seg(0, 0, 100, 0)] }

// --- extend: stroke starts 3 px from A's end, runs away from it -------------------------
{
  const m = mergeDrawn([seg(103, 2, 200, 2)], [A], 12)
  check('extend at end: kind', m.kind, 'extend')
  if (m.kind === 'extend') {
    check('extend at end: same trace', m.traceId, 'A')
    // A, then an L connector closing the 3 px / 2 px gap, then the stroke.
    check('extend at end: segments', m.segments, [seg(0, 0, 100, 0), seg(100, 0, 103, 0), seg(103, 0, 103, 2), seg(103, 2, 200, 2)])
  }
}

// --- extend: stroke drawn TOWARD A's start (its end sits on A's start) ---------------------
{
  const m = mergeDrawn([seg(-50, 0, 0, 0)], [A], 12)
  check('extend at start: kind', m.kind, 'extend')
  if (m.kind === 'extend') check('extend at start: stroke prepended', m.segments, [seg(-50, 0, 0, 0), seg(0, 0, 100, 0)])
}

// --- tee: stroke ends 4 px off the MIDDLE of A -> new trace, end pulled onto A ---------------
{
  const m = mergeDrawn([seg(50, 80, 50, 4)], [A], 12)
  check('tee: kind', m.kind, 'new')
  if (m.kind === 'new') {
    check('tee: joined', m.joinedTo, ['A'])
    check('tee: end on A', m.segments[m.segments.length - 1], seg(50, 4, 50, 0))
  }
}

// --- far from everything -> untouched new trace --------------------------------------------
{
  const m = mergeDrawn([seg(500, 500, 600, 500)], [A], 12)
  check('isolated', m, { kind: 'new', segments: [seg(500, 500, 600, 500)], joinedTo: [] })
}

// --- never fuse two traces: end-to-end with A, other end on B's end -> extend A, tee to B ---
{
  const B = { id: 'B', segments: [seg(300, 0, 400, 0)] }
  const m = mergeDrawn([seg(100, 0, 300, 0)], [A, B], 12)
  check('bridge: extends A only', m.kind === 'extend' && m.traceId, 'A')
  check('bridge: B named, not absorbed', m.kind === 'extend' && m.joinedTo, 'B')
}

// --- erase: brush across the middle splits a run into two pieces ----------------------------
check('erase splits', eraseSegments([seg(0, 0, 100, 0)], [[50, 0]], 10), [seg(0, 0, 40, 0), seg(60, 0, 100, 0)])
check('erase misses', eraseSegments([seg(0, 0, 100, 0)], [[50, 30]], 10), null)

// --- directional erase: wipe ALONG a horizontal pipe that a vertical pipe crosses ------------
{
  const r = 10
  const H = seg(0, 100, 400, 100) // the pipe being wiped
  const V = seg(200, 0, 200, 300) // crosses it at x=200
  // A slightly wobbly horizontal stroke from x=120 to x=280, right over H and through V.
  const raw: [number, number][] = [[120, 100], [160, 102], [200, 99], [240, 101], [280, 100]]
  const brush = densify(raw, r / 2)
  const tan = strokeTangents(brush, r)
  check('stroke has direction', tan !== null, true)
  const cutH = eraseSegments([H], brush, r, tan)
  check('along: H is cut', cutH !== null && cutH.length === 2, true)
  check('across: V untouched', eraseSegments([V], brush, r, tan), null)
  // Without direction (old behaviour) V would have been nicked: proves the filter does the work.
  check('undirected would cut V', eraseSegments([V], brush, r) !== null, true)
}

// --- a dab has no direction and erases whatever is under it ---------------------------------
{
  const r = 10
  const dab = densify([[200, 100], [204, 101]], r / 2)
  check('dab: no tangents', strokeTangents(dab, r), null)
  check('dab: cuts the crossing pipe too', eraseSegments([seg(200, 0, 200, 300)], dab, r, null) !== null, true)
}

// --- an L-shaped stroke follows an L-shaped trace round the corner --------------------------
{
  const r = 10
  const L = [seg(0, 0, 200, 0), seg(200, 0, 200, 200)]
  const brush = densify([[50, 0], [200, 0], [200, 150]], r / 2)
  const out = eraseSegments(L, brush, r, strokeTangents(brush, r))
  // Only the unbrushed ends survive; no stub left at the corner.
  check('L stroke: follows the corner', out, [seg(0, 0, 40, 0), seg(200, 160, 200, 200)])
}

// --- orthogonal: a wobbly L becomes two axis runs --------------------------------------------
check('ortho L', orthogonal([[0, 0], [100, 3], [102, 80]]), [[0, 0], [100, 0], [100, 80]])

if (failures > 0) throw new Error(`${failures} traceGeometry check(s) failed`)
console.log('traceGeometry: all checks passed')
