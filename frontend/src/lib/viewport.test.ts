/** Runnable check: `bun run src/lib/viewport.test.ts` */
import { clampPan, fitScale, HANDLES, handlePoint, moveBox, resizeBox, type Box } from './viewport'

let failures = 0
function check(label: string, actual: unknown, expected: unknown) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures += 1
    console.error(`FAIL ${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
  }
}

const VIEW = { w: 800, h: 600 }
// A sheet zoomed to 2x the view on both axes.
const big = (p: { x: number; y: number }) => clampPan(p, 2, 800, 600, VIEW.w, VIEW.h)

// --- Larger than the view: edges can never come inside the frame ---------
check('cannot drag past the left edge', big({ x: 200, y: 0 }).x, 0)
check('cannot drag past the right edge', big({ x: -5000, y: 0 }).x, -800)
check('cannot drag past the top', big({ x: 0, y: 300 }).y, 0)
check('cannot drag past the bottom', big({ x: 0, y: -5000 }).y, -600)
check('a legal pan is untouched', big({ x: -400, y: -300 }), { x: -400, y: -300 })

// --- Smaller than the view: centred, so there is no lopsided gutter ------
const small = clampPan({ x: 0, y: 0 }, 0.5, 800, 600, VIEW.w, VIEW.h)
check('centred horizontally', small.x, (800 - 400) / 2)
check('centred vertically', small.y, (600 - 300) / 2)
// Dragging cannot shift a centred sheet off-centre.
check('drag cannot decentre it', clampPan({ x: -999, y: 999 }, 0.5, 800, 600, VIEW.w, VIEW.h), small)

// --- At fit scale one axis fits exactly and the other is centred ---------
const s = fitScale(3961, 3224, 800, 600)
check('fit scale picks the tighter axis', Math.abs(s - 600 / 3224) < 1e-9, true)
const atFit = clampPan({ x: 0, y: 0 }, s, 3961, 3224, 800, 600)
check('fitted axis is flush', Math.round(atFit.y), 0)
check('other axis is centred', Math.round(atFit.x), Math.round((800 - 3961 * s) / 2))

// --- No gap invariant: for any pan, the covered area is maximal ----------
const cases = [
  { x: 9999, y: 9999 },
  { x: -9999, y: -9999 },
  { x: 0, y: 0 },
  { x: 137, y: -42 },
]
cases.forEach((c) => {
  const p = clampPan(c, 2, 800, 600, VIEW.w, VIEW.h)
  // content is 1600x1200 over an 800x600 view, so both edges must be outside
  check(`no horizontal gap for ${JSON.stringify(c)}`, p.x <= 0 && p.x + 1600 >= 800, true)
  check(`no vertical gap for ${JSON.stringify(c)}`, p.y <= 0 && p.y + 1200 >= 600, true)
})

// Degenerate inputs must not produce NaN. A zero-size view counts as
// "content is larger", so the sheet pins to the origin rather than centring.
check('zero view is safe', clampPan({ x: 5, y: 5 }, 1, 100, 100, 0, 0), { x: 0, y: 0 })
check('fit scale with no image', fitScale(0, 0, 800, 600), 1)


/* --------------------------------------------------------------------------
   Box editing
   -------------------------------------------------------------------------- */

const B: Box = { Left: 100, Top: 100, Width: 200, Height: 100 } // right 300, bottom 200
const SHEET = { w: 1000, h: 800 }
const rs = (h: Parameters<typeof resizeBox>[1], x: number, y: number) =>
  resizeBox(B, h, x, y, SHEET.w, SHEET.h)

// --- Each anchor moves only the edges it names --------------------------
check('nw moves left+top', rs('nw', 60, 70), { Left: 60, Top: 70, Width: 240, Height: 130 })
check('se moves right+bottom', rs('se', 360, 260), { Left: 100, Top: 100, Width: 260, Height: 160 })
check('n moves top only', rs('n', 999, 40), { Left: 100, Top: 40, Width: 200, Height: 160 })
check('s moves bottom only', rs('s', 999, 260), { Left: 100, Top: 100, Width: 200, Height: 160 })
check('w moves left only', rs('w', 40, 999), { Left: 40, Top: 100, Width: 260, Height: 100 })
check('e moves right only', rs('e', 360, 999), { Left: 100, Top: 100, Width: 260, Height: 100 })

// --- Anchor positions are where the handles get drawn -------------------
check('nw point', handlePoint(B, 'nw'), { x: 100, y: 100 })
check('se point', handlePoint(B, 'se'), { x: 300, y: 200 })
check('n point is the top midpoint', handlePoint(B, 'n'), { x: 200, y: 100 })
check('e point is the right midpoint', handlePoint(B, 'e'), { x: 300, y: 150 })
check('there are eight anchors', HANDLES.length, 8)

// --- Dragging an edge past its opposite flips instead of going negative -
const flipped = rs('e', 20, 999)
check('flip keeps width positive', flipped.Width > 0, true)
check('flip lands left of the old left edge', flipped.Left, 20)
check('flip right edge is the old left edge', flipped.Left + flipped.Width, 100)

// --- Never leaves the sheet, never collapses to nothing -----------------
const outside = rs('nw', -500, -500)
check('clamped to the sheet origin', [outside.Left, outside.Top], [0, 0])
const past = rs('se', 99999, 99999)
check('clamped to the sheet extent', [past.Left + past.Width, past.Top + past.Height], [SHEET.w, SHEET.h])
// Collapsing an edge onto its opposite must still leave a grabbable box.
const collapsed = rs('e', 100, 999)
check('minimum width is honoured', collapsed.Width >= 4, true)
const collapsedV = rs('s', 999, 100)
check('minimum height is honoured', collapsedV.Height >= 4, true)

HANDLES.forEach((h) => {
  const r = resizeBox(B, h, -9999, -9999, SHEET.w, SHEET.h)
  const inside =
    r.Left >= 0 && r.Top >= 0 && r.Left + r.Width <= SHEET.w && r.Top + r.Height <= SHEET.h
  check(`${h} stays inside the sheet`, inside, true)
  check(`${h} keeps a usable size`, r.Width >= 4 && r.Height >= 4, true)
})

// --- Moving ------------------------------------------------------------
check('move by a delta', moveBox(B, 25, -30, SHEET.w, SHEET.h), {
  Left: 125, Top: 70, Width: 200, Height: 100,
})
check('move cannot cross the top-left', moveBox(B, -9999, -9999, SHEET.w, SHEET.h), {
  Left: 0, Top: 0, Width: 200, Height: 100,
})
check('move cannot cross the bottom-right', moveBox(B, 9999, 9999, SHEET.w, SHEET.h), {
  Left: SHEET.w - 200, Top: SHEET.h - 100, Width: 200, Height: 100,
})
// Moving preserves size — a drag must never resize.
const moved = moveBox(B, 40, 40, SHEET.w, SHEET.h)
check('move preserves size', [moved.Width, moved.Height], [B.Width, B.Height])

// --- Integer output: these are pixel coordinates on a raster ------------
const fract = resizeBox({ Left: 10.4, Top: 10.6, Width: 50.5, Height: 50.5 }, 'se', 80.7, 90.2, SHEET.w, SHEET.h)
check('coordinates are integers', Object.values(fract).every(Number.isInteger), true)

if (failures > 0) throw new Error(`${failures} viewport check(s) failed`)
console.log('viewport (with box editing): all checks passed')
