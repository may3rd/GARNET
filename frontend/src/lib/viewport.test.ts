/** Runnable check: `bun run src/lib/viewport.test.ts` */
import { clampPan, fitScale } from './viewport'

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

if (failures > 0) throw new Error(`${failures} viewport check(s) failed`)
console.log('viewport: all checks passed')
