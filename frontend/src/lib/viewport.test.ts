/** Runnable check: `bun run src/lib/viewport.test.ts` */
import { clampPan, fitScale, HANDLES, handlePoint, moveBox, resizeBox, wheelIntent, wheelPixels, zoomAbout, type Box } from './viewport'

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

/* --------------------------------------------------------------------------
   Wheel normalisation
   -------------------------------------------------------------------------- */

// Pixel mode passes through.
check('pixel mode is unchanged', wheelPixels(100, 0), 100)
check('pixel mode keeps sign', wheelPixels(-100, 0), -100)
// A wheel notch is ~3 lines; as raw pixels that would barely register.
check('line mode scales up', wheelPixels(3, 1), 48)
check('page mode uses the page height', wheelPixels(1, 2, false, 800), 240)
// A trackpad pinch sends small ctrlKey deltas.
check('pinch is amplified', wheelPixels(10, 0, true), 40)
check('pinch and scroll stay comparable', wheelPixels(10, 0, true) > wheelPixels(10, 0), true)
// One frame of input must never cross the whole zoom range.
check('clamped high', wheelPixels(99999, 0), 240)
check('clamped low', wheelPixels(-99999, 0), -240)
check('clamped after line scaling', wheelPixels(1000, 1), 240)
// A monotonic input must not change direction after normalising.
;[1, 3, 10, 100].forEach((d) => {
  check(`sign preserved for ${d}`, Math.sign(wheelPixels(d, 1)), 1)
  check(`sign preserved for -${d}`, Math.sign(wheelPixels(-d, 1)), -1)
})

/* --------------------------------------------------------------------------
   Zooming about a point
   -------------------------------------------------------------------------- */

const IMG = { w: 4000, h: 3000 }
const VP = { w: 900, h: 700 }
// Start zoomed in enough that both axes overflow, so clamping does not centre.
const startScale = 1
const startPan = { x: -1200, y: -900 }

// The image point under the cursor must not move.
;[
  { cx: 100, cy: 100 },
  { cx: 450, cy: 350 },
  { cx: 880, cy: 690 },
].forEach(({ cx, cy }) => {
  ;[1.25, 1 / 1.25, 2, 0.5].forEach((f) => {
    const r = zoomAbout(startPan, startScale, f, cx, cy, IMG.w, IMG.h, VP.w, VP.h)
    const before = { x: (cx - startPan.x) / startScale, y: (cy - startPan.y) / startScale }
    const after = { x: (cx - r.pan.x) / r.scale, y: (cy - r.pan.y) / r.scale }
    // Only meaningful while clamping has not taken over the axis.
    const clampedX = r.pan.x === 0 || Math.abs(r.pan.x + IMG.w * r.scale - VP.w) < 0.5
    const clampedY = r.pan.y === 0 || Math.abs(r.pan.y + IMG.h * r.scale - VP.h) < 0.5
    if (!clampedX) check(`x anchored at ${cx},${cy} x${f}`, Math.abs(before.x - after.x) < 0.5, true)
    if (!clampedY) check(`y anchored at ${cx},${cy} x${f}`, Math.abs(before.y - after.y) < 0.5, true)
  })
})

// The factor is applied exactly once — the bug was applying it twice.
const once = zoomAbout(startPan, 1, 1.5, 450, 350, IMG.w, IMG.h, VP.w, VP.h)
check('factor applied once', once.scale, 1.5)
check('not applied twice', once.scale === 1.5 * 1.5, false)

// Range is respected, and a no-op returns the input untouched.
check('clamped to max', zoomAbout(startPan, 8, 4, 450, 350, IMG.w, IMG.h, VP.w, VP.h).scale, 8)
check('clamped to min', zoomAbout(startPan, 0.02, 0.1, 450, 350, IMG.w, IMG.h, VP.w, VP.h).scale, 0.02)
const noop = zoomAbout(startPan, 8, 2, 450, 350, IMG.w, IMG.h, VP.w, VP.h)
check('no-op keeps the pan identical', noop.pan, startPan)

// Zooming in then out by the same factor returns to where it started.
const inThenOut = (() => {
  const a = zoomAbout(startPan, 1, 1.25, 450, 350, IMG.w, IMG.h, VP.w, VP.h)
  return zoomAbout(a.pan, a.scale, 1 / 1.25, 450, 350, IMG.w, IMG.h, VP.w, VP.h)
})()
check('round trip restores scale', Math.abs(inThenOut.scale - 1) < 1e-9, true)
check('round trip restores pan', 
  Math.abs(inThenOut.pan.x - startPan.x) < 0.5 && Math.abs(inThenOut.pan.y - startPan.y) < 0.5, true)

// Whatever the zoom, the result never opens a gutter.
;[0.05, 0.2, 1, 3, 8].forEach((sc) => {
  const r = zoomAbout(startPan, sc, 1.3, 450, 350, IMG.w, IMG.h, VP.w, VP.h)
  const settled = clampPan(r.pan, r.scale, IMG.w, IMG.h, VP.w, VP.h)
  check(`no gutter at scale ${sc}`, r.pan, settled)
})

/* --------------------------------------------------------------------------
   Wheel intent: pan by default, zoom with Ctrl/Cmd
   -------------------------------------------------------------------------- */

const wheel = (over: Partial<Parameters<typeof wheelIntent>[0]>) =>
  wheelIntent(
    { deltaX: 0, deltaY: 0, deltaMode: 0, ctrlKey: false, metaKey: false, ...over },
    900,
    700
  )

// A bare wheel pans, on whichever axis the input carries.
check('bare wheel pans', wheel({ deltaY: 120 }), { kind: 'pan', dx: 0, dy: 120 })
check('bare wheel pans horizontally', wheel({ deltaX: 90 }), { kind: 'pan', dx: 90, dy: 0 })
check('a trackpad two-axis scroll pans both', wheel({ deltaX: 30, deltaY: -40 }), {
  kind: 'pan', dx: 30, dy: -40,
})
// Sign is preserved so the caller can decide which way the sheet moves.
check('scroll down is positive', (wheel({ deltaY: 10 }) as { dy: number }).dy > 0, true)
check('scroll up is negative', (wheel({ deltaY: -10 }) as { dy: number }).dy < 0, true)

// Ctrl zooms — and so does a trackpad pinch, which arrives as ctrlKey.
check('ctrl zooms', wheel({ deltaY: -100, ctrlKey: true }).kind, 'zoom')
check('cmd zooms', wheel({ deltaY: -100, metaKey: true }).kind, 'zoom')
// Pinch deltas are tiny, so they keep the amplification; a deliberate
// Cmd+wheel is already a full-sized delta and must not be multiplied.
check('pinch is amplified', wheel({ deltaY: -10, ctrlKey: true }), { kind: 'zoom', delta: -40 })
check('cmd+wheel is not amplified', wheel({ deltaY: -10, metaKey: true }), { kind: 'zoom', delta: -10 })

// Other modifiers must not hijack the wheel into zooming.
check('shift still pans', wheel({ deltaY: 50, deltaMode: 0 }).kind, 'pan')

// Line/page delta modes are normalised in both branches.
check('line mode pans in pixels', wheel({ deltaY: 3, deltaMode: 1 }), { kind: 'pan', dx: 0, dy: 48 })
check('line mode zooms in pixels', wheel({ deltaY: 3, deltaMode: 1, metaKey: true }), {
  kind: 'zoom', delta: 48,
})
// A pan is clamped per event just like a zoom, so one event cannot fling the
// sheet across the whole raster.
check('pan is clamped', wheel({ deltaY: 99999 }), { kind: 'pan', dx: 0, dy: 240 })

if (failures > 0) throw new Error(`${failures} viewport check(s) failed`)
console.log('viewport (box editing + wheel + zoom + intent): all checks passed')
