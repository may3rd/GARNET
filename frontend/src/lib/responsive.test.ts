/** Runnable check: `bun run src/lib/responsive.test.ts` */
import {
  availabilityOf,
  BREAKPOINTS,
  controlHeight,
  GATE_AVAILABILITY,
  widthFor,
} from './responsive'

let failures = 0
function check(label: string, actual: unknown, expected: unknown) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures += 1
    console.error(`FAIL ${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
  }
}

// --- Buckets, including the exact boundaries ------------------------------
check('below tablet is phone', widthFor(767), 'phone')
check('at tablet is tablet', widthFor(BREAKPOINTS.tablet), 'tablet')
check('just below desktop is tablet', widthFor(1279), 'tablet')
check('at desktop is desktop', widthFor(BREAKPOINTS.desktop), 'desktop')
check('phone frame from the canvas', widthFor(390), 'phone')
// An unmeasured viewport (hidden tab, collapsed pane) must not be mistaken
// for a phone, or it withholds screens a phone is not offered.
check('zero width is not a phone', widthFor(0), 'desktop')
check('negative width is not a phone', widthFor(-1), 'desktop')
check('NaN width is not a phone', widthFor(Number.NaN), 'desktop')
check('tablet frame from the canvas', widthFor(1024), 'tablet')
check('desktop frame from the canvas', widthFor(1440), 'desktop')

// --- The artboard's per-screen table -------------------------------------
check('detection not offered on phone', availabilityOf('detection', 'phone'), 'not-offered')
check('detection read-only on tablet', availabilityOf('detection', 'tablet'), 'read-only')
check('detection full on desktop', availabilityOf('detection', 'desktop'), 'full')
check('sheets full everywhere', [
  availabilityOf('sheets', 'phone'),
  availabilityOf('sheets', 'tablet'),
  availabilityOf('sheets', 'desktop'),
], ['full', 'full', 'full'])
check('task full everywhere', availabilityOf('task', 'phone'), 'full')
check('run full on phone', availabilityOf('run', 'phone'), 'full')
check('exports read-only on phone', availabilityOf('exports', 'phone'), 'read-only')

// --- Per-gate: drawing gates are withheld, deciding gates are not --------
check('gate 1 not offered on phone', GATE_AVAILABILITY[1].phone, 'not-offered')
check('gate 2 not offered on phone', GATE_AVAILABILITY[2].phone, 'not-offered')
check('gate 3 read-only on phone', GATE_AVAILABILITY[3].phone, 'read-only')
// Gate 4 is a choice, not a drawing, so a phone can do it in full.
check('gate 4 full on phone', GATE_AVAILABILITY[4].phone, 'full')
;([1, 2, 3, 4] as const).forEach((g) => {
  check(`gate ${g} full on tablet`, GATE_AVAILABILITY[g].tablet, 'full')
  check(`gate ${g} full on desktop`, GATE_AVAILABILITY[g].desktop, 'full')
})

// --- Availability never improves as the screen gets smaller --------------
const RANK = { 'not-offered': 0, 'read-only': 1, full: 2 } as const
;(['sheets', 'task', 'detection', 'run', 'review', 'merge', 'exports'] as const).forEach((s) => {
  const p = RANK[availabilityOf(s, 'phone')]
  const t = RANK[availabilityOf(s, 'tablet')]
  const d = RANK[availabilityOf(s, 'desktop')]
  check(`${s} degrades monotonically`, p <= t && t <= d, true)
})

// --- Touch targets grow as the pointer gets coarser ----------------------
check('desktop controls', controlHeight('desktop'), 36)
check('tablet controls', controlHeight('tablet'), 40)
check('phone controls meet the 48px target', controlHeight('phone'), 48)

if (failures > 0) throw new Error(`${failures} responsive check(s) failed`)
console.log('responsive: all checks passed')
