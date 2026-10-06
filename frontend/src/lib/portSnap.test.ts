/**
 * Runnable check for equipment port snapping: `bun run src/lib/portSnap.test.ts`
 * Throws (non-zero exit) on any failure.
 */
import { snapToEdge } from './portSnap'

let failures = 0
function check(label: string, actual: unknown, expected: unknown) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures += 1
    console.error(`FAIL ${label}\n  expected ${JSON.stringify(expected)}\n  got      ${JSON.stringify(actual)}`)
  }
}
// V-2501 from garnet/tests/input/14780-8120-25-25-0002A_equipment_bboxes.json
const B = { x_min: 1697, y_min: 884, x_max: 1951, y_max: 2090 }

check('inside, near top', snapToEdge(B, 1824.4, 890), { side: 'top', point_px: [1824, 884] })
check('inside, near right', snapToEdge(B, 1940, 1500), { side: 'right', point_px: [1951, 1500] })
check('outside below, x clamped', snapToEdge(B, 1990, 2100), { side: 'bottom', point_px: [1951, 2090] })
// Level with the top edge but far out to the left: the left edge is nearer than the top segment.
check('outside left, near top level', snapToEdge(B, 1597, 885), { side: 'left', point_px: [1697, 885] })
check('exactly on left edge', snapToEdge(B, 1697, 1200), { side: 'left', point_px: [1697, 1200] })

if (failures) throw new Error(`${failures} portSnap check(s) failed`)
console.log('portSnap: all checks passed')
