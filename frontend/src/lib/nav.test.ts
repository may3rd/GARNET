/**
 * Runnable check for navigation: `bun run src/lib/nav.test.ts`
 *
 * Breadcrumb shapes and rail highlights are transcribed from the artboards,
 * so these assertions are what keeps the chrome honest if a screen moves.
 */
import { breadcrumbFor, RAIL_FOR, RAIL_TARGET, topbarChip, type Screen } from './nav'

let failures = 0
function check(label: string, actual: unknown, expected: unknown) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures += 1
    console.error(`FAIL ${label}\n  expected ${JSON.stringify(expected)}\n  got      ${JSON.stringify(actual)}`)
  }
}

const ctx = { project: 'Unit 210' }
const labels = (s: Screen, c = ctx) => breadcrumbFor(s, c).map((x) => x.label)

// --- Breadcrumb shape per artboard -----------------------------------------
check('sheets crumbs', labels('sheets'), ['Projects', 'Unit 210', 'New run'])
check('task crumbs', labels('task'), ['Unit 210', 'New run', 'Task'])
check('run crumbs', labels('run'), ['Unit 210', 'New run', 'Extraction'])
check('exports crumbs', labels('exports'), ['Unit 210', 'New run', 'Exports'])
check('merge crumbs', labels('merge'), ['Unit 210', 'New run', 'Merge'])

// Sheet-scoped screens put the sheet in the middle, as the gate artboards do.
const sheetCtx = { project: 'Unit 210', sheetLabel: 'PDF-210-DEB p.2', gate: 4 as const }
check('gate crumbs', labels('review', sheetCtx), ['Unit 210', 'PDF-210-DEB p.2', 'Gate 4'])
check('detection crumbs', labels('detection', sheetCtx), [
  'Unit 210',
  'PDF-210-DEB p.2',
  'Detection results',
])

// --- The last segment is never a link; earlier ones are --------------------
const taskCrumbs = breadcrumbFor('task', ctx)
check('last crumb is not a link', taskCrumbs[taskCrumbs.length - 1].to, undefined)
check('project crumb links to sheets', taskCrumbs[0].to, 'sheets')
// On a gate, the sheet segment goes back to the run monitor.
check('gate sheet crumb links to run', breadcrumbFor('review', sheetCtx)[1].to, 'run')

// Every non-final segment that is shown must be reachable, whether or not a
// sheet is in focus — otherwise the crumb renders as dead text.
;(['task', 'detection', 'run', 'review', 'merge', 'exports'] as Screen[]).forEach((s) => {
  const crumbs = breadcrumbFor(s, ctx)
  crumbs.slice(0, -1).forEach((c, i) => {
    check(`${s} crumb ${i} (${c.label}) is reachable`, typeof c.to === 'string', true)
  })
})

// --- Rail highlight per artboard ------------------------------------------
check('sheets rail', RAIL_FOR.sheets, 'sheets')
check('task rail is Detection', RAIL_FOR.task, 'detection')
check('detection rail', RAIL_FOR.detection, 'detection')
check('run rail is Extraction', RAIL_FOR.run, 'extraction')
check('review rail', RAIL_FOR.review, 'review')
check('exports rail', RAIL_FOR.exports, 'exports')

// Every rail target must round-trip back to that rail icon, or the icon would
// light up a different entry than the one just clicked.
;(Object.keys(RAIL_TARGET) as (keyof typeof RAIL_TARGET)[]).forEach((key) => {
  check(`rail ${key} round-trips`, RAIL_FOR[RAIL_TARGET[key]], key)
})

// --- Topbar chip ----------------------------------------------------------
check('chip on run', topbarChip('run', ctx), 'Extraction running')
check('chip on gate', topbarChip('review', sheetCtx), 'Gate 4 of 4')
check('chip default is project', topbarChip('sheets', ctx), 'Unit 210')

if (failures > 0) throw new Error(`${failures} nav check(s) failed`)
console.log('nav: all checks passed')
