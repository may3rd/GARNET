/**
 * Runnable check for navigation: `bun run src/lib/nav.test.ts`
 *
 * The breadcrumb is the pipeline path, so these assertions are mostly about
 * that path staying ordered, reachable and free of invented hierarchy.
 */
import { breadcrumbFor, RAIL_FOR, RAIL_TARGET, topbarChip, type Screen } from './nav'

let failures = 0
function check(label: string, actual: unknown, expected: unknown) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures += 1
    console.error(`FAIL ${label}\n  expected ${JSON.stringify(expected)}\n  got      ${JSON.stringify(actual)}`)
  }
}

const ALL: Screen[] = ['sheets', 'task', 'detection', 'run', 'review', 'merge', 'exports']
const labels = (s: Screen, c = {}) => breadcrumbFor(s, c).map((x) => x.label)

// --- The pipeline path -----------------------------------------------------
check('sheets is the root', labels('sheets'), ['Sheets'])
check('task follows sheets', labels('task'), ['Sheets', 'Task'])
check('run follows task', labels('run'), ['Sheets', 'Task', 'Run'])
check('detection hangs off task', labels('detection'), ['Sheets', 'Task', 'Detection results'])
check('merge hangs off run', labels('merge'), ['Sheets', 'Task', 'Run', 'Merge'])
check('exports hangs off run', labels('exports'), ['Sheets', 'Task', 'Run', 'Exports'])

// Only the gate screens name a sheet, since only they act on one at a time.
const gateCtx = { sheetLabel: 'PDF-210-DEB p.2', gate: 4 as const }
check('gate names its sheet', labels('review', gateCtx), [
  'Sheets',
  'Task',
  'Run',
  'PDF-210-DEB p.2',
  'Gate 4',
])
check('gate without a sheet still works', labels('review'), ['Sheets', 'Task', 'Run', 'Review'])

// --- No invented hierarchy -------------------------------------------------
// There is no project entity in the backend, so no crumb may claim one.
const FICTION = ['Projects', 'Project', 'Unit 210', 'New run', 'Workspace']
ALL.forEach((s) => {
  labels(s, gateCtx).forEach((l) => {
    check(`${s} crumb "${l}" is not invented hierarchy`, FICTION.includes(l), false)
  })
})

// --- Every non-final crumb is reachable, or it renders as dead text --------
ALL.forEach((s) => {
  const crumbs = breadcrumbFor(s, gateCtx)
  crumbs.slice(0, -1).forEach((c, i) => {
    check(`${s} crumb ${i} (${c.label}) is reachable`, typeof c.to === 'string', true)
  })
  check(`${s} last crumb is not a link`, crumbs[crumbs.length - 1].to, undefined)
})

// --- The path is ordered: each crumb's screen precedes the next ------------
const ORDER: Screen[] = ['sheets', 'task', 'run']
ALL.forEach((s) => {
  const steps = breadcrumbFor(s, gateCtx)
    .map((c) => c.to)
    .filter((t): t is Screen => Boolean(t) && ORDER.includes(t as Screen))
  const idx = steps.map((t) => ORDER.indexOf(t))
  check(`${s} path is in pipeline order`, idx.join(','), [...idx].sort((a, b) => a - b).join(','))
})

// --- Rail highlight per artboard ------------------------------------------
check('task rail is Detection', RAIL_FOR.task, 'detection')
check('run rail is Extraction', RAIL_FOR.run, 'extraction')
check('review rail', RAIL_FOR.review, 'review')

// Clicking a rail icon must land somewhere that lights that same icon.
;(Object.keys(RAIL_TARGET) as (keyof typeof RAIL_TARGET)[]).forEach((key) => {
  check(`rail ${key} round-trips`, RAIL_FOR[RAIL_TARGET[key]], key)
})

// --- Topbar chip carries run state, never a project name ------------------
check('chip on run', topbarChip('run', {}), 'Extraction')
check('chip on gate', topbarChip('review', gateCtx), 'Gate 4 of 4')
check('chip on detection names the sheet', topbarChip('detection', gateCtx), 'Detection · PDF-210-DEB p.2')
check('chip counts sheets', topbarChip('sheets', { sheetCount: 3 }), '3 sheets')
check('chip singular', topbarChip('sheets', { sheetCount: 1 }), '1 sheet')
check('chip empty with no sheets', topbarChip('sheets', { sheetCount: 0 }), '')
check('chip empty when unknown', topbarChip('sheets', {}), '')

if (failures > 0) throw new Error(`${failures} nav check(s) failed`)
console.log('nav: all checks passed')
