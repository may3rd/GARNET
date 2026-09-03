/** Runnable check: `bun run src/lib/detectionClasses.test.ts` */
import { classColor, normalizeClass, summarizeClasses } from './detectionClasses'
import type { DetectedObject } from '@/types'

let failures = 0
function check(label: string, actual: unknown, expected: unknown) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures += 1
    console.error(`FAIL ${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
  }
}

const obj = (Object_: string, i: number): DetectedObject => ({
  Index: i,
  Object: Object_,
  CategoryID: 0,
  ObjectID: i,
  Left: 0,
  Top: 0,
  Width: 10,
  Height: 10,
  Score: 0.9,
  Text: '',
})

// --- Normalising the names the backend actually returns -------------------
check('underscores', normalizeClass('line_number'), 'line number')
check('case', normalizeClass('Instrument DCS'), 'instrument dcs')
check('whitespace', normalizeClass('  pump '), 'pump')

// --- The design's named palette ------------------------------------------
check('gate valve', classColor('gate valve'), '#dc2626')
check('gate valve via raw casing', classColor('Gate Valve'), '#dc2626')
check('line number via underscore', classColor('line_number'), '#0891b2')

// --- Unnamed classes still get a distinct, stable colour -----------------
// These are real classes the current weights emit that no config names.
const unnamed = ['node', 'arrow', 'reducer', 'spectacle blind', 'strainer', 'sampling point']
unnamed.forEach((n) => {
  check(`${n} is stable`, classColor(n), classColor(n))
  check(`${n} is not the accent fallback`, classColor(n).startsWith('#'), true)
})
// At least most of them must differ from each other, or the legend is useless.
const distinct = new Set(unnamed.map(classColor)).size
check('unnamed classes are mostly distinct', distinct >= 4, true)

// --- Summary ordering -----------------------------------------------------
const objects = [
  obj('gate valve', 1),
  obj('gate valve', 2),
  obj('gate valve', 3),
  obj('pump', 4),
  obj('line_number', 5),
  obj('line number', 6),
]
const summary = summarizeClasses(objects)
check('counts and order', summary.map((s) => [s.name, s.count]), [
  ['gate valve', 3],
  ['line number', 2],
  ['pump', 1],
])
check('summary carries colour', summary[0].color, '#dc2626')
check('total is preserved', summary.reduce((n, s) => n + s.count, 0), objects.length)

if (failures > 0) throw new Error(`${failures} detectionClasses check(s) failed`)
console.log('detectionClasses: all checks passed')
