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
// Alphabetical by name, not by how many were found — a legend you scan for a
// class you already have in mind, so the order has to be predictable.
check('counts and alphabetical order', summary.map((s) => [s.name, s.count]), [
  ['gate valve', 3],
  ['line number', 2],
  ['pump', 1],
])
check('summary carries colour', summary[0].color, '#dc2626')
check('total is preserved', summary.reduce((n, s) => n + s.count, 0), objects.length)

// The fixture above is alphabetical and count-descending at once, so it
// cannot tell the two apart. This one can: the most numerous class sorts last.
const byName = summarizeClasses([
  obj('zebra valve', 1),
  obj('zebra valve', 2),
  obj('zebra valve', 3),
  obj('apple valve', 4),
])
check('name wins over count', byName.map((s) => s.name), ['apple valve', 'zebra valve'])
check('counts still correct', byName.map((s) => s.count), [1, 3])

if (failures > 0) throw new Error(`${failures} detectionClasses check(s) failed`)
console.log('detectionClasses: all checks passed')
