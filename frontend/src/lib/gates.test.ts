/**
 * Runnable check for the gate state machine: `bun run src/lib/gates.test.ts`
 *
 * These assertions encode the rules the previous UI enforced. The one that
 * matters most is the last: Gate 4 must resume with stop_after 11, or exports
 * and the connection overlay are silently skipped.
 */
import { activeGate, firstLegStopAfter, GATES, isRunComplete, isStaleFrom, runPercent } from './gates'
import type { PipelineStageManifest } from '@/types'

const s = (
  num: number,
  name: string,
  status: PipelineStageManifest['status'] = 'completed'
): PipelineStageManifest => ({ num, name, status })

let failures = 0
function check(label: string, actual: unknown, expected: unknown) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected)
  if (!ok) {
    failures += 1
    console.error(`FAIL ${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
  }
}

// --- Gate 1: parked after stage 4, nothing at or past stage 5 has run -------
const afterStage4 = [s(1, 'stage1_input_normalization'), s(2, 'stage2_ocr_discovery'), s(4, 'stage4_object_detection'), s(5, 'stage5_pipe_mask', 'pending')]
check('gate 1 open', activeGate({ status: 'completed', stop_after: 4 }, afterStage4), 1)
check('no gate while running', activeGate({ status: 'running', stop_after: 4 }, afterStage4), null)
check('no gate when failed', activeGate({ status: 'failed', stop_after: 4 }, afterStage4), null)

// --- Gate 2: 5b done, 6 not -------------------------------------------------
const afterStage5b = [...afterStage4.slice(0, 3), s(5, 'stage5_pipe_mask'), s(5, 'stage5b_pipe_trace'), s(6, 'stage6_trace_associations', 'pending')]
check('gate 2 open', activeGate({ status: 'completed', stop_after: 5 }, afterStage5b), 2)

// --- Gate 3: 6 done, 7 not --------------------------------------------------
const afterStage6 = [...afterStage5b.slice(0, 5), s(6, 'stage6_trace_associations'), s(7, 'stage7_geometric_graph_assembly', 'pending')]
check('gate 3 open', activeGate({ status: 'completed', stop_after: 6 }, afterStage6), 3)

// --- Gate 4: 8 done, 9 not --------------------------------------------------
const afterStage8 = [...afterStage6.slice(0, 6), s(7, 'stage7_geometric_graph_assembly'), s(8, 'stage8_graph_qa'), s(9, 'stage9_apply_review_decisions', 'pending')]
check('gate 4 open', activeGate({ status: 'completed', stop_after: 8 }, afterStage8), 4)

// Once stage 9 has run, no gate is open.
const afterStage9 = [...afterStage8.slice(0, 8), s(9, 'stage9_apply_review_decisions'), s(10, 'stage10_process_exports')]
check('no gate after stage 9', activeGate({ status: 'completed', stop_after: 11 }, afterStage9), null)
check('run complete', isRunComplete(afterStage9), true)
check('run not complete at gate 4', isRunComplete(afterStage8), false)

// --- Staleness --------------------------------------------------------------
const staleFrom9 = [...afterStage8.slice(0, 8), s(9, 'stage9_apply_review_decisions', 'stale')]
check('stale from 9 detected', isStaleFrom(staleFrom9, 9), true)
check('not stale below 9', isStaleFrom(staleFrom9, 10), false)

// --- Progress clamping ------------------------------------------------------
check('percent clamps low', runPercent({ status: 'running', manifest: { stages: [], stop_after: 11 } as never }), 10)
check('percent 100 when complete', runPercent({ status: 'completed', manifest: null }), 100)

// --- First-leg stop_after (Task fork's two switches) ------------------------
check('pausing parks at gate 1', firstLegStopAfter({ pauseAtEveryGate: true, stopAfterStage: 11 }), 4)
check('not pausing runs to QA', firstLegStopAfter({ pauseAtEveryGate: false, stopAfterStage: 11 }), 8)
// A lower ceiling always wins, so the run cannot overshoot what was asked for.
check('ceiling below gate 1', firstLegStopAfter({ pauseAtEveryGate: true, stopAfterStage: 2 }), 2)
check('ceiling below QA', firstLegStopAfter({ pauseAtEveryGate: false, stopAfterStage: 5 }), 5)

// --- The resume contract (the bug that was fixed upstream) ------------------
check('gate 4 resumes through stage 11', GATES[4].resumeStopAfter, 11)
check('gate 4 resumes at stage 9', GATES[4].resumeStage, 'stage9_apply_review_decisions')

if (failures > 0) {
  throw new Error(`${failures} gate check(s) failed`)
}
console.log('gates: all checks passed')
