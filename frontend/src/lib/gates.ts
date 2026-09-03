import type { PipelineJob, PipelineStageManifest } from '@/types'

/**
 * The four human-in-the-loop gates.
 *
 * The pipeline deliberately has no stage 3: the CLI stage order is
 * 1, 2, 4, 5, 5b, 6, 7, 7c, 7b, 8, 9, 10, 11 and the gaps are where a person
 * has to look. Each gate is a `stop_after` boundary the backend parks on, and
 * each writes to a different store.
 */
export type GateId = 1 | 2 | 3 | 4

export type Gate = {
  id: GateId
  /** Stage the run parks after, i.e. the `stop_after` that produces this gate. */
  stopAfter: number
  /** Stage the run resumes from once the gate is confirmed. */
  resumeStage: string
  /** `stop_after` to pass on resume. Omitted means "let the backend decide". */
  resumeStopAfter?: number
  label: string
  stageLabel: string
}

export const GATES: Record<GateId, Gate> = {
  1: {
    id: 1,
    stopAfter: 4,
    resumeStage: 'stage5_pipe_mask',
    resumeStopAfter: 5,
    label: 'Objects',
    stageLabel: 'stage 4',
  },
  2: {
    id: 2,
    stopAfter: 5,
    resumeStage: 'stage6_trace_associations',
    resumeStopAfter: 6,
    label: 'Traces',
    stageLabel: 'stage 5b',
  },
  3: {
    id: 3,
    stopAfter: 6,
    resumeStage: 'stage7_geometric_graph_assembly',
    resumeStopAfter: 8,
    label: 'Line association',
    stageLabel: 'stage 6',
  },
  4: {
    id: 4,
    stopAfter: 8,
    resumeStage: 'stage9_apply_review_decisions',
    // Must be 11 explicitly. Without it the API falls back to
    // max(job.stop_after=8, 9) = 9, which silently truncates the run before
    // stage 10 (exports) and stage 11 (connection overlay).
    resumeStopAfter: 11,
    label: 'Graph QA',
    stageLabel: 'stage 8',
  },
}

const stage = (stages: PipelineStageManifest[], name: string) =>
  stages.find((s) => s.name === name)

const isComplete = (stages: PipelineStageManifest[], name: string) =>
  stage(stages, name)?.status === 'completed'

/**
 * Which gate, if any, this job is currently parked on.
 *
 * A gate opens only when the job has come to rest (`completed`) at the right
 * stage boundary, its prerequisite stage finished, and the stage it feeds has
 * not run yet. Mirrors the checks the previous UI made per gate.
 */
export function activeGate(
  job: Pick<PipelineJob, 'status' | 'stop_after'>,
  stages: PipelineStageManifest[]
): GateId | null {
  if (job.status !== 'completed') return null

  const stopAfter = job.stop_after ?? 4
  const stage5Started = stages.some((s) => (s.num ?? 0) >= 5 && s.status !== 'pending')

  if (!stage5Started && stopAfter <= 4) return 1
  if (isComplete(stages, 'stage5b_pipe_trace') && !isComplete(stages, 'stage6_trace_associations') && stopAfter <= 5) {
    return 2
  }
  if (isComplete(stages, 'stage6_trace_associations') && !isComplete(stages, 'stage7_geometric_graph_assembly') && stopAfter <= 6) {
    return 3
  }
  if (isComplete(stages, 'stage8_graph_qa') && !isComplete(stages, 'stage9_apply_review_decisions') && stopAfter <= 8) {
    return 4
  }
  return null
}

/** True once the run has produced its final exports and overlay. */
export function isRunComplete(stages: PipelineStageManifest[]): boolean {
  return isComplete(stages, 'stage10_process_exports')
}

/**
 * Stages the backend marked stale because a review write invalidated them.
 * Saving Gate 4 decisions, for instance, marks stage 9 onward stale.
 */
export function staleStages(stages: PipelineStageManifest[]): PipelineStageManifest[] {
  return stages.filter((s) => s.status === 'stale')
}

export function isStaleFrom(stages: PipelineStageManifest[], stageNum: number): boolean {
  return stages.some((s) => s.status === 'stale' && (s.num ?? 0) >= stageNum)
}

/** Percent for the run monitor. Clamped so a fresh run never reads as 0%. */
export function runPercent(job: Pick<PipelineJob, 'status' | 'manifest'>): number {
  if (job.status === 'completed') return 100
  const total = Math.max(job.manifest?.stages.length ?? job.manifest?.stop_after ?? 1, 1)
  const done = job.manifest?.stages.filter((s) => s.status === 'completed').length ?? 0
  return Math.min(95, Math.max(10, Math.round((done / total) * 100)))
}

export function humanStage(name: string | null | undefined): string {
  return (name ?? '').replaceAll('_', ' ')
}
