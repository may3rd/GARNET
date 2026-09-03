import type { GateId } from '@/lib/gates'

/**
 * Screens, one per canvas artboard. The rail has six entries but nine
 * artboards: Gates 1-4 all live under Review, and merge/exports share
 * artboard 9.
 */
export type Screen =
  | 'sheets' // 1 · Main
  | 'task' // 2 · TaskFork
  | 'detection' // 3 · DetectionRun
  | 'run' // 4 · ExtractionRun
  | 'review' // 5-8 · Gate1..Gate4
  | 'merge' // 9 · Exports and merge
  | 'exports' // 9 · Exports and merge

export type RailKey = 'sheets' | 'detection' | 'extraction' | 'review' | 'merge' | 'exports'

/** Which rail icon lights up per screen. Taken from the artboards. */
export const RAIL_FOR: Record<Screen, RailKey> = {
  sheets: 'sheets',
  task: 'detection',
  detection: 'detection',
  run: 'extraction',
  review: 'review',
  merge: 'merge',
  exports: 'exports',
}

/** Where each rail icon goes. */
export const RAIL_TARGET: Record<RailKey, Screen> = {
  sheets: 'sheets',
  detection: 'detection',
  extraction: 'run',
  review: 'review',
  merge: 'merge',
  exports: 'exports',
}

export const RAIL_ORDER: RailKey[] = [
  'sheets',
  'detection',
  'extraction',
  'review',
  'merge',
  'exports',
]

export const RAIL_LABEL: Record<RailKey, string> = {
  sheets: 'Sheets',
  detection: 'Detection',
  extraction: 'Extraction',
  review: 'Review',
  merge: 'Merge',
  exports: 'Exports',
}

/** A breadcrumb segment. `to` omitted means "you are here" (not a link). */
export type Crumb = { label: string; to?: Screen }

export type NavContext = {
  /** Label of the sheet in focus, for the sheet-scoped screens. */
  sheetLabel?: string
  /** Gate in focus on the review screen. */
  gate?: GateId | null
  /** How many sheets are staged, for the topbar chip. */
  sheetCount?: number
}

/**
 * The breadcrumb is the pipeline path, not a project tree: there is no project
 * entity anywhere in the backend (no user model, an in-memory job store, and
 * API_KEY_ENABLED as the only notion of auth), so a "Projects › Unit 210"
 * hierarchy would be inventing structure that does not exist.
 *
 * What a person actually needs to know is where they are along the pipeline
 * and how to step back:
 *
 *   Sheets › Task › Run › <sheet> › Gate N
 *
 * Only the gate screens name a sheet, because only they act on one sheet at a
 * time; the run-wide screens hang off Run.
 */
const SHEETS: Crumb = { label: 'Sheets', to: 'sheets' }
const TASK: Crumb = { label: 'Task', to: 'task' }
const RUN: Crumb = { label: 'Run', to: 'run' }

export function breadcrumbFor(screen: Screen, ctx: NavContext): Crumb[] {
  const sheet = ctx.sheetLabel

  switch (screen) {
    case 'sheets':
      return [{ label: 'Sheets' }]
    case 'task':
      return [SHEETS, { label: 'Task' }]
    case 'detection':
      // Sheet-scoped, but it never went through a run, so it hangs off Task.
      return [SHEETS, TASK, { label: 'Detection results' }]
    case 'run':
      return [SHEETS, TASK, { label: 'Run' }]
    case 'review':
      return [
        SHEETS,
        TASK,
        RUN,
        ...(sheet ? [{ label: sheet, to: 'run' as Screen }] : []),
        { label: ctx.gate ? `Gate ${ctx.gate}` : 'Review' },
      ]
    case 'merge':
      return [SHEETS, TASK, RUN, { label: 'Merge' }]
    case 'exports':
      return [SHEETS, TASK, RUN, { label: 'Exports' }]
  }
}

/**
 * The contextual chip on the right of the topbar. With no project to name, it
 * carries the run's own state instead. Empty string means "show nothing".
 */
export function topbarChip(screen: Screen, ctx: NavContext): string {
  switch (screen) {
    case 'detection':
      return ctx.sheetLabel ? `Detection · ${ctx.sheetLabel}` : 'Detection'
    case 'run':
      return 'Extraction'
    case 'review':
      return ctx.gate ? `Gate ${ctx.gate} of 4` : 'Review'
    default: {
      const n = ctx.sheetCount ?? 0
      return n === 0 ? '' : `${n} sheet${n === 1 ? '' : 's'}`
    }
  }
}
