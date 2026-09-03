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
  project: string
  /** Label of the sheet in focus, for the sheet-scoped screens. */
  sheetLabel?: string
  /** Gate in focus on the review screen. */
  gate?: GateId | null
}

/**
 * Breadcrumbs follow the artboards: project › (run | sheet) › screen.
 * Sheet-scoped screens (detection results, gates) put the sheet in the
 * middle; run-scoped ones put "New run".
 */
export function breadcrumbFor(screen: Screen, ctx: NavContext): Crumb[] {
  const project = ctx.project
  const sheet = ctx.sheetLabel

  switch (screen) {
    case 'sheets':
      return [{ label: 'Projects' }, { label: project }, { label: 'New run' }]
    case 'task':
      return [{ label: project, to: 'sheets' }, { label: 'New run', to: 'sheets' }, { label: 'Task' }]
    case 'detection':
      return [
        { label: project, to: 'sheets' },
        // Whether it names a sheet or falls back to the run, this segment is
        // the way back to intake.
        { label: sheet ?? 'New run', to: 'sheets' },
        { label: 'Detection results' },
      ]
    case 'run':
      return [
        { label: project, to: 'sheets' },
        { label: 'New run', to: 'sheets' },
        { label: 'Extraction' },
      ]
    case 'review':
      return [
        { label: project, to: 'sheets' },
        { label: sheet ?? 'New run', to: 'run' },
        { label: ctx.gate ? `Gate ${ctx.gate}` : 'Review' },
      ]
    case 'merge':
      return [
        { label: project, to: 'sheets' },
        { label: 'New run', to: 'sheets' },
        { label: 'Merge' },
      ]
    case 'exports':
      return [
        { label: project, to: 'sheets' },
        { label: 'New run', to: 'sheets' },
        { label: 'Exports' },
      ]
  }
}

/** The contextual chip on the right of the topbar, per the artboards. */
export function topbarChip(screen: Screen, ctx: NavContext): string {
  switch (screen) {
    case 'detection':
      return ctx.sheetLabel ? `Detection ${ctx.sheetLabel}` : 'Detection'
    case 'run':
      return 'Extraction running'
    case 'review':
      return ctx.gate ? `Gate ${ctx.gate} of 4` : 'Review'
    default:
      return ctx.project
  }
}
