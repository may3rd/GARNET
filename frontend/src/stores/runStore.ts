import { create } from 'zustand'
import {
  APIError,
  extractPdfPages,
  getPipelineJob,
  getPipelineStageStatus,
  resumePipelineFromStage,
  startPipelineJob,
} from '@/lib/api'
import { activeGate, GATES, humanStage, runPercent, type GateId } from '@/lib/gates'
import type { OcrRoute, PipelineJob, PipelineStageManifest } from '@/types'

export type TaskKind = 'detection' | 'extraction'

/**
 * One sheet is one job. `POST /api/pipeline/jobs` takes a single file, so a
 * multi-page PDF is expanded into one sheet per page, each with its own job
 * and its own gate queue. Nothing merges until every sheet clears Gate 4.
 */
export type Sheet = {
  id: string
  file: File
  label: string
  previewUrl: string
  /** Natural pixel size, read off the decoded image. Null until it loads. */
  size: { width: number; height: number } | null
  /** Task and OCR route are per sheet: the intake table lets each differ. */
  task: TaskKind
  ocrRoute: OcrRoute
  jobId: string | null
  job: PipelineJob | null
  stages: PipelineStageManifest[]
  progress: { step: string; percent: number } | null
  error: string | null
}

export type RunConfig = {
  ocrRoute: OcrRoute
  weightFile: string
  geminiPostprocessMatchThreshold: number
  debugArtifacts: boolean
}

type RunState = {
  sheets: Sheet[]
  task: TaskKind
  config: RunConfig
  theme: 'default' | 'dark'
  isExtracting: boolean
  intakeError: string | null

  addFiles: (files: File[]) => Promise<void>
  setSheetTask: (id: string, task: TaskKind) => void
  setSheetOcrRoute: (id: string, route: OcrRoute) => void
  removeSheet: (id: string) => void
  clearSheets: () => void
  setTask: (task: TaskKind) => void
  setConfig: (patch: Partial<RunConfig>) => void
  toggleTheme: () => void
  startRun: () => Promise<void>
  resumeGate: (sheetId: string, gate: GateId) => Promise<void>
  gateFor: (sheetId: string) => GateId | null
}

const IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp']

let seq = 0
const nextId = () => `sheet_${Date.now().toString(36)}_${(seq += 1)}`

function base64ToFile(b64: string, name: string): File {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
  return new File([bytes], name, { type: 'image/png' })
}

function makeSheet(file: File, label: string, task: TaskKind, ocrRoute: OcrRoute): Sheet {
  return {
    id: nextId(),
    file,
    label,
    previewUrl: URL.createObjectURL(file),
    size: null,
    task,
    ocrRoute,
    jobId: null,
    job: null,
    stages: [],
    progress: null,
    error: null,
  }
}

/** Read natural dimensions so the intake table can show a real resolution. */
function measure(sheet: Sheet, set: SetFn) {
  const img = new Image()
  img.onload = () => {
    patchSheet(set, sheet.id, { size: { width: img.naturalWidth, height: img.naturalHeight } })
  }
  img.src = sheet.previewUrl
}

export const useRunStore = create<RunState>((set, get) => ({
  sheets: [],
  task: 'extraction',
  config: {
    // Matches the design's staged-sheet default and api.ts's own default.
    ocrRoute: 'ocrmac',
    weightFile: '',
    geminiPostprocessMatchThreshold: 0.1,
    debugArtifacts: false,
  },
  theme: 'default',
  isExtracting: false,
  intakeError: null,

  addFiles: async (files) => {
    set({ intakeError: null })
    const { task, config } = get()
    const added: Sheet[] = []

    for (const file of files) {
      if (file.type === 'application/pdf') {
        set({ isExtracting: true })
        try {
          const result = await extractPdfPages(file)
          const stem = file.name.replace(/\.pdf$/i, '')
          result.pages.forEach((page, index) => {
            const pageFile = base64ToFile(page, `${stem}_p${index + 1}.png`)
            added.push(makeSheet(pageFile, `${file.name} · p.${index + 1}`, task, config.ocrRoute))
          })
        } catch (error) {
          set({
            intakeError: error instanceof Error ? error.message : 'PDF extraction failed',
          })
        } finally {
          set({ isExtracting: false })
        }
        continue
      }

      if (!IMAGE_TYPES.includes(file.type)) {
        set({ intakeError: `Unsupported file type: ${file.name}. Use PDF, PNG, JPG or WEBP.` })
        continue
      }

      added.push(makeSheet(file, file.name, task, config.ocrRoute))
    }

    if (added.length > 0) {
      set((state) => ({ sheets: [...state.sheets, ...added] }))
      added.forEach((sheet) => measure(sheet, set))
    }
  },

  setSheetTask: (id, task) => patchSheet(set, id, { task }),
  setSheetOcrRoute: (id, ocrRoute) => patchSheet(set, id, { ocrRoute }),

  removeSheet: (id) =>
    set((state) => {
      const sheet = state.sheets.find((s) => s.id === id)
      if (sheet) URL.revokeObjectURL(sheet.previewUrl)
      return { sheets: state.sheets.filter((s) => s.id !== id) }
    }),

  clearSheets: () =>
    set((state) => {
      state.sheets.forEach((s) => URL.revokeObjectURL(s.previewUrl))
      return { sheets: [] }
    }),

  setTask: (task) => set({ task }),

  setConfig: (patch) => set((state) => ({ config: { ...state.config, ...patch } })),

  toggleTheme: () => {
    const next = get().theme === 'dark' ? 'default' : 'dark'
    document.documentElement.setAttribute('data-theme', next)
    set({ theme: next })
  },

  startRun: async () => {
    const { sheets, config } = get()
    if (sheets.length === 0) return

    await Promise.all(
      sheets.map(async (sheet) => {
        if (sheet.jobId) return
        patchSheet(set, sheet.id, {
          error: null,
          progress: { step: 'Creating job…', percent: 5 },
        })
        try {
          const { job_id } = await startPipelineJob(sheet.file, {
            // Extraction parks at Gate 1 (stop_after 4) for the object review.
            // Detection is the legacy single-pass route and does not gate.
            stopAfter: GATES[1].stopAfter,
            ocrRoute: sheet.ocrRoute,
            geminiPostprocessMatchThreshold: config.geminiPostprocessMatchThreshold,
            weightFile: config.weightFile,
            debugArtifacts: config.debugArtifacts,
          })
          patchSheet(set, sheet.id, { jobId: job_id })
          await pollUntilRest(set, sheet.id, job_id)
        } catch (error) {
          patchSheet(set, sheet.id, {
            progress: null,
            error:
              error instanceof APIError && error.isCanceled
                ? 'Canceled'
                : error instanceof Error
                  ? error.message
                  : 'Failed to start job',
          })
        }
      })
    )
  },

  resumeGate: async (sheetId, gate) => {
    const sheet = get().sheets.find((s) => s.id === sheetId)
    if (!sheet?.jobId) return
    const { resumeStage, resumeStopAfter } = GATES[gate]

    patchSheet(set, sheetId, {
      error: null,
      progress: { step: `Resuming from ${humanStage(resumeStage)}`, percent: 8 },
    })
    try {
      await resumePipelineFromStage(sheet.jobId, resumeStage, { stopAfter: resumeStopAfter })
      await pollUntilRest(set, sheetId, sheet.jobId)
    } catch (error) {
      patchSheet(set, sheetId, {
        progress: null,
        error: error instanceof Error ? error.message : 'Resume failed',
      })
    }
  },

  gateFor: (sheetId) => {
    const sheet = get().sheets.find((s) => s.id === sheetId)
    if (!sheet?.job) return null
    return activeGate(sheet.job, sheet.stages)
  },
}))

type SetFn = (fn: (state: RunState) => Partial<RunState>) => void

function patchSheet(set: SetFn, id: string, patch: Partial<Sheet>) {
  set((state) => ({
    sheets: state.sheets.map((s) => (s.id === id ? { ...s, ...patch } : s)),
  }))
}

/** Poll a job until it stops moving. 400ms matches the previous UI's cadence. */
async function pollUntilRest(set: SetFn, sheetId: string, jobId: string) {
  for (;;) {
    const job = await getPipelineJob(jobId)

    let stages = job.manifest?.stages ?? []
    try {
      stages = (await getPipelineStageStatus(jobId)).stages
    } catch {
      // stage-status is a convenience view; the manifest is authoritative.
    }

    patchSheet(set, sheetId, {
      job,
      stages,
      progress: {
        step: job.status === 'queued' ? 'Queued…' : humanStage(job.current_stage) || 'Running…',
        percent: runPercent(job),
      },
    })

    if (job.status === 'completed') {
      patchSheet(set, sheetId, { progress: null })
      return
    }
    if (job.status === 'failed') {
      patchSheet(set, sheetId, { progress: null, error: job.error || 'Pipeline failed' })
      return
    }

    await new Promise((resolve) => window.setTimeout(resolve, 400))
  }
}
