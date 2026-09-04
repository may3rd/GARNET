import { create } from 'zustand'
import {
  APIError,
  createResultObject,
  deleteResultObject,
  extractPdfPages,
  getPipelineJob,
  getWeightFiles,
  getPipelineStageStatus,
  resumePipelineFromStage,
  runDetection as runDetectionApi,
  startPipelineJob,
  updateResultObject,
} from '@/lib/api'
import { activeGate, firstLegStopAfter, GATES, humanStage, runPercent, type GateId } from '@/lib/gates'
import type { Screen } from '@/lib/nav'
import type { DetectedObject, DetectionResult, OcrRoute, PipelineJob, PipelineStageManifest } from '@/types'

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
  /** Detection-task result. Held per sheet so each keeps its own boxes. */
  detection: DetectionResult | null
}

export type RunConfig = {
  ocrRoute: OcrRoute
  weightFile: string
  geminiPostprocessMatchThreshold: number
  debugArtifacts: boolean
  /** Final stage the run targets. One of the API's accepted stop_after values. */
  stopAfterStage: number
  /**
   * On: park at each gate boundary in turn (first leg stops after stage 4).
   * Off: run straight through to stage 8 and queue all four gates at once.
   */
  pauseAtEveryGate: boolean
  /** Detection-task settings (the legacy /api/detect path). */
  confTh: number
  imageSize: number
  overlapRatio: number
  textOCR: boolean
}

type RunState = {
  screen: Screen
  /** Sheet in focus for the sheet-scoped screens (detection results, gates). */
  selectedSheetId: string | null
  sheets: Sheet[]
  /** Run-level task, chosen on the Task fork screen. Sheets inherit it. */
  task: TaskKind
  config: RunConfig
  theme: 'default' | 'dark'
  isExtracting: boolean
  intakeError: string | null
  /** Weight files the server actually has, from GET /api/weight-files. */
  weightFiles: string[]

  setScreen: (screen: Screen) => void
  selectSheet: (id: string | null) => void
  loadWeightFiles: () => Promise<void>
  addFiles: (files: File[]) => Promise<void>
  setSheetTask: (id: string, task: TaskKind) => void
  setSheetOcrRoute: (id: string, route: OcrRoute) => void
  removeSheet: (id: string) => void
  clearSheets: () => void
  /** Sets the run-level task and re-assigns every sheet that has not started. */
  setTask: (task: TaskKind) => void
  setConfig: (patch: Partial<RunConfig>) => void
  toggleTheme: () => void
  startRun: () => Promise<void>
  resumeGate: (sheetId: string, gate: GateId) => Promise<void>
  gateFor: (sheetId: string) => GateId | null

  /** Detection task: POST /api/detect for one sheet, or re-run it. */
  runDetectionFor: (sheetId: string) => Promise<void>
  /** Object edits persist to the server's in-memory result store. */
  updateObject: (sheetId: string, obj: DetectedObject) => Promise<void>
  deleteObject: (sheetId: string, obj: DetectedObject) => Promise<void>
  addObject: (
    sheetId: string,
    // Mirrors the backend's CreateObjectRequest: CategoryID/ObjectID/Score are
    // genuinely optional there (the server assigns ObjectID), not just absent
    // from this particular call.
    obj: Pick<DetectedObject, 'Object' | 'Left' | 'Top' | 'Width' | 'Height' | 'Text'> &
      Partial<Pick<DetectedObject, 'CategoryID' | 'ObjectID' | 'Score'>>
  ) => Promise<DetectedObject | undefined>
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
    detection: null,
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
  screen: 'sheets',
  selectedSheetId: null,
  sheets: [],
  task: 'extraction',
  config: {
    // Matches the design's staged-sheet default and api.ts's own default.
    ocrRoute: 'ocrmac',
    weightFile: '',
    geminiPostprocessMatchThreshold: 0.1,
    debugArtifacts: false,
    stopAfterStage: 11,
    pauseAtEveryGate: true,
    confTh: 0.8,
    imageSize: 640,
    overlapRatio: 0.2,
    textOCR: false,
  },
  theme: 'default',
  isExtracting: false,
  intakeError: null,
  weightFiles: [],

  setScreen: (screen) => set({ screen }),

  selectSheet: (selectedSheetId) => set({ selectedSheetId }),

  loadWeightFiles: async () => {
    if (get().weightFiles.length > 0) return
    try {
      set({ weightFiles: await getWeightFiles() })
    } catch {
      // A missing list is not worth an error banner; the field falls back to
      // the server default.
    }
  },

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
      return {
        sheets: state.sheets.filter((s) => s.id !== id),
        selectedSheetId: state.selectedSheetId === id ? null : state.selectedSheetId,
      }
    }),

  clearSheets: () =>
    set((state) => {
      state.sheets.forEach((s) => URL.revokeObjectURL(s.previewUrl))
      return { sheets: [], selectedSheetId: null }
    }),

  setTask: (task) =>
    set((state) => ({
      task,
      // Sheets inherit the run-level choice; started jobs keep what they ran with.
      sheets: state.sheets.map((s) => (s.jobId ? s : { ...s, task })),
    })),

  setConfig: (patch) => set((state) => ({ config: { ...state.config, ...patch } })),

  toggleTheme: () => {
    const next = get().theme === 'dark' ? 'default' : 'dark'
    document.documentElement.setAttribute('data-theme', next)
    set({ theme: next })
  },

  startRun: async () => {
    const { sheets, config } = get()
    if (sheets.length === 0) return

    const stopAfter = firstLegStopAfter(config)

    await Promise.all(
      sheets.map(async (sheet) => {
        if (sheet.jobId) return
        if (sheet.task === 'detection') {
          await get().runDetectionFor(sheet.id)
          return
        }
        patchSheet(set, sheet.id, {
          error: null,
          progress: { step: 'Creating job…', percent: 5 },
        })
        try {
          const { job_id } = await startPipelineJob(sheet.file, {
            stopAfter,
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

  runDetectionFor: async (sheetId) => {
    const { sheets, config } = get()
    const sheet = sheets.find((s) => s.id === sheetId)
    if (!sheet) return

    patchSheet(set, sheetId, {
      error: null,
      progress: { step: 'Detecting…', percent: 20 },
    })
    try {
      const result = await runDetectionApi(sheet.file, {
        confTh: config.confTh,
        imageSize: config.imageSize,
        overlapRatio: config.overlapRatio,
        textOCR: config.textOCR,
        weightFile: config.weightFile,
      })
      patchSheet(set, sheetId, { detection: result, progress: null })
    } catch (error) {
      patchSheet(set, sheetId, {
        progress: null,
        error:
          error instanceof APIError && error.isCanceled
            ? 'Canceled'
            : error instanceof Error
              ? error.message
              : 'Detection failed',
      })
    }
  },

  updateObject: async (sheetId, obj) => {
    const sheet = get().sheets.find((s) => s.id === sheetId)
    if (!sheet?.detection) return
    // Optimistic: the panel should not lag a keystroke behind the canvas.
    patchDetection(set, sheetId, (objects) =>
      objects.map((o) => (o.Index === obj.Index ? obj : o))
    )
    try {
      await updateResultObject(sheet.detection.id, obj.Index, obj)
    } catch (error) {
      patchSheet(set, sheetId, {
        error: error instanceof Error ? error.message : 'Could not save the edit',
      })
    }
  },

  deleteObject: async (sheetId, obj) => {
    const sheet = get().sheets.find((s) => s.id === sheetId)
    if (!sheet?.detection) return
    patchDetection(set, sheetId, (objects) => objects.filter((o) => o.Index !== obj.Index))
    try {
      await deleteResultObject(sheet.detection.id, obj.Index)
    } catch (error) {
      patchSheet(set, sheetId, {
        error: error instanceof Error ? error.message : 'Could not delete the object',
      })
    }
  },

  addObject: async (sheetId, obj) => {
    const sheet = get().sheets.find((s) => s.id === sheetId)
    if (!sheet?.detection) return
    try {
      const created = await createResultObject(sheet.detection.id, obj)
      patchDetection(set, sheetId, (objects) => [...objects, created])
      return created
    } catch (error) {
      patchSheet(set, sheetId, {
        error: error instanceof Error ? error.message : 'Could not add the object',
      })
    }
  },
}))

type SetFn = (fn: (state: RunState) => Partial<RunState>) => void

/** Replace a sheet's detection objects, keeping count in sync. */
function patchDetection(
  set: SetFn,
  id: string,
  fn: (objects: DetectedObject[]) => DetectedObject[]
) {
  set((state) => ({
    sheets: state.sheets.map((s) => {
      if (s.id !== id || !s.detection) return s
      const objects = fn(s.detection.objects)
      return { ...s, detection: { ...s.detection, objects, count: objects.length } }
    }),
  }))
}

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
