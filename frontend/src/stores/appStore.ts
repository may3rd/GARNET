import { create } from 'zustand'
import type { AppView, BatchItem, DetectedObject, DetectionResult } from '@/types'
import { APIError, runDetection, type DetectionOptions } from '@/lib/api'
import { useHistoryStore, type HistoryAction } from '@/stores/historyStore'
import { objectKey } from '@/lib/objectKey'

export type AppState = {
  currentView: AppView
  imageFile: File | null
  imageUrl: string | null
  imageMeta: { width: number; height: number } | null
  options: DetectionOptions
  result: DetectionResult | null
  resultRunId: number
  reviewStatus: Record<string, 'accepted' | 'rejected'>
  selectedObjectKey: string | null
  confidenceFilter: number
  isProcessing: boolean
  progress: { step: string; percent: number } | null
  error: string | null
  darkMode: boolean
  batch: {
    items: BatchItem[]
    isRunning: boolean
    optionsSnapshot: DetectionOptions | null
    activeItemId: string | null
    locked: boolean
    paused: boolean
  }
}

export type AppActions = {
  setImageFile: (file: File | null) => void
  setBatchFiles: (files: File[]) => void
  addBatchFiles: (files: File[]) => void
  removeBatchItem: (id: string) => void
  resetBatchItem: (id: string) => void
  clearBatch: () => void
  setImageMeta: (meta: { width: number; height: number } | null) => void
  setOptions: (options: Partial<DetectionOptions>) => void
  setView: (view: AppView) => void
  runDetection: () => Promise<void>
  runBatchDetection: () => Promise<void>
  cancelDetection: () => void
  cancelBatch: () => void
  toggleBatchPause: () => void
  openBatchResult: (id: string) => void
  openNextBatchResult: () => void
  openPrevBatchResult: () => void
  retryBatchFailed: () => Promise<void>
  goBack: () => void
  toggleTheme: () => void
  setReviewStatus: (key: string, status: 'accepted' | 'rejected' | null) => void
  setSelectedObjectKey: (key: string | null) => void
  setConfidenceFilter: (value: number) => void
  addObject: (obj: DetectionResult['objects'][number]) => void
  updateObject: (updated: DetectionResult['objects'][number]) => void
  removeObject: (index: number) => void
  undoAction: () => void
  redoAction: () => void
  restoreObject: (obj: DetectedObject, atIndex?: number) => void
}

const defaultOptions: DetectionOptions = {
  selectedModel: 'ultralytics',
  weightFile: '',
  configFile: 'datasets/yaml/data.yaml',
  confTh: 0.8,
  imageSize: 640,
  overlapRatio: 0.2,
  textOCR: false,
}

const THEME_KEY = 'garnet-theme'

const initialDarkMode = (() => {
  if (typeof window === 'undefined') return false
  const stored = window.localStorage.getItem(THEME_KEY)
  return stored === 'dark'
})()

if (initialDarkMode && typeof document !== 'undefined') {
  document.documentElement.classList.add('dark')
}

let progressTimer: number | null = null
let activeAbortController: AbortController | null = null
let batchAbortController: AbortController | null = null

const emptyBatchState = {
  items: [],
  isRunning: false,
  optionsSnapshot: null,
  activeItemId: null,
  locked: false,
  paused: false,
}

const createBatchItemId = () => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

const buildReviewMap = (result: DetectionResult | null) => {
  const reviewFromPayload: Record<string, 'accepted' | 'rejected'> = {}
  if (!result) return reviewFromPayload
  result.objects.forEach((obj) => {
    const status = obj.ReviewStatus
    if (status === 'accepted' || status === 'rejected') {
      reviewFromPayload[`${obj.CategoryID}-${obj.ObjectID}-${obj.Index}`] = status
    }
  })
  return reviewFromPayload
}

export const useAppStore = create<AppState & AppActions>((set, get) => ({
  currentView: 'empty',
  imageFile: null,
  imageUrl: null,
  imageMeta: null,
  options: defaultOptions,
  result: null,
  resultRunId: 0,
  reviewStatus: {},
  selectedObjectKey: null,
  confidenceFilter: defaultOptions.confTh,
  isProcessing: false,
  progress: null,
  error: null,
  darkMode: initialDarkMode,
  batch: emptyBatchState,

  setConfidenceFilter: (value) => set({ confidenceFilter: value }),

  setImageFile: (file) => {
    const previous = get().imageUrl
    if (previous) {
      URL.revokeObjectURL(previous)
    }

    if (!file) {
      set({
        imageFile: null,
        imageUrl: null,
        imageMeta: null,
        result: null,
        currentView: 'empty',
        batch: emptyBatchState,
      })
      return
    }

    const url = URL.createObjectURL(file)
    set({
      imageFile: file,
      imageUrl: url,
      currentView: 'preview',
      batch: emptyBatchState,
    })
  },

  setBatchFiles: (files) => {
    const previous = get().imageUrl
    if (previous) {
      URL.revokeObjectURL(previous)
    }
    const items: BatchItem[] = files.map((file) => ({
      id: createBatchItemId(),
      file,
      fileName: file.name,
      status: 'queued',
    }))
    set({
      imageFile: null,
      imageUrl: null,
      imageMeta: null,
      result: null,
      reviewStatus: {},
      selectedObjectKey: null,
      currentView: 'batch',
      batch: {
        items,
        isRunning: false,
        optionsSnapshot: null,
        activeItemId: null,
        locked: false,
        paused: false,
      },
      error: null,
      progress: null,
    })
  },

  clearBatch: () => {
    if (batchAbortController) {
      batchAbortController.abort()
    }
    set({ batch: emptyBatchState })
  },

  addBatchFiles: (files) => {
    if (!files.length) return
    set((state) => ({
      currentView: 'batch',
      batch: {
        ...state.batch,
        items: [
          ...state.batch.items,
          ...files.map((file) => ({
            id: createBatchItemId(),
            file,
            fileName: file.name,
            status: 'queued',
          })),
        ],
      },
    }))
  },

  removeBatchItem: (id) => {
    set((state) => {
      const items = state.batch.items.filter((item) => item.id !== id)
      const wasActive = state.batch.activeItemId === id
      const nextActive = wasActive ? null : state.batch.activeItemId
      return {
        batch: {
          ...state.batch,
          items,
          activeItemId: nextActive,
        },
        ...(wasActive
          ? {
            result: null,
            reviewStatus: {},
            selectedObjectKey: null,
            currentView: state.currentView === 'results' ? 'batch' : state.currentView,
          }
          : {}),
      }
    })
  },

  resetBatchItem: (id) => {
    set((state) => ({
      batch: {
        ...state.batch,
        items: state.batch.items.map((item) =>
          item.id === id
            ? { ...item, status: 'queued', result: undefined, error: undefined }
            : item
        ),
      },
    }))
  },

  setImageMeta: (meta) => set({ imageMeta: meta }),

  setOptions: (options) =>
    set((state) => {
      if (state.batch.locked) return state
      return { options: { ...state.options, ...options }, error: null }
    }),

  setView: (view) => set({ currentView: view }),

  goBack: () => {
    const { currentView, batch } = get()
    if (currentView === 'results' && batch.items.length > 0) {
      set({ currentView: 'batch' })
      return
    }
    if (currentView === 'batch') {
      set({
        batch: emptyBatchState,
        imageFile: null,
        imageUrl: null,
        imageMeta: null,
        result: null,
        reviewStatus: {},
        selectedObjectKey: null,
        currentView: 'empty',
      })
      return
    }
    get().setImageFile(null)
  },

  runDetection: async () => {
    const { imageFile, options } = get()
    if (!imageFile) return

    if (progressTimer) {
      window.clearInterval(progressTimer)
    }
    if (activeAbortController) {
      activeAbortController.abort()
    }
    activeAbortController = new AbortController()

    set({
      isProcessing: true,
      currentView: 'processing',
      error: null,
      progress: { step: 'Uploading image...', percent: 8 },
    })

    let percent = 8
    progressTimer = window.setInterval(() => {
      percent = Math.min(percent + 4, 92)
      const step = percent < 40
        ? 'Running detection...'
        : percent < 75
          ? 'Extracting objects...'
          : 'Finalizing results...'
      set({ progress: { step, percent } })
    }, 500)

    try {
      const result = await runDetection(imageFile, options, activeAbortController.signal)
      const reviewFromPayload = buildReviewMap(result)
      set({
        result,
        resultRunId: Date.now(),
        reviewStatus: reviewFromPayload,
        selectedObjectKey: null,
        isProcessing: false,
        currentView: 'results',
        progress: { step: 'Complete', percent: 100 },
        batch: { ...get().batch, activeItemId: null },
      })
    } catch (error) {
      if (error instanceof APIError && error.isCanceled) {
        set({
          isProcessing: false,
          currentView: 'preview',
          error: 'Detection canceled',
          progress: null,
        })
        return
      }
      set({
        isProcessing: false,
        currentView: 'preview',
        error: error instanceof Error ? error.message : 'Detection failed',
        progress: null,
      })
    } finally {
      if (progressTimer) {
        window.clearInterval(progressTimer)
        progressTimer = null
      }
      activeAbortController = null
    }
  },

  cancelDetection: () => {
    if (activeAbortController) {
      activeAbortController.abort()
    }
  },

  runBatchDetection: async () => {
    const { batch, options } = get()
    if (batch.isRunning || batch.items.length === 0) return

    if (batchAbortController) {
      batchAbortController.abort()
    }
    batchAbortController = new AbortController()

    const snapshot: DetectionOptions = { ...options }
    set((state) => ({
      batch: {
        ...state.batch,
        isRunning: true,
        locked: true,
        paused: false,
        optionsSnapshot: snapshot,
      },
      currentView: 'batch',
      error: null,
    }))

    for (const item of get().batch.items) {
      if (batchAbortController.signal.aborted) {
        set((state) => ({
          batch: {
            ...state.batch,
            items: state.batch.items.map((entry) =>
              entry.status === 'running' || entry.status === 'queued'
                ? { ...entry, status: 'canceled', error: 'Canceled' }
                : entry
            ),
            isRunning: false,
            locked: false,
            paused: false,
          },
        }))
        batchAbortController = null
        return
      }

      if (item.status !== 'queued' && item.status !== 'failed') {
        continue
      }

      while (get().batch.paused && !batchAbortController.signal.aborted) {
        await new Promise((resolve) => setTimeout(resolve, 250))
      }

      set((state) => ({
        batch: {
          ...state.batch,
          items: state.batch.items.map((entry) =>
            entry.id === item.id
              ? { ...entry, status: 'running', error: undefined }
              : entry
          ),
        },
      }))

      try {
        const result = await runDetection(item.file, snapshot, batchAbortController.signal)
        set((state) => ({
          batch: {
            ...state.batch,
            items: state.batch.items.map((entry) =>
              entry.id === item.id
                ? { ...entry, status: 'done', result, error: undefined }
                : entry
              ),
            },
          }))
      } catch (error) {
        if (error instanceof APIError && error.isCanceled) {
          set((state) => ({
            batch: {
              ...state.batch,
              items: state.batch.items.map((entry) =>
                entry.id === item.id
                  ? { ...entry, status: 'canceled', error: 'Canceled' }
                  : entry
              ),
              isRunning: false,
              locked: false,
              paused: false,
            },
          }))
          batchAbortController = null
          return
        }
        set((state) => ({
          batch: {
            ...state.batch,
            items: state.batch.items.map((entry) =>
              entry.id === item.id
                ? {
                  ...entry,
                  status: 'failed',
                  error: error instanceof Error ? error.message : 'Detection failed',
                }
                : entry
            ),
          },
        }))
      }
    }

    set((state) => ({
      batch: {
        ...state.batch,
        isRunning: false,
        locked: false,
        paused: false,
      },
    }))
    batchAbortController = null
  },

  retryBatchFailed: async () => {
    const { batch } = get()
    if (batch.isRunning) return
    const hasFailed = batch.items.some((item) => item.status === 'failed')
    if (!hasFailed) return
    set((state) => ({
      batch: {
        ...state.batch,
        items: state.batch.items.map((item) =>
          item.status === 'failed' ? { ...item, status: 'queued', error: undefined } : item
        ),
      },
    }))
    await get().runBatchDetection()
  },

  cancelBatch: () => {
    if (batchAbortController) {
      batchAbortController.abort()
    }
    batchAbortController = null
    set((state) => ({
      batch: {
        ...state.batch,
        isRunning: false,
        locked: false,
        paused: false,
      },
    }))
  },

  toggleBatchPause: () => {
    set((state) => ({
      batch: {
        ...state.batch,
        paused: !state.batch.paused,
      },
    }))
  },

  openBatchResult: (id) => {
    const { batch } = get()
    const item = batch.items.find((entry) => entry.id === id)
    if (!item?.result) return
    set({
      result: item.result,
      resultRunId: Date.now(),
      reviewStatus: buildReviewMap(item.result),
      selectedObjectKey: null,
      currentView: 'results',
      batch: { ...batch, activeItemId: id },
    })
  },

  openNextBatchResult: () => {
    const { batch } = get()
    if (!batch.activeItemId) return
    const currentIndex = batch.items.findIndex((item) => item.id === batch.activeItemId)
    if (currentIndex === -1) return
    const nextDone = batch.items.slice(currentIndex + 1).find((item) => item.status === 'done' && item.result)
    const fallback = batch.items[currentIndex + 1]
    const target = nextDone ?? fallback
    if (!target) return
    if (target.result) {
      get().openBatchResult(target.id)
      return
    }
    set({
      result: null,
      resultRunId: Date.now(),
      reviewStatus: {},
      selectedObjectKey: null,
      currentView: 'results',
      batch: { ...batch, activeItemId: target.id },
    })
  },

  openPrevBatchResult: () => {
    const { batch } = get()
    if (!batch.activeItemId) return
    const currentIndex = batch.items.findIndex((item) => item.id === batch.activeItemId)
    if (currentIndex === -1) return
    const prevDone = [...batch.items.slice(0, currentIndex)].reverse()
      .find((item) => item.status === 'done' && item.result)
    const fallback = batch.items[currentIndex - 1]
    const target = prevDone ?? fallback
    if (!target) return
    if (target.result) {
      get().openBatchResult(target.id)
      return
    }
    set({
      result: null,
      resultRunId: Date.now(),
      reviewStatus: {},
      selectedObjectKey: null,
      currentView: 'results',
      batch: { ...batch, activeItemId: target.id },
    })
  },

  toggleTheme: () => {
    const next = !get().darkMode
    if (next) {
      document.documentElement.classList.add('dark')
      window.localStorage.setItem(THEME_KEY, 'dark')
    } else {
      document.documentElement.classList.remove('dark')
      window.localStorage.setItem(THEME_KEY, 'light')
    }
    set({ darkMode: next })
  },

  setReviewStatus: (key, status) => {
    set((state) => {
      const next = { ...state.reviewStatus }
      if (!status) {
        delete next[key]
      } else {
        next[key] = status
      }
      return { reviewStatus: next }
    })
  },

  setSelectedObjectKey: (key) => {
    set({ selectedObjectKey: key })
  },

  addObject: (obj) => {
    set((state) => {
      if (!state.result) return state
      const nextObjects = [...state.result.objects, obj]
      const nextReview = { ...state.reviewStatus }
      const status = obj.ReviewStatus
      const key = `${obj.CategoryID}-${obj.ObjectID}-${obj.Index}`
      if (status === 'accepted' || status === 'rejected') {
        nextReview[key] = status
      }
      const nextResult = {
        ...state.result,
        objects: nextObjects,
        count: nextObjects.length,
      }
      const nextBatch = state.batch.activeItemId
        ? {
          ...state.batch,
          items: state.batch.items.map((item) =>
            item.id === state.batch.activeItemId ? { ...item, result: nextResult } : item
          ),
        }
        : state.batch
      return {
        result: nextResult,
        reviewStatus: nextReview,
        batch: nextBatch,
      }
    })
  },

  updateObject: (updated) => {
    set((state) => {
      if (!state.result) return state
      const nextObjects = state.result.objects.map((obj) =>
        obj.Index === updated.Index ? { ...obj, ...updated } : obj
      )
      const nextReview = { ...state.reviewStatus }
      const status = updated.ReviewStatus
      const key = `${updated.CategoryID}-${updated.ObjectID}-${updated.Index}`
      if (status === 'accepted' || status === 'rejected') {
        nextReview[key] = status
      } else {
        delete nextReview[key]
      }
      const nextResult = {
        ...state.result,
        objects: nextObjects,
      }
      const nextBatch = state.batch.activeItemId
        ? {
          ...state.batch,
          items: state.batch.items.map((item) =>
            item.id === state.batch.activeItemId ? { ...item, result: nextResult } : item
          ),
        }
        : state.batch
      return {
        result: nextResult,
        reviewStatus: nextReview,
        batch: nextBatch,
      }
    })
  },

  removeObject: (index) => {
    set((state) => {
      if (!state.result) return state
      const target = state.result.objects.find((obj) => obj.Index === index)
      const nextObjects = state.result.objects.filter((obj) => obj.Index !== index)
      const nextReview = { ...state.reviewStatus }
      if (target) {
        delete nextReview[`${target.CategoryID}-${target.ObjectID}-${target.Index}`]
      }
      const nextResult = {
        ...state.result,
        objects: nextObjects,
        count: nextObjects.length,
      }
      const nextBatch = state.batch.activeItemId
        ? {
          ...state.batch,
          items: state.batch.items.map((item) =>
            item.id === state.batch.activeItemId ? { ...item, result: nextResult } : item
          ),
        }
        : state.batch
      return {
        result: nextResult,
        reviewStatus: nextReview,
        batch: nextBatch,
      }
    })
  },

  restoreObject: (obj, atIndex) => {
    set((state) => {
      if (!state.result) return state
      const nextObjects = [...state.result.objects]
      if (atIndex !== undefined && atIndex >= 0 && atIndex <= nextObjects.length) {
        nextObjects.splice(atIndex, 0, obj)
      } else {
        nextObjects.push(obj)
      }
      const nextReview = { ...state.reviewStatus }
      const status = obj.ReviewStatus
      const key = objectKey(obj)
      if (status === 'accepted' || status === 'rejected') {
        nextReview[key] = status
      }
      const nextResult = {
        ...state.result,
        objects: nextObjects,
        count: nextObjects.length,
      }
      const nextBatch = state.batch.activeItemId
        ? {
          ...state.batch,
          items: state.batch.items.map((item) =>
            item.id === state.batch.activeItemId ? { ...item, result: nextResult } : item
          ),
        }
        : state.batch
      return {
        result: nextResult,
        reviewStatus: nextReview,
        batch: nextBatch,
      }
    })
  },

  undoAction: () => {
    const action = useHistoryStore.getState().undo()
    if (!action) return

    const state = get()
    if (!state.result) return

    switch (action.type) {
      case 'review': {
        // Reverse: set to prev status
        const nextReview = { ...state.reviewStatus }
        if (!action.prev) {
          delete nextReview[action.key]
        } else {
          nextReview[action.key] = action.prev
        }
        set({ reviewStatus: nextReview })
        break
      }
      case 'update': {
        // Reverse: restore prev object
        const nextObjects = state.result.objects.map((obj) =>
          obj.Index === action.prev.Index ? action.prev : obj
        )
        const nextResult = { ...state.result, objects: nextObjects }
        const nextBatch = state.batch.activeItemId
          ? {
            ...state.batch,
            items: state.batch.items.map((item) =>
              item.id === state.batch.activeItemId ? { ...item, result: nextResult } : item
            ),
          }
          : state.batch
        set({ result: nextResult, batch: nextBatch })
        break
      }
      case 'delete': {
        // Reverse: restore deleted object
        get().restoreObject(action.object, action.index)
        break
      }
      case 'create': {
        // Reverse: remove created object
        const nextObjects = state.result.objects.filter(
          (obj) => obj.Index !== action.object.Index
        )
        const nextReview = { ...state.reviewStatus }
        delete nextReview[objectKey(action.object)]
        const nextResult = { ...state.result, objects: nextObjects, count: nextObjects.length }
        const nextBatch = state.batch.activeItemId
          ? {
            ...state.batch,
            items: state.batch.items.map((item) =>
              item.id === state.batch.activeItemId ? { ...item, result: nextResult } : item
            ),
          }
          : state.batch
        set({
          result: nextResult,
          reviewStatus: nextReview,
          batch: nextBatch,
        })
        break
      }
    }
  },

  redoAction: () => {
    const action = useHistoryStore.getState().redo()
    if (!action) return

    const state = get()
    if (!state.result) return

    switch (action.type) {
      case 'review': {
        // Redo: set to next status
        const nextReview = { ...state.reviewStatus }
        if (!action.next) {
          delete nextReview[action.key]
        } else {
          nextReview[action.key] = action.next
        }
        set({ reviewStatus: nextReview })
        break
      }
      case 'update': {
        // Redo: apply next object
        const nextObjects = state.result.objects.map((obj) =>
          obj.Index === action.next.Index ? action.next : obj
        )
        const nextResult = { ...state.result, objects: nextObjects }
        const nextBatch = state.batch.activeItemId
          ? {
            ...state.batch,
            items: state.batch.items.map((item) =>
              item.id === state.batch.activeItemId ? { ...item, result: nextResult } : item
            ),
          }
          : state.batch
        set({ result: nextResult, batch: nextBatch })
        break
      }
      case 'delete': {
        // Redo: remove object again
        const nextObjects = state.result.objects.filter(
          (obj) => obj.Index !== action.object.Index
        )
        const nextReview = { ...state.reviewStatus }
        delete nextReview[objectKey(action.object)]
        const nextResult = { ...state.result, objects: nextObjects, count: nextObjects.length }
        const nextBatch = state.batch.activeItemId
          ? {
            ...state.batch,
            items: state.batch.items.map((item) =>
              item.id === state.batch.activeItemId ? { ...item, result: nextResult } : item
            ),
          }
          : state.batch
        set({
          result: nextResult,
          reviewStatus: nextReview,
          batch: nextBatch,
        })
        break
      }
      case 'create': {
        // Redo: add object back
        get().restoreObject(action.object)
        break
      }
    }
  },
}))
