import { create } from 'zustand'
import type { AppView, DetectedObject, DetectionResult } from '@/types'
import { runDetection, type DetectionOptions } from '@/lib/api'
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
  isProcessing: boolean
  progress: { step: string; percent: number } | null
  error: string | null
  darkMode: boolean
}

export type AppActions = {
  setImageFile: (file: File | null) => void
  setImageMeta: (meta: { width: number; height: number } | null) => void
  setOptions: (options: Partial<DetectionOptions>) => void
  setView: (view: AppView) => void
  runDetection: () => Promise<void>
  cancelDetection: () => void
  toggleTheme: () => void
  setReviewStatus: (key: string, status: 'accepted' | 'rejected' | null) => void
  setSelectedObjectKey: (key: string | null) => void
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
  isProcessing: false,
  progress: null,
  error: null,
  darkMode: initialDarkMode,

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
      })
      return
    }

    const url = URL.createObjectURL(file)
    set({ imageFile: file, imageUrl: url, currentView: 'preview' })
  },

  setImageMeta: (meta) => set({ imageMeta: meta }),

  setOptions: (options) =>
    set((state) => ({ options: { ...state.options, ...options } })),

  setView: (view) => set({ currentView: view }),

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
      const reviewFromPayload: Record<string, 'accepted' | 'rejected'> = {}
      result.objects.forEach((obj) => {
        const status = obj.ReviewStatus
        if (status === 'accepted' || status === 'rejected') {
          reviewFromPayload[`${obj.CategoryID}-${obj.ObjectID}-${obj.Index}`] = status
        }
      })
      set({
        result,
        resultRunId: Date.now(),
        reviewStatus: reviewFromPayload,
        selectedObjectKey: null,
        isProcessing: false,
        currentView: 'results',
        progress: { step: 'Complete', percent: 100 },
      })
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
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
      return {
        result: {
          ...state.result,
          objects: nextObjects,
          count: nextObjects.length,
        },
        reviewStatus: nextReview,
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
      return {
        result: {
          ...state.result,
          objects: nextObjects,
        },
        reviewStatus: nextReview,
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
      return {
        result: {
          ...state.result,
          objects: nextObjects,
          count: nextObjects.length,
        },
        reviewStatus: nextReview,
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
      return {
        result: {
          ...state.result,
          objects: nextObjects,
          count: nextObjects.length,
        },
        reviewStatus: nextReview,
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
        set({ result: { ...state.result, objects: nextObjects } })
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
        set({
          result: { ...state.result, objects: nextObjects, count: nextObjects.length },
          reviewStatus: nextReview,
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
        set({ result: { ...state.result, objects: nextObjects } })
        break
      }
      case 'delete': {
        // Redo: remove object again
        const nextObjects = state.result.objects.filter(
          (obj) => obj.Index !== action.object.Index
        )
        const nextReview = { ...state.reviewStatus }
        delete nextReview[objectKey(action.object)]
        set({
          result: { ...state.result, objects: nextObjects, count: nextObjects.length },
          reviewStatus: nextReview,
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
