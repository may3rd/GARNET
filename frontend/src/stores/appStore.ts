import { create } from 'zustand'
import type { AppView, DetectionResult } from '@/types'
import { runDetection, type DetectionOptions } from '@/lib/api'

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
  toggleTheme: () => void
  setReviewStatus: (key: string, status: 'accepted' | 'rejected' | null) => void
  setSelectedObjectKey: (key: string | null) => void
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

    set({
      isProcessing: true,
      currentView: 'processing',
      error: null,
      progress: { step: 'Uploading image...', percent: 8 },
    })

    let percent = 8
    progressTimer = window.setInterval(() => {
      percent = Math.min(percent + 5, 92)
      const step = percent < 40
        ? 'Running detection...'
        : percent < 75
          ? 'Extracting objects...'
          : 'Finalizing results...'
      set({ progress: { step, percent } })
    }, 800)

    try {
      const result = await runDetection(imageFile, options)
      set({
        result,
        resultRunId: Date.now(),
        reviewStatus: {},
        selectedObjectKey: null,
        isProcessing: false,
        currentView: 'results',
        progress: { step: 'Complete', percent: 100 },
      })
    } catch (error) {
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
}))
