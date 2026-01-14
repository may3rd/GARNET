/**
 * Canvas state store - manages viewport and zoom
 */
import { create } from 'zustand'

interface CanvasState {
    // State
    zoom: number
    panX: number
    panY: number
    canvasWidth: number
    canvasHeight: number

    // Actions
    setZoom: (zoom: number) => void
    zoomIn: () => void
    zoomOut: () => void
    resetZoom: () => void
    fitToWindow: () => void
    setPan: (x: number, y: number) => void
    setCanvasSize: (width: number, height: number) => void
}

const MIN_ZOOM = 0.1
const MAX_ZOOM = 5
const ZOOM_STEP = 0.25

export const useCanvasStore = create<CanvasState>((set, get) => ({
    zoom: 1,
    panX: 0,
    panY: 0,
    canvasWidth: 0,
    canvasHeight: 0,

    setZoom: (zoom) =>
        set({ zoom: Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom)) }),

    zoomIn: () =>
        set((state) => ({
            zoom: Math.min(MAX_ZOOM, state.zoom + ZOOM_STEP),
        })),

    zoomOut: () =>
        set((state) => ({
            zoom: Math.max(MIN_ZOOM, state.zoom - ZOOM_STEP),
        })),

    resetZoom: () => set({ zoom: 1, panX: 0, panY: 0 }),

    fitToWindow: () => {
        // Will be implemented when canvas is integrated
        set({ zoom: 1, panX: 0, panY: 0 })
    },

    setPan: (x, y) => set({ panX: x, panY: y }),

    setCanvasSize: (width, height) =>
        set({ canvasWidth: width, canvasHeight: height }),
}))
