/**
 * Detection state store - manages detected objects and image state
 */
import { create } from 'zustand'
import type { DetectedObject } from '@/types/detection'

interface ProgressState {
    step: string
    percent: number
    objectCount: number
}

interface DetectionState {
    // State
    objects: DetectedObject[]
    selectedIds: Set<number>
    imageUrl: string | null
    imageName: string | null
    isLoading: boolean
    progress: ProgressState | null

    // Actions
    setObjects: (objects: DetectedObject[]) => void
    selectObject: (id: number | null) => void
    toggleSelection: (id: number) => void
    selectAll: () => void
    clearSelection: () => void
    updateObjectStatus: (id: number, status: 'accepted' | 'rejected' | 'pending') => void
    updateSelectedStatus: (status: 'accepted' | 'rejected') => void
    setImage: (url: string, name: string) => void
    setLoading: (loading: boolean) => void
    setProgress: (progress: ProgressState | null) => void
    reset: () => void
}

const initialState = {
    objects: [],
    selectedIds: new Set<number>(),
    imageUrl: null,
    imageName: null,
    isLoading: false,
    progress: null,
}

export const useDetectionStore = create<DetectionState>((set, get) => ({
    ...initialState,

    setObjects: (objects) => set({ objects }),

    selectObject: (id) =>
        set({ selectedIds: id !== null ? new Set([id]) : new Set() }),

    toggleSelection: (id) =>
        set((state) => {
            const newIds = new Set(state.selectedIds)
            if (newIds.has(id)) {
                newIds.delete(id)
            } else {
                newIds.add(id)
            }
            return { selectedIds: newIds }
        }),

    selectAll: () =>
        set((state) => ({
            selectedIds: new Set(state.objects.map((o) => o.id)),
        })),

    clearSelection: () => set({ selectedIds: new Set() }),

    updateObjectStatus: (id, status) =>
        set((state) => ({
            objects: state.objects.map((obj) =>
                obj.id === id ? { ...obj, status } : obj
            ),
        })),

    updateSelectedStatus: (status) =>
        set((state) => ({
            objects: state.objects.map((obj) =>
                state.selectedIds.has(obj.id) ? { ...obj, status } : obj
            ),
        })),

    setImage: (url, name) => set({ imageUrl: url, imageName: name }),

    setLoading: (loading) => set({ isLoading: loading }),

    setProgress: (progress) => set({ progress }),

    reset: () => set(initialState),
}))
