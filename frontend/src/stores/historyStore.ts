import { create } from 'zustand'
import type { DetectedObject } from '@/types'

const MAX_HISTORY = 50

export type HistoryAction =
    | { type: 'review'; key: string; prev: 'accepted' | 'rejected' | null; next: 'accepted' | 'rejected' | null }
    | { type: 'update'; prev: DetectedObject; next: DetectedObject }
    | { type: 'delete'; object: DetectedObject; index: number }
    | { type: 'create'; object: DetectedObject }

export type HistoryState = {
    past: HistoryAction[]
    future: HistoryAction[]
}

export type HistoryActions = {
    pushAction: (action: HistoryAction) => void
    undo: () => HistoryAction | null
    redo: () => HistoryAction | null
    clear: () => void
    canUndo: () => boolean
    canRedo: () => boolean
}

export const useHistoryStore = create<HistoryState & HistoryActions>((set, get) => ({
    past: [],
    future: [],

    pushAction: (action) => {
        set((state) => {
            const newPast = [...state.past, action]
            // Limit history size
            if (newPast.length > MAX_HISTORY) {
                newPast.shift()
            }
            return {
                past: newPast,
                future: [], // Clear redo stack on new action
            }
        })
    },

    undo: () => {
        const { past, future } = get()
        if (past.length === 0) return null

        const action = past[past.length - 1]
        set({
            past: past.slice(0, -1),
            future: [...future, action],
        })
        return action
    },

    redo: () => {
        const { past, future } = get()
        if (future.length === 0) return null

        const action = future[future.length - 1]
        set({
            past: [...past, action],
            future: future.slice(0, -1),
        })
        return action
    },

    clear: () => {
        set({ past: [], future: [] })
    },

    canUndo: () => get().past.length > 0,
    canRedo: () => get().future.length > 0,
}))
