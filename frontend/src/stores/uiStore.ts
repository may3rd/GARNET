/**
 * UI state store - manages sidebar, theme, and filters
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { DEFAULT_CONFIDENCE } from '@/lib/constants'

type Theme = 'light' | 'dark' | 'system'
type AppView = 'upload' | 'preview' | 'processing' | 'results'

interface UIState {
    // State
    theme: Theme
    sidebarCollapsed: boolean
    confidenceFilter: number
    searchQuery: string
    currentView: AppView
    hiddenCategories: Set<number>

    // Actions
    setTheme: (theme: Theme) => void
    toggleSidebar: () => void
    setSidebarCollapsed: (collapsed: boolean) => void
    setConfidenceFilter: (value: number) => void
    setSearchQuery: (query: string) => void
    setCurrentView: (view: AppView) => void
    toggleCategoryVisibility: (categoryId: number) => void
    showAllCategories: () => void
    hideAllCategories: (categoryIds: number[]) => void
}

export const useUIStore = create<UIState>()(
    persist(
        (set) => ({
            theme: 'system',
            sidebarCollapsed: false,
            confidenceFilter: DEFAULT_CONFIDENCE,
            searchQuery: '',
            currentView: 'upload',
            hiddenCategories: new Set(),

            setTheme: (theme) => {
                set({ theme })
                // Apply theme to document
                const root = document.documentElement
                if (theme === 'system') {
                    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
                    root.classList.toggle('dark', isDark)
                } else {
                    root.classList.toggle('dark', theme === 'dark')
                }
            },

            toggleSidebar: () =>
                set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

            setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),

            setConfidenceFilter: (value) => set({ confidenceFilter: value }),

            setSearchQuery: (query) => set({ searchQuery: query }),

            setCurrentView: (view) => set({ currentView: view }),

            toggleCategoryVisibility: (categoryId) =>
                set((state) => {
                    const newHidden = new Set(state.hiddenCategories)
                    if (newHidden.has(categoryId)) {
                        newHidden.delete(categoryId)
                    } else {
                        newHidden.add(categoryId)
                    }
                    return { hiddenCategories: newHidden }
                }),

            showAllCategories: () => set({ hiddenCategories: new Set() }),

            hideAllCategories: (categoryIds) =>
                set({ hiddenCategories: new Set(categoryIds) }),
        }),
        {
            name: 'garnet-ui-settings',
            partialize: (state) => ({
                theme: state.theme,
                sidebarCollapsed: state.sidebarCollapsed,
                confidenceFilter: state.confidenceFilter,
            }),
        }
    )
)
