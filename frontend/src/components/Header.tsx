import { ArrowLeft, Moon, Redo2, Sun, Undo2 } from 'lucide-react'
import { useMemo } from 'react'
import { useAppStore } from '@/stores/appStore'
import { useHistoryStore } from '@/stores/historyStore'
import { Button } from '@/components/ui/button'
import { objectKey } from '@/lib/objectKey'

export function Header() {
  const currentView = useAppStore((state) => state.currentView)
  const imageFile = useAppStore((state) => state.imageFile)
  const batch = useAppStore((state) => state.batch)
  const toggleTheme = useAppStore((state) => state.toggleTheme)
  const darkMode = useAppStore((state) => state.darkMode)
  const goBack = useAppStore((state) => state.goBack)
  const result = useAppStore((state) => state.result)
  const reviewStatus = useAppStore((state) => state.reviewStatus)
  const confidenceFilter = useAppStore((state) => state.confidenceFilter)
  const undoAction = useAppStore((state) => state.undoAction)
  const redoAction = useAppStore((state) => state.redoAction)
  const canUndo = useHistoryStore((state) => state.canUndo())
  const canRedo = useHistoryStore((state) => state.canRedo())

  // Filter objects by confidence threshold (same as sidebar)
  const visibleObjects = useMemo(() => {
    if (!result) return []
    return result.objects.filter((obj) => obj.Score >= confidenceFilter)
  }, [result, confidenceFilter])

  const reviewedCount = visibleObjects.filter((obj) => reviewStatus[objectKey(obj)]).length
  const activeBatchItem = batch.items.find((item) => item.id === batch.activeItemId)
  const displayName = imageFile?.name ?? activeBatchItem?.fileName ?? null

  return (
    <header className="flex items-center justify-between h-14 px-4 md:px-6 border-b border-[var(--border-muted)] bg-[var(--bg-secondary)]">
      <div className="flex items-center gap-3">
        {currentView !== 'empty' && (
          <Button
            variant="ghost"
            size="icon"
            onClick={goBack}
            aria-label="Back"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
        )}
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-[var(--accent)] text-white font-semibold flex items-center justify-center">
            G
          </div>
          <div>
            <div className="text-sm font-semibold tracking-wide">GARNET</div>
            <div className="text-[11px] text-[var(--text-secondary)]">P&amp;ID Detection</div>
          </div>
        </div>
        {displayName && (
          <div className="hidden md:block text-xs text-[var(--text-secondary)] ml-4">
            {displayName}
          </div>
        )}
      </div>

      {currentView === 'results' && result && (
        <div className="hidden md:flex items-center gap-4 text-sm">
          <span className="text-[var(--text-primary)] font-semibold">{visibleObjects.length} objects</span>
          <span className="text-[var(--text-secondary)]">Review: {reviewedCount}/{visibleObjects.length}</span>
        </div>
      )}

      <div className="flex items-center gap-1">
        {currentView === 'results' && (
          <>
            <Button
              variant="ghost"
              size="icon"
              onClick={undoAction}
              disabled={!canUndo}
              aria-label="Undo (Ctrl+Z)"
              title="Undo (Ctrl+Z)"
            >
              <Undo2 className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={redoAction}
              disabled={!canRedo}
              aria-label="Redo (Ctrl+Shift+Z)"
              title="Redo (Ctrl+Shift+Z)"
            >
              <Redo2 className="h-4 w-4" />
            </Button>
            <div className="w-px h-6 bg-[var(--border-muted)] mx-1" />
          </>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
        </Button>
      </div>
    </header>
  )
}
