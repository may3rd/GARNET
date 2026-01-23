import { ArrowLeft, Moon, Sun } from 'lucide-react'
import { useAppStore } from '@/stores/appStore'
import { cn } from '@/lib/utils'

export function Header() {
  const currentView = useAppStore((state) => state.currentView)
  const imageFile = useAppStore((state) => state.imageFile)
  const toggleTheme = useAppStore((state) => state.toggleTheme)
  const darkMode = useAppStore((state) => state.darkMode)
  const setImageFile = useAppStore((state) => state.setImageFile)

  return (
    <header className={cn(
      'flex items-center justify-between',
      'h-14 px-4 md:px-6',
      'border-b border-[var(--border-muted)]',
      'bg-[var(--bg-secondary)]'
    )}>
      <div className="flex items-center gap-3">
        {currentView !== 'empty' && (
          <button
            onClick={() => setImageFile(null)}
            className={cn(
              'h-9 w-9 rounded-lg',
              'flex items-center justify-center',
              'text-[var(--text-secondary)] hover:text-[var(--text-primary)]',
              'hover:bg-[var(--bg-primary)] transition-colors'
            )}
            aria-label="Back"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
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
        {imageFile && (
          <div className="hidden md:block text-xs text-[var(--text-secondary)] ml-4">
            {imageFile.name}
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={toggleTheme}
          className={cn(
            'h-9 w-9 rounded-lg',
            'flex items-center justify-center',
            'text-[var(--text-secondary)] hover:text-[var(--text-primary)]',
            'hover:bg-[var(--bg-primary)] transition-colors'
          )}
          aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
        </button>
      </div>
    </header>
  )
}
