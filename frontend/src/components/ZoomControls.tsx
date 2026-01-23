import { Minus, Plus, RefreshCw, Maximize2 } from 'lucide-react'
import { cn } from '@/lib/utils'

type ZoomControlsProps = {
  zoomPercent: number
  onZoomIn: () => void
  onZoomOut: () => void
  onReset: () => void
  onFit: () => void
}

export function ZoomControls({ zoomPercent, onZoomIn, onZoomOut, onReset, onFit }: ZoomControlsProps) {
  const buttonClass = cn(
    'h-9 w-9 rounded-lg border border-[var(--border-muted)]',
    'flex items-center justify-center',
    'bg-[var(--bg-secondary)] text-[var(--text-primary)]',
    'hover:border-[var(--accent)] transition-colors'
  )

  return (
    <div className="flex items-center gap-2 bg-[var(--bg-secondary)] border border-[var(--border-muted)] rounded-xl px-3 py-2">
      <button className={buttonClass} onClick={onZoomOut} aria-label="Zoom out">
        <Minus className="h-4 w-4" />
      </button>
      <div className="text-xs font-semibold w-12 text-center">{zoomPercent}%</div>
      <button className={buttonClass} onClick={onZoomIn} aria-label="Zoom in">
        <Plus className="h-4 w-4" />
      </button>
      <div className="h-6 w-px bg-[var(--border-muted)]" />
      <button className={buttonClass} onClick={onFit} aria-label="Fit to screen">
        <Maximize2 className="h-4 w-4" />
      </button>
      <button className={buttonClass} onClick={onReset} aria-label="Reset zoom">
        <RefreshCw className="h-4 w-4" />
      </button>
    </div>
  )
}
