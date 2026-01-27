import { Minus, Plus, RefreshCw, Maximize2 } from 'lucide-react'
import { Button } from '@/components/ui/button'

type ZoomControlsProps = {
  zoomPercent: number
  onZoomIn: () => void
  onZoomOut: () => void
  onReset: () => void
  onFit: () => void
}

export function ZoomControls({ zoomPercent, onZoomIn, onZoomOut, onReset, onFit }: ZoomControlsProps) {
  return (
    <div className="flex items-center gap-2 bg-[var(--bg-secondary)] border border-[var(--border-muted)] rounded-xl px-3 py-2">
      <Button variant="ghost" size="icon" onClick={onZoomOut} aria-label="Zoom out">
        <Minus className="h-4 w-4" />
      </Button>
      <div className="text-xs font-semibold w-12 text-center">{zoomPercent}%</div>
      <Button variant="ghost" size="icon" onClick={onZoomIn} aria-label="Zoom in">
        <Plus className="h-4 w-4" />
      </Button>
      <div className="h-6 w-px bg-[var(--border-muted)]" />
      <Button variant="ghost" size="icon" onClick={onFit} aria-label="Fit to screen">
        <Maximize2 className="h-4 w-4" />
      </Button>
      <Button variant="ghost" size="icon" onClick={onReset} aria-label="Reset zoom">
        <RefreshCw className="h-4 w-4" />
      </Button>
    </div>
  )
}
