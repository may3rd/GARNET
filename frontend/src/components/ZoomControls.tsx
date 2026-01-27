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
    <div className="group flex items-center gap-2 bg-[var(--bg-secondary)]/80 backdrop-blur border border-[var(--border-muted)] rounded-xl px-2 py-1.5 scale-90 opacity-60 hover:scale-100 hover:opacity-100 hover:px-3 hover:py-2 transition-all duration-200 ease-out motion-reduce:transition-none motion-reduce:hover:scale-90 motion-reduce:hover:px-2 motion-reduce:hover:py-1.5">
      <Button variant="ghost" size="icon" className="h-7 w-7 group-hover:h-9 group-hover:w-9 transition-all duration-200" onClick={onZoomOut} aria-label="Zoom out">
        <Minus className="h-3.5 w-3.5 group-hover:h-4 group-hover:w-4 transition-all duration-200" />
      </Button>
      <div className="text-[10px] group-hover:text-xs font-semibold w-10 group-hover:w-12 text-center transition-all duration-200">{zoomPercent}%</div>
      <Button variant="ghost" size="icon" className="h-7 w-7 group-hover:h-9 group-hover:w-9 transition-all duration-200" onClick={onZoomIn} aria-label="Zoom in">
        <Plus className="h-3.5 w-3.5 group-hover:h-4 group-hover:w-4 transition-all duration-200" />
      </Button>
      <div className="h-4 group-hover:h-6 w-px bg-[var(--border-muted)] transition-all duration-200" />
      <Button variant="ghost" size="icon" className="h-7 w-7 group-hover:h-9 group-hover:w-9 transition-all duration-200" onClick={onFit} aria-label="Fit to screen">
        <Maximize2 className="h-3.5 w-3.5 group-hover:h-4 group-hover:w-4 transition-all duration-200" />
      </Button>
      <Button variant="ghost" size="icon" className="h-7 w-7 group-hover:h-9 group-hover:w-9 transition-all duration-200" onClick={onReset} aria-label="Reset zoom">
        <RefreshCw className="h-3.5 w-3.5 group-hover:h-4 group-hover:w-4 transition-all duration-200" />
      </Button>
    </div>
  )
}
