import { useEffect, useRef, useState } from 'react'
import { Minus, Plus, RefreshCw, Maximize2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

type ZoomControlsProps = {
  zoomPercent: number
  onZoomIn: () => void
  onZoomOut: () => void
  onReset: () => void
  onFit: () => void
  onZoomTo: (percent: number) => void
}

export function ZoomControls({ zoomPercent, onZoomIn, onZoomOut, onReset, onFit, onZoomTo }: ZoomControlsProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [value, setValue] = useState(`${zoomPercent}`)
  const inputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    if (!isEditing) {
      setValue(`${zoomPercent}`)
    }
  }, [zoomPercent, isEditing])

  useEffect(() => {
    if (isEditing) {
      inputRef.current?.focus()
      inputRef.current?.select()
    }
  }, [isEditing])

  const commit = () => {
    const nextValue = value.replace('%', '').trim()
    const parsed = Number(nextValue)
    if (!Number.isFinite(parsed)) {
      setIsEditing(false)
      setValue(`${zoomPercent}`)
      return
    }
    onZoomTo(parsed)
    setIsEditing(false)
  }

  return (
    <div className="group flex items-center gap-2 bg-[var(--bg-primary)]/95 backdrop-blur-md border border-[var(--border-muted)] rounded-xl px-2 py-1.5 scale-90 opacity-90 hover:scale-100 hover:opacity-100 hover:px-3 hover:py-2 shadow-lg shadow-black/40 hover:shadow-xl hover:shadow-black/50 transition-all duration-200 ease-out motion-reduce:transition-none motion-reduce:hover:scale-90 motion-reduce:hover:px-2 motion-reduce:hover:py-1.5">
      <Button variant="ghost" size="icon" className="h-7 w-7 group-hover:h-9 group-hover:w-9 transition-all duration-200" onClick={onZoomOut} aria-label="Zoom out">
        <Minus className="h-3.5 w-3.5 group-hover:h-4 group-hover:w-4 transition-all duration-200" />
      </Button>
      {isEditing ? (
        <div className="w-12 group-hover:w-14 transition-all duration-200">
          <Input
            ref={inputRef}
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onBlur={commit}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault()
                commit()
              }
              if (event.key === 'Escape') {
                event.preventDefault()
                setIsEditing(false)
                setValue(`${zoomPercent}`)
              }
            }}
            inputMode="numeric"
            className="h-6 px-2 text-[10px] group-hover:text-xs text-center font-semibold"
            aria-label="Zoom percent"
          />
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setIsEditing(true)}
          className="text-[10px] group-hover:text-xs font-bold w-10 group-hover:w-12 text-center cursor-pointer text-[var(--text-primary)] hover:text-[var(--accent)] transition-all duration-200"
          aria-label="Set zoom percent"
        >
          {zoomPercent}%
        </button>
      )}
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
