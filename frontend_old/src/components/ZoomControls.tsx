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
    <div className="group flex items-center gap-2 rounded-xl border border-slate-300/80 bg-white/95 px-2 py-1.5 text-slate-700 opacity-95 shadow-lg shadow-black/30 backdrop-blur-md transition-all duration-200 ease-out hover:scale-100 hover:px-3 hover:py-2 hover:opacity-100 hover:shadow-xl hover:shadow-black/40 motion-reduce:transition-none motion-reduce:hover:px-2 motion-reduce:hover:py-1.5">
      <Button variant="ghost" size="icon" className="h-7 w-7 text-slate-600 transition-all duration-200 hover:bg-slate-100 hover:text-slate-950 group-hover:h-9 group-hover:w-9" onClick={onZoomOut} aria-label="Zoom out">
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
            className="h-6 border-slate-300 bg-white px-2 text-center text-[10px] font-semibold text-slate-950 placeholder:text-slate-500 group-hover:text-xs"
            aria-label="Zoom percent"
          />
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setIsEditing(true)}
          className="w-10 cursor-pointer text-center text-[10px] font-bold text-slate-950 transition-all duration-200 hover:text-blue-700 group-hover:w-12 group-hover:text-xs"
          aria-label="Set zoom percent"
        >
          {zoomPercent}%
        </button>
      )}
      <Button variant="ghost" size="icon" className="h-7 w-7 text-slate-600 transition-all duration-200 hover:bg-slate-100 hover:text-slate-950 group-hover:h-9 group-hover:w-9" onClick={onZoomIn} aria-label="Zoom in">
        <Plus className="h-3.5 w-3.5 group-hover:h-4 group-hover:w-4 transition-all duration-200" />
      </Button>
      <div className="h-4 w-px bg-slate-300 transition-all duration-200 group-hover:h-6" />
      <Button variant="ghost" size="icon" className="h-7 w-7 text-slate-600 transition-all duration-200 hover:bg-slate-100 hover:text-slate-950 group-hover:h-9 group-hover:w-9" onClick={onFit} aria-label="Fit to screen">
        <Maximize2 className="h-3.5 w-3.5 group-hover:h-4 group-hover:w-4 transition-all duration-200" />
      </Button>
      <Button variant="ghost" size="icon" className="h-7 w-7 text-slate-600 transition-all duration-200 hover:bg-slate-100 hover:text-slate-950 group-hover:h-9 group-hover:w-9" onClick={onReset} aria-label="Reset zoom">
        <RefreshCw className="h-3.5 w-3.5 group-hover:h-4 group-hover:w-4 transition-all duration-200" />
      </Button>
    </div>
  )
}
