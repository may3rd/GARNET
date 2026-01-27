import { useCallback, useRef, useState } from 'react'
import { FileImage, Sparkles, Upload } from 'lucide-react'
import { useAppStore } from '@/stores/appStore'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export function UploadZone() {
  const inputRef = useRef<HTMLInputElement>(null)
  const setImageFile = useAppStore((state) => state.setImageFile)
  const setImageMeta = useAppStore((state) => state.setImageMeta)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const handleFile = useCallback((file: File) => {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp']
    if (file.type === 'application/pdf') {
      setUploadError('PDF uploads are disabled until page-count validation is available.')
      return
    }
    if (!validTypes.includes(file.type)) {
      setUploadError('Unsupported file type. Upload a JPG, PNG, or WEBP image.')
      return
    }
    setUploadError(null)

    const img = new Image()
    img.onload = () => {
      setImageMeta({ width: img.width, height: img.height })
      URL.revokeObjectURL(img.src)
    }
    img.src = URL.createObjectURL(file)

    setImageFile(file)
  }, [setImageFile, setImageMeta])

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
    const file = event.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const handleDragEnter = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
  }, [])

  const handleChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) handleFile(file)
  }, [handleFile])

  const handleSample = useCallback(async () => {
    const response = await fetch('/static/images/prediction_results.png')
    if (!response.ok) return
    const blob = await response.blob()
    handleFile(new File([blob], 'sample_pid.png', { type: blob.type }))
  }, [handleFile])

  return (
    <div className="flex flex-col items-center justify-center h-full px-6 py-10">
      <div
        onClick={() => inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(event) => event.preventDefault()}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            inputRef.current?.click()
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload P&ID image"
        className={cn(
          'w-full max-w-3xl min-h-[320px]',
          'border-2 border-dashed',
          isDragging ? 'border-[var(--accent)] bg-[var(--accent)]/5' : 'border-[var(--border-muted)]',
          'rounded-2xl bg-[var(--bg-secondary)]',
          'flex items-center justify-center relative',
          'cursor-pointer transition-all',
          'hover:border-[var(--accent)]',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] focus-visible:ring-offset-2'
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp,.pdf"
          onChange={handleChange}
          className="hidden"
        />
        <div className="flex flex-col items-center gap-4 text-center px-6">
          <div className="h-14 w-14 rounded-full bg-[var(--bg-primary)] flex items-center justify-center">
            <FileImage className="h-7 w-7 text-[var(--accent)]" />
          </div>
          <div>
            <div className="text-lg font-semibold">Drop P&amp;ID image here</div>
            <div className="text-sm text-[var(--text-secondary)] mt-1">or click to browse</div>
          </div>
          <div className="text-xs text-[var(--text-secondary)]">Supports: JPG, PNG, WEBP</div>
          <div className="text-[11px] text-[var(--text-secondary)]">PDF uploads are temporarily disabled.</div>
        </div>
        <Upload className="absolute right-6 bottom-6 h-5 w-5 text-[var(--text-secondary)]" />
      </div>

      {uploadError && (
        <div className="mt-4 text-xs text-[var(--danger)] bg-[var(--bg-secondary)] border border-[var(--border-muted)] px-3 py-2 rounded-lg">
          {uploadError}
        </div>
      )}

      <div className="flex items-center gap-3 text-xs text-[var(--text-secondary)] mt-6">
        <span className="h-px w-14 bg-[var(--border-muted)]" />
        or
        <span className="h-px w-14 bg-[var(--border-muted)]" />
      </div>

      <Button onClick={handleSample} variant="default" className="mt-5">
        <Sparkles className="h-4 w-4" />
        Try with Sample P&amp;ID
      </Button>
    </div>
  )
}
