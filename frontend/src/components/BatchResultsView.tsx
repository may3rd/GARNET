import { useEffect, useMemo, useRef, useState } from 'react'
import { useAppStore } from '@/stores/appStore'
import { buildYoloClasses, exportCocoBatch, exportYolo } from '@/lib/exportFormats'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { cn } from '@/lib/utils'
import { ChevronLeft, ChevronRight, Download, Eye, Pause, Play, Plus, RefreshCw, RotateCcw, Trash2, XCircle } from 'lucide-react'

type FilterMode = 'all' | 'accepted' | 'rejected'

export function BatchResultsView() {
  const batch = useAppStore((state) => state.batch)
  const runBatchDetection = useAppStore((state) => state.runBatchDetection)
  const cancelBatch = useAppStore((state) => state.cancelBatch)
  const retryBatchFailed = useAppStore((state) => state.retryBatchFailed)
  const openBatchResult = useAppStore((state) => state.openBatchResult)
  const addBatchFiles = useAppStore((state) => state.addBatchFiles)
  const removeBatchItem = useAppStore((state) => state.removeBatchItem)
  const resetBatchItem = useAppStore((state) => state.resetBatchItem)
  const toggleBatchPause = useAppStore((state) => state.toggleBatchPause)

  const [filter, setFilter] = useState<FilterMode>('all')
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [previewItemId, setPreviewItemId] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  const previewCandidates = useMemo(
    () => batch.items.filter((item) => item.status !== 'done'),
    [batch.items]
  )
  const previewIndex = previewItemId
    ? previewCandidates.findIndex((item) => item.id === previewItemId)
    : -1
  const previewItem = previewIndex >= 0 ? previewCandidates[previewIndex] : null

  useEffect(() => {
    if (!previewItemId) {
      setPreviewUrl(null)
      return
    }
    const item = batch.items.find((entry) => entry.id === previewItemId)
    if (!item) {
      setPreviewItemId(null)
      return
    }
    const url = URL.createObjectURL(item.file)
    setPreviewUrl(url)
    return () => {
      URL.revokeObjectURL(url)
    }
  }, [batch.items, previewItemId])

  const doneItems = useMemo(
    () => batch.items.filter((item) => item.status === 'done' && item.result),
    [batch.items]
  )
  const hasRunnable = batch.items.some((item) => item.status === 'queued' || item.status === 'failed')

  const totalCount = batch.items.length
  const doneCount = doneItems.length
  const failedCount = batch.items.filter((item) => item.status === 'failed').length
  const runningCount = batch.items.filter((item) => item.status === 'running').length

  const filterObjects = (objects: NonNullable<typeof doneItems[number]['result']>['objects']) => {
    if (filter === 'accepted') return objects.filter((obj) => obj.ReviewStatus === 'accepted')
    if (filter === 'rejected') return objects.filter((obj) => obj.ReviewStatus === 'rejected')
    return objects
  }

  const download = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    link.click()
    URL.revokeObjectURL(url)
  }

  const handleExportYolo = () => {
    if (!doneItems.length) return
    const allObjects = doneItems.flatMap((item) => filterObjects(item.result!.objects))
    const { classNames, classIdMap } = buildYoloClasses(allObjects)
    download(new Blob([classNames.join('\n') + '\n'], { type: 'text/plain' }), 'batch.classes.txt')
    doneItems.forEach((item) => {
      const result = item.result!
      const objects = filterObjects(result.objects)
      const txt = exportYolo(objects, result.image_width, result.image_height, classIdMap)
      const baseName = item.fileName.replace(/\.[^.]+$/, '')
      download(new Blob([txt], { type: 'text/plain' }), `${baseName}.txt`)
    })
  }

  const handleExportCoco = () => {
    if (!doneItems.length) return
    const images = doneItems.map((item) => ({
      fileName: item.fileName,
      width: item.result!.image_width,
      height: item.result!.image_height,
      objects: filterObjects(item.result!.objects),
    }))
    const coco = exportCocoBatch(images)
    download(new Blob([JSON.stringify(coco, null, 2)], { type: 'application/json' }), 'garnet-batch.coco.json')
  }

  const statusBadge = (status: string) => {
    const base = 'text-xs px-2 py-0.5 rounded-full border'
    if (status === 'done') return cn(base, 'border-emerald-400/40 text-emerald-500 bg-emerald-400/10')
    if (status === 'running') return cn(base, 'border-blue-400/40 text-blue-500 bg-blue-400/10')
    if (status === 'failed') return cn(base, 'border-red-400/40 text-red-500 bg-red-400/10')
    if (status === 'canceled') return cn(base, 'border-amber-400/40 text-amber-500 bg-amber-400/10')
    return cn(base, 'border-[var(--border-muted)] text-[var(--text-secondary)]')
  }

  const reviewCounts = (result: NonNullable<typeof doneItems[number]['result']>) => {
    const accepted = result.objects.filter((obj) => obj.ReviewStatus === 'accepted').length
    const rejected = result.objects.filter((obj) => obj.ReviewStatus === 'rejected').length
    const all = result.objects.length
    return { accepted, rejected, all }
  }

  const handleAddClick = () => {
    fileInputRef.current?.click()
  }

  const handleAddChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (files && files.length > 0) {
      addBatchFiles(Array.from(files))
    }
    event.target.value = ''
  }

  const canPause = batch.isRunning

  const goPrev = () => {
    if (previewIndex <= 0) return
    setPreviewItemId(previewCandidates[previewIndex - 1].id)
  }

  const goNext = () => {
    if (previewIndex < 0 || previewIndex >= previewCandidates.length - 1) return
    setPreviewItemId(previewCandidates[previewIndex + 1].id)
  }

  const handleRemovePreview = () => {
    if (!previewItemId) return
    removeBatchItem(previewItemId)
    setPreviewItemId(null)
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      <Dialog open={Boolean(previewItemId)} onOpenChange={(open) => {
        if (!open) setPreviewItemId(null)
      }}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Image preview</DialogTitle>
          </DialogHeader>
          <div className="absolute right-12 top-4">
            <Button size="sm" variant="outline" onClick={handleRemovePreview} disabled={!previewItemId}>
              <Trash2 className="h-4 w-4" />
              Remove
            </Button>
          </div>
          {previewUrl ? (
            <div className="relative flex items-center justify-center max-h-[70vh] overflow-auto">
              <button
                type="button"
                onClick={goPrev}
                disabled={previewIndex <= 0}
                className={cn(
                  'absolute left-2 top-1/2 -translate-y-1/2',
                  'h-10 w-10 rounded-full border border-[var(--border-muted)] bg-[var(--bg-primary)]/90',
                  'flex items-center justify-center shadow-md',
                  'hover:bg-[var(--bg-secondary)]',
                  previewIndex <= 0 && 'opacity-40 cursor-not-allowed'
                )}
                aria-label="Previous image"
              >
                <ChevronLeft className="h-5 w-5" />
              </button>
              <img
                src={previewUrl}
                alt="Batch preview"
                className="max-w-full h-auto rounded-lg border border-[var(--border-muted)]"
              />
              <button
                type="button"
                onClick={goNext}
                disabled={previewIndex < 0 || previewIndex >= previewCandidates.length - 1}
                className={cn(
                  'absolute right-2 top-1/2 -translate-y-1/2',
                  'h-10 w-10 rounded-full border border-[var(--border-muted)] bg-[var(--bg-primary)]/90',
                  'flex items-center justify-center shadow-md',
                  'hover:bg-[var(--bg-secondary)]',
                  (previewIndex < 0 || previewIndex >= previewCandidates.length - 1) && 'opacity-40 cursor-not-allowed'
                )}
                aria-label="Next image"
              >
                <ChevronRight className="h-5 w-5" />
              </button>
            </div>
          ) : (
            <div className="text-sm text-[var(--text-secondary)]">Loading preview...</div>
          )}
          {previewItem && (
            <div className="text-xs text-[var(--text-secondary)]">
              {previewItem.fileName} · {previewIndex + 1} / {previewCandidates.length}
            </div>
          )}
        </DialogContent>
      </Dialog>
      <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border-muted)]">
        <div>
          <div className="text-sm font-semibold">Batch Processing</div>
          <div className="text-xs text-[var(--text-secondary)]">
            {doneCount}/{totalCount} complete
            {runningCount > 0 ? ` · ${runningCount} running` : ''}
            {failedCount > 0 ? ` · ${failedCount} failed` : ''}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <input
            ref={fileInputRef}
            className="hidden"
            type="file"
            accept=".jpg,.jpeg,.png,.webp"
            multiple
            onChange={handleAddChange}
          />
          <Button variant="outline" size="sm" onClick={handleAddClick} disabled={batch.locked}>
            <Plus className="h-4 w-4" />
            Add images
          </Button>
          <Select value={filter} onValueChange={(value) => setFilter(value as FilterMode)}>
            <SelectTrigger className="h-8 w-[130px] text-xs">
              <SelectValue placeholder="Filter" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All objects</SelectItem>
              <SelectItem value="accepted">Accepted only</SelectItem>
              <SelectItem value="rejected">Rejected only</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm" onClick={handleExportYolo} disabled={!doneItems.length}>
            <Download className="h-4 w-4" />
            YOLO
          </Button>
          <Button variant="outline" size="sm" onClick={handleExportCoco} disabled={!doneItems.length}>
            <Download className="h-4 w-4" />
            COCO
          </Button>
          {canPause && (
            <Button variant="outline" size="sm" onClick={toggleBatchPause}>
              <Pause className="h-4 w-4" />
              {batch.paused ? 'Resume' : 'Pause'}
            </Button>
          )}
          {!batch.isRunning ? (
            <Button size="sm" onClick={runBatchDetection} disabled={!hasRunnable}>
              <Play className="h-4 w-4" />
              Run Batch
            </Button>
          ) : (
            <Button variant="outline" size="sm" onClick={cancelBatch}>
              <XCircle className="h-4 w-4" />
              Stop
            </Button>
          )}
          {failedCount > 0 && !batch.isRunning && (
            <Button variant="ghost" size="sm" onClick={retryBatchFailed}>
              <RotateCcw className="h-4 w-4" />
              Retry failed
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="divide-y divide-[var(--border-muted)]">
          {batch.items.map((item, index) => {
            const result = item.result
            const counts = result ? reviewCounts(result) : null
            return (
              <div key={item.id} className="relative flex items-center justify-between px-6 py-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="text-xs text-[var(--text-secondary)] w-7 text-right">{index + 1}</div>
                  <div className="min-w-0">
                    <div className="text-sm font-medium truncate">{item.fileName}</div>
                    {result ? (
                      <div className="flex flex-wrap items-center gap-2 text-[11px] text-[var(--text-secondary)]">
                        <span className="px-2 py-0.5 rounded-full border border-emerald-400/40 text-emerald-500 bg-emerald-400/10">
                          Accepted {counts?.accepted ?? 0}
                        </span>
                        <span className="px-2 py-0.5 rounded-full border border-red-400/40 text-red-500 bg-red-400/10">
                          Rejected {counts?.rejected ?? 0}
                        </span>
                        <span className="px-2 py-0.5 rounded-full border border-[var(--border-muted)] text-[var(--text-secondary)]">
                          All {counts?.all ?? 0}
                        </span>
                      </div>
                    ) : (
                      <div className="text-xs text-[var(--text-secondary)]">
                        {item.error || 'Waiting'}
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className={statusBadge(item.status)}>{item.status}</span>
                  {!result && (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setPreviewItemId(item.id)}
                    >
                      <Eye className="h-4 w-4" />
                      Preview
                    </Button>
                  )}
                  {item.status === 'done' && (
                    <Button size="sm" variant="ghost" onClick={() => openBatchResult(item.id)}>
                      Open
                    </Button>
                  )}
                  {item.status !== 'running' && item.status !== 'queued' && (
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => resetBatchItem(item.id)}
                      aria-label="Re-queue detection"
                      title="Re-queue detection"
                    >
                      <RefreshCw className="h-4 w-4 text-[var(--text-secondary)]" />
                    </Button>
                  )}
                  {item.status !== 'running' && (
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => removeBatchItem(item.id)}
                      aria-label="Remove from batch"
                      title="Remove from batch"
                    >
                      <Trash2 className="h-4 w-4 text-[var(--text-secondary)]" />
                    </Button>
                  )}
                </div>
                {item.status === 'running' && (
                  <div className="absolute left-0 right-0 bottom-0 h-2 bg-blue-500/20">
                    <div className="h-full w-1/3 bg-blue-500 animate-pulse" />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
