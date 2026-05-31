import { useEffect, useMemo, useRef, useState } from 'react'
import { useAppStore } from '@/stores/appStore'
import { useHistoryStore } from '@/stores/historyStore'
import { CanvasView, type CanvasViewHandle } from '@/components/CanvasView'
import { ObjectSidebar } from '@/components/ObjectSidebar'
import { objectKey } from '@/lib/objectKey'
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'
import { useInlineEdit, type EditDraft } from '@/hooks/useInlineEdit'
import { createResultObject, deleteResultObject, exportResultsToExcel, updateResultObject } from '@/lib/api'
import { buildYoloClasses, exportCoco, exportLabelMe, exportYolo, type ExportFormat } from '@/lib/exportFormats'
import { generatePdfReport, getImageAsDataUrl } from '@/lib/pdfExport'
import { GripVertical } from 'lucide-react'
import { cn } from '@/lib/utils'

export function ResultsView() {
  const result = useAppStore((state) => state.result)
  const resultRunId = useAppStore((state) => state.resultRunId)
  const reviewStatus = useAppStore((state) => state.reviewStatus)
  const setReviewStatus = useAppStore((state) => state.setReviewStatus)
  const selectedObjectKey = useAppStore((state) => state.selectedObjectKey)
  const setSelectedObjectKey = useAppStore((state) => state.setSelectedObjectKey)
  const updateObject = useAppStore((state) => state.updateObject)
  const addObject = useAppStore((state) => state.addObject)
  const removeObject = useAppStore((state) => state.removeObject)
  const undoAction = useAppStore((state) => state.undoAction)
  const redoAction = useAppStore((state) => state.redoAction)
  const confidenceFilter = useAppStore((state) => state.confidenceFilter)
  const setConfidenceFilter = useAppStore((state) => state.setConfidenceFilter)
  const batch = useAppStore((state) => state.batch)
  const pushHistory = useHistoryStore((state) => state.pushAction)
  const [hiddenClasses, setHiddenClasses] = useState<Set<string>>(new Set())
  const canvasRef = useRef<CanvasViewHandle>(null)
  const edit = useInlineEdit()
  const [editError, setEditError] = useState<string | null>(null)
  const [isCreating, setIsCreating] = useState(false)
  const [createDraft, setCreateDraft] = useState<EditDraft | null>(null)
  const [createError, setCreateError] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(
    () => (typeof window !== 'undefined' ? window.innerWidth >= 1024 : true)
  )
  const [localImageUrl, setLocalImageUrl] = useState<string | null>(null)
  const [syncNotice, setSyncNotice] = useState<string | null>(null)
  const hasResult = Boolean(result)

  const handleExport = (format: ExportFormat, filter: 'all' | 'accepted' | 'rejected' | 'visible') => {
    if (!result) return
    const exportObjects = (() => {
      if (filter === 'visible') return visibleObjects
      if (filter === 'accepted') {
        return result.objects.filter((obj) => reviewStatus[objectKey(obj)] === 'accepted')
      }
      if (filter === 'rejected') {
        return result.objects.filter((obj) => reviewStatus[objectKey(obj)] === 'rejected')
      }
      return result.objects
    })()
    const imageFileName = result.image_url.split('/').pop() || 'image.png'
    const baseName = imageFileName.replace(/\.[^.]+$/, '')

    const download = (blob: Blob, filename: string) => {
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.click()
      URL.revokeObjectURL(url)
    }

    if (format === 'json') {
      download(
        new Blob([JSON.stringify(exportObjects, null, 2)], { type: 'application/json' }),
        `garnet-results-${filter}.json`
      )
      return
    }

    if (format === 'excel') {
      ; (async () => {
        try {
          const blob = await exportResultsToExcel(
            [{ file_name: imageFileName, objects: exportObjects.map((obj) => ({ ...obj })) }],
            `garnet-results-${filter}.xlsx`
          )
          download(blob, `garnet-results-${filter}.xlsx`)
        } catch {
          setSyncNotice('Unable to export Excel right now. Please try again.')
        }
      })()
      return
    }

    if (!result.image_width || !result.image_height) {
      download(
        new Blob([JSON.stringify({ error: 'Missing image size for export' }, null, 2)], { type: 'application/json' }),
        `garnet-export-error.json`
      )
      return
    }

    if (format === 'yolo') {
      const { classNames, classIdMap } = buildYoloClasses(exportObjects)
      const txt = exportYolo(exportObjects, result.image_width, result.image_height, classIdMap)
      download(new Blob([txt], { type: 'text/plain' }), `${baseName}.txt`)
      download(new Blob([classNames.join('\n') + '\n'], { type: 'text/plain' }), `${baseName}.classes.txt`)
      return
    }

    if (format === 'coco') {
      const coco = exportCoco(exportObjects, result.image_width, result.image_height, imageFileName)
      download(new Blob([JSON.stringify(coco, null, 2)], { type: 'application/json' }), `${baseName}.coco.json`)
      return
    }

    if (format === 'pdf') {
      // PDF export is async
      ; (async () => {
        try {
          const imageDataUrl = await getImageAsDataUrl(result.image_url)
          const blob = await generatePdfReport(result, reviewStatus, imageDataUrl)
          download(blob, `${baseName}-report.pdf`)
        } catch {
          // Fallback: try without image
          try {
            const blob = await generatePdfReport(result, reviewStatus, '')
            download(blob, `${baseName}-report.pdf`)
          } catch {
            setSyncNotice('Unable to generate PDF report right now. Please try again.')
          }
        }
      })()
      return
    }

    const labelme = exportLabelMe(exportObjects, result.image_width, result.image_height, imageFileName)
    download(new Blob([JSON.stringify(labelme, null, 2)], { type: 'application/json' }), `${baseName}.labelme.json`)
  }

  const normalizeKey = (value: string) => value.toLowerCase().replace(/_/g, ' ').trim()

  const visibleObjects = useMemo(() => {
    if (!result) return []
    return result.objects.filter((obj) => {
      const key = normalizeKey(obj.Object)
      if (hiddenClasses.has(key)) return false
      return obj.Score >= confidenceFilter
    })
  }, [result, hiddenClasses, confidenceFilter])

  const toggleClass = (classKey: string) => {
    setHiddenClasses((prev) => {
      const next = new Set(prev)
      if (next.has(classKey)) {
        next.delete(classKey)
      } else {
        next.add(classKey)
      }
      return next
    })
  }

  const selectedObject = useMemo(() => {
    if (!result || !selectedObjectKey) return null
    return result.objects.find((obj) => objectKey(obj) === selectedObjectKey) || null
  }, [result, selectedObjectKey])

  useEffect(() => {
    if (!selectedObjectKey) {
      edit.cancelEditing()
    }
  }, [selectedObjectKey])

  const activeFile = useMemo(() => {
    if (!batch.activeItemId) return null
    return batch.items.find((item) => item.id === batch.activeItemId)?.file || null
  }, [batch.activeItemId, batch.items])

  useEffect(() => {
    if (!activeFile) {
      setLocalImageUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev)
        return null
      })
      return
    }
    const url = URL.createObjectURL(activeFile)
    setLocalImageUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return url
    })
    return () => {
      URL.revokeObjectURL(url)
    }
  }, [activeFile])

  useEffect(() => {
    if (!batch.activeItemId) return
    setHiddenClasses(new Set())
    setSelectedObjectKey(null)
    edit.cancelEditing()
    setIsCreating(false)
    setCreateDraft(null)
    setEditError(null)
    setCreateError(null)
  }, [batch.activeItemId])

  useEffect(() => {
    if (!isCreating) {
      setCreateDraft(null)
      setCreateError(null)
    }
  }, [isCreating])

  useEffect(() => {
    if (!syncNotice) return
    const timeoutId = window.setTimeout(() => setSyncNotice(null), 4000)
    return () => window.clearTimeout(timeoutId)
  }, [syncNotice])

  const selectAndCenter = (key: string | null) => {
    if (!result) return
    setSelectedObjectKey(key)
    if (!key || !canvasRef.current) return
    const obj = result.objects.find((o) => objectKey(o) === key)
    if (obj) {
      canvasRef.current.centerOnObject(obj)
    }
  }

  const setReviewStatusWithHistory = (key: string, status: 'accepted' | 'rejected' | null) => {
    // Record previous status for undo
    const prevStatus = reviewStatus[key] || null
    pushHistory({ type: 'review', key, prev: prevStatus, next: status })
    setReviewStatus(key, status)
  }

  const navigatePrevious = () => {
    if (!selectedObjectKey) {
      if (visibleObjects.length > 0) {
        selectAndCenter(objectKey(visibleObjects[visibleObjects.length - 1]))
      }
      return
    }
    const orderedKeys = visibleObjects.map((obj) => objectKey(obj))
    const currentIndex = orderedKeys.indexOf(selectedObjectKey)
    if (currentIndex > 0) {
      selectAndCenter(orderedKeys[currentIndex - 1])
    }
  }

  const navigateNext = () => {
    if (!selectedObjectKey) {
      if (visibleObjects.length > 0) {
        selectAndCenter(objectKey(visibleObjects[0]))
      }
      return
    }
    const orderedKeys = visibleObjects.map((obj) => objectKey(obj))
    const currentIndex = orderedKeys.indexOf(selectedObjectKey)
    if (currentIndex < orderedKeys.length - 1) {
      selectAndCenter(orderedKeys[currentIndex + 1])
    }
  }

  const persistReviewStatus = async (key: string, status: 'accepted' | 'rejected' | null) => {
    if (!result) return
    const obj = result.objects.find((o) => objectKey(o) === key)
    if (!obj) return
    try {
      const updated = await updateResultObject(result.id, obj.Index, { ReviewStatus: status })
      updateObject(updated)
    } catch {
      setSyncNotice('Saved locally, but failed to sync review status to backend.')
    }
  }

  const setReviewStatusPersisted = (key: string, status: 'accepted' | 'rejected' | null) => {
    setReviewStatusWithHistory(key, status)
    void persistReviewStatus(key, status)
  }

  useKeyboardShortcuts({
    objects: visibleObjects,
    selectedObjectKey,
    onSelectObject: selectAndCenter,
    onAccept: (key) => setReviewStatusPersisted(key, 'accepted'),
    onReject: (key) => setReviewStatusPersisted(key, 'rejected'),
    onFit: () => canvasRef.current?.fitToScreen(),
    onReset: () => canvasRef.current?.resetZoom(),
    onZoomIn: () => canvasRef.current?.zoomIn(),
    onZoomOut: () => canvasRef.current?.zoomOut(),
    onUndo: undoAction,
    onRedo: redoAction,
  })

  const handleDeleteSelected = async () => {
    if (!result || !selectedObject) return
    setEditError(null)
    setCreateError(null)
    try {
      const index = result.objects.indexOf(selectedObject)
      await deleteResultObject(result.id, selectedObject.Index)
      pushHistory({ type: 'delete', object: selectedObject, index })
      removeObject(selectedObject.Index)
      selectAndCenter(null)
    } catch (error) {
      setEditError(error instanceof Error ? error.message : 'Unable to delete object')
    }
  }

  const handleSaveEdit = async () => {
    if (!result || !selectedObject || !edit.draft) return
    setEditError(null)
    const payload = {
      Object: edit.draft.Object,
      Text: edit.draft.Text,
      Left: Number(edit.draft.Left),
      Top: Number(edit.draft.Top),
      Width: Number(edit.draft.Width),
      Height: Number(edit.draft.Height),
    }
    try {
      const updated = await updateResultObject(result.id, selectedObject.Index, payload)
      pushHistory({ type: 'update', prev: selectedObject, next: updated })
      updateObject(updated)
      edit.cancelEditing()
    } catch (error) {
      setEditError(error instanceof Error ? error.message : 'Unable to save edits')
    }
  }

  const handleCreateSave = async () => {
    if (!result || !createDraft) return
    setCreateError(null)
    const payload = {
      Object: createDraft.Object,
      Text: createDraft.Text,
      Left: Number(createDraft.Left),
      Top: Number(createDraft.Top),
      Width: Number(createDraft.Width),
      Height: Number(createDraft.Height),
    }
    try {
      const created = await createResultObject(result.id, payload)
      pushHistory({ type: 'create', object: created })
      addObject(created)
      setSelectedObjectKey(objectKey(created))
      setIsCreating(false)
      setCreateDraft(null)
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : 'Unable to add new object')
    }
  }

  const handleSidebarSelect = (key: string) => {
    selectAndCenter(key)
  }

  if (!hasResult && batch.activeItemId) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-sm text-[var(--text-secondary)]">
        {localImageUrl ? (
          <div className="max-w-[85%] max-h-[85%] relative">
            <img
              src={localImageUrl}
              alt="Batch preview"
              className="rounded-xl border border-[var(--border-muted)] max-h-[70vh]"
            />
            <div className="absolute bottom-3 right-3 text-[11px] px-2 py-1 rounded-lg bg-black/60 text-white">
              Waiting for results...
            </div>
          </div>
        ) : (
          <div>Waiting for results...</div>
        )}
      </div>
    )
  }

  if (!hasResult) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-[var(--text-secondary)]">
        No results to display.
      </div>
    )
  }

  return (
    <div className="flex h-full relative">
      <div className="flex-1 relative">
        <CanvasView
          ref={canvasRef}
          key={resultRunId}
          imageUrl={localImageUrl ?? result!.image_url}
          objects={visibleObjects}
          selectedObjectKey={selectedObjectKey}
          selectedObject={selectedObject}
          reviewStatus={reviewStatus}
          onSelectObject={(key) => {
            if (isCreating) return
            setSelectedObjectKey(key)
          }}
          onSetReviewStatus={setReviewStatusPersisted}
          isEditing={edit.isEditing}
          editDraft={edit.draft}
          onStartEdit={(obj) => {
            setEditError(null)
            edit.startEditing(obj)
          }}
          onCancelEdit={() => {
            setEditError(null)
            edit.cancelEditing()
          }}
          onChangeEdit={(field, value) => {
            if (!edit.draft) return
            if (field === 'Object' || field === 'Text') {
              edit.setDraft({ ...edit.draft, [field]: value })
              return
            }
            const parsed = Number(value)
            edit.setDraft({
              ...edit.draft,
              [field]: Number.isFinite(parsed) ? parsed : edit.draft[field],
            })
          }}
          onReplaceEditDraft={(next) => edit.setDraft(next)}
          onSaveEdit={handleSaveEdit}
          onDeleteSelected={handleDeleteSelected}
          onNavigatePrevious={navigatePrevious}
          onNavigateNext={navigateNext}
          isCreating={isCreating}
          createDraft={createDraft}
          onCreateDraftChange={setCreateDraft}
          fitKey={String(resultRunId)}
        />
        {editError && (
          <div className="absolute bottom-6 left-6 max-w-sm text-xs text-[var(--danger)] bg-[var(--bg-secondary)] border border-[var(--border-muted)] px-3 py-2 rounded-lg">
            {editError}
          </div>
        )}
        {createError && (
          <div className="absolute bottom-6 right-6 max-w-sm text-xs text-[var(--danger)] bg-[var(--bg-secondary)] border border-[var(--border-muted)] px-3 py-2 rounded-lg">
            {createError}
          </div>
        )}
        {syncNotice && (
          <div className="absolute top-4 left-4 z-20 max-w-sm text-xs text-[var(--danger)] bg-[var(--bg-secondary)] border border-[var(--border-muted)] px-3 py-2 rounded-lg">
            {syncNotice}
          </div>
        )}
      </div>
      {/* Sidebar toggle handle */}
      <button
        type="button"
        onClick={() => setSidebarOpen((prev) => !prev)}
        className={cn(
          'absolute right-0 top-0 z-40 h-full w-7 md:static md:w-3 shrink-0 flex items-center justify-center',
          'bg-[var(--bg-secondary)] border-l-2 border-[var(--border-muted)] shadow-sm md:shadow-none',
          'hover:bg-[var(--bg-primary)] hover:border-[var(--accent)] transition-colors cursor-pointer'
        )}
        title={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
      >
        <GripVertical className="h-4 w-4 text-[var(--text-secondary)]" />
      </button>
      {/* Sidebar */}
      <div className={cn(
        'absolute right-0 top-0 z-30 h-full md:relative md:z-auto shrink-0 transition-all duration-200 ease-out overflow-hidden',
        sidebarOpen ? 'w-[85vw] max-w-[320px] md:w-[320px]' : 'w-0'
      )}>
        <ObjectSidebar
          objects={result!.objects}
          visibleObjects={visibleObjects}
          hiddenClasses={hiddenClasses}
          confidenceFilter={confidenceFilter}
          onToggleClass={toggleClass}
          onConfidenceChange={setConfidenceFilter}
          reviewStatus={reviewStatus}
          onSetReviewStatus={setReviewStatusPersisted}
          selectedObjectKey={selectedObjectKey}
          onSelectObject={handleSidebarSelect}
          isCreating={isCreating}
          createDraft={createDraft}
          onStartCreate={() => {
            setCreateError(null)
            setIsCreating(true)
            setCreateDraft((draft) => draft ?? {
              Object: selectedObject?.Object ?? 'custom',
              Left: 0,
              Top: 0,
              Width: 0,
              Height: 0,
              Text: '',
            })
            setSelectedObjectKey(null)
            edit.cancelEditing()
          }}
          onCancelCreate={() => {
            setIsCreating(false)
            setCreateDraft(null)
          }}
          onUpdateCreateDraft={(field, value) => {
            setCreateDraft((draft) => {
              if (!draft) return draft
              if (field === 'Object' || field === 'Text') {
                return { ...draft, [field]: value }
              }
              const parsed = Number(value)
              return { ...draft, [field]: Number.isFinite(parsed) ? parsed : draft[field] }
            })
          }}
          onSaveCreate={handleCreateSave}
          onExport={handleExport}
        />
      </div>
    </div>
  )
}
