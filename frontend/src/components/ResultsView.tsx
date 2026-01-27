import { useEffect, useMemo, useRef, useState } from 'react'
import { useAppStore } from '@/stores/appStore'
import { useHistoryStore } from '@/stores/historyStore'
import { CanvasView, type CanvasViewHandle } from '@/components/CanvasView'
import { ObjectSidebar } from '@/components/ObjectSidebar'
import { objectKey } from '@/lib/objectKey'
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'
import { useInlineEdit, type EditDraft } from '@/hooks/useInlineEdit'
import { createResultObject, deleteResultObject, updateResultObject } from '@/lib/api'
import { buildYoloClasses, exportCoco, exportLabelMe, exportYolo, type ExportFormat } from '@/lib/exportFormats'
import { generatePdfReport, getImageAsDataUrl } from '@/lib/pdfExport'

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
  const pushHistory = useHistoryStore((state) => state.pushAction)
  const [hiddenClasses, setHiddenClasses] = useState<Set<string>>(new Set())
  const [confidenceFilter, setConfidenceFilter] = useState(0)
  const canvasRef = useRef<CanvasViewHandle>(null)
  const edit = useInlineEdit()
  const [editError, setEditError] = useState<string | null>(null)
  const [isCreating, setIsCreating] = useState(false)
  const [createDraft, setCreateDraft] = useState<EditDraft | null>(null)
  const [createError, setCreateError] = useState<string | null>(null)

  if (!result) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-[var(--text-secondary)]">
        No results to display.
      </div>
    )
  }

  const handleExport = (format: ExportFormat, filter: 'all' | 'accepted' | 'rejected' | 'visible') => {
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
        } catch (error) {
          console.error('PDF export failed:', error)
          // Fallback: try without image
          try {
            const blob = await generatePdfReport(result, reviewStatus, '')
            download(blob, `${baseName}-report.pdf`)
          } catch (fallbackError) {
            console.error('PDF export fallback failed:', fallbackError)
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
    return result.objects.filter((obj) => {
      const key = normalizeKey(obj.Object)
      if (hiddenClasses.has(key)) return false
      return obj.Score >= confidenceFilter
    })
  }, [result.objects, hiddenClasses, confidenceFilter])

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
    if (!selectedObjectKey) return null
    return result.objects.find((obj) => objectKey(obj) === selectedObjectKey) || null
  }, [result.objects, selectedObjectKey])

  useEffect(() => {
    if (!selectedObjectKey) {
      edit.cancelEditing()
    }
  }, [selectedObjectKey])

  useEffect(() => {
    if (!isCreating) {
      setCreateDraft(null)
      setCreateError(null)
    }
  }, [isCreating])

  const selectAndCenter = (key: string | null) => {
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
    const obj = result.objects.find((o) => objectKey(o) === key)
    if (!obj) return
    try {
      const updated = await updateResultObject(result.id, obj.Index, { ReviewStatus: status })
      updateObject(updated)
    } catch (error) {
      // Keep UX responsive; surface error only if needed later.
      console.warn('Failed to persist review status', error)
    }
  }

  const setReviewStatusPersisted = (key: string, status: 'accepted' | 'rejected' | null) => {
    setReviewStatusWithHistory(key, status)
    void persistReviewStatus(key, status)
  }

  useKeyboardShortcuts({
    objects: visibleObjects,
    selectedObjectKey,
    onSelectObject: setSelectedObjectKey,
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
    if (!selectedObject) return
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
    if (!selectedObject || !edit.draft) return
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
    if (!createDraft) return
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

  return (
    <div className="flex h-full">
      <div className="flex-1 relative">
        <CanvasView
          ref={canvasRef}
          key={resultRunId}
          imageUrl={result.image_url}
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
      </div>
      <div className="w-[320px] shrink-0">
        <ObjectSidebar
          objects={result.objects}
          visibleObjects={visibleObjects}
          hiddenClasses={hiddenClasses}
          confidenceFilter={confidenceFilter}
          onToggleClass={toggleClass}
          onConfidenceChange={setConfidenceFilter}
          reviewStatus={reviewStatus}
          onSetReviewStatus={setReviewStatusAndAdvancePersisted}
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
