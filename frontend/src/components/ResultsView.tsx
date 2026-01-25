import { useEffect, useMemo, useRef, useState } from 'react'
import { useAppStore } from '@/stores/appStore'
import { CanvasView, type CanvasViewHandle } from '@/components/CanvasView'
import { ObjectSidebar } from '@/components/ObjectSidebar'
import { objectKey } from '@/lib/objectKey'
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'
import { useInlineEdit } from '@/hooks/useInlineEdit'
import { updateResultObject } from '@/lib/api'

export function ResultsView() {
  const result = useAppStore((state) => state.result)
  const resultRunId = useAppStore((state) => state.resultRunId)
  const reviewStatus = useAppStore((state) => state.reviewStatus)
  const setReviewStatus = useAppStore((state) => state.setReviewStatus)
  const selectedObjectKey = useAppStore((state) => state.selectedObjectKey)
  const setSelectedObjectKey = useAppStore((state) => state.setSelectedObjectKey)
  const updateObject = useAppStore((state) => state.updateObject)
  const [hiddenClasses, setHiddenClasses] = useState<Set<string>>(new Set())
  const [confidenceFilter, setConfidenceFilter] = useState(0)
  const canvasRef = useRef<CanvasViewHandle>(null)
  const edit = useInlineEdit()
  const [editError, setEditError] = useState<string | null>(null)

  if (!result) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-[var(--text-secondary)]">
        No results to display.
      </div>
    )
  }

  const handleExport = (filter: 'all' | 'accepted' | 'rejected' | 'visible') => {
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
    const blob = new Blob([JSON.stringify(exportObjects, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `garnet-results-${filter}.json`
    link.click()
    URL.revokeObjectURL(url)
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

  useKeyboardShortcuts({
    objects: visibleObjects,
    selectedObjectKey,
    onSelectObject: setSelectedObjectKey,
    onAccept: (key) => setReviewStatus(key, 'accepted'),
    onReject: (key) => setReviewStatus(key, 'rejected'),
    onFit: () => canvasRef.current?.fitToScreen(),
    onReset: () => canvasRef.current?.resetZoom(),
    onZoomIn: () => canvasRef.current?.zoomIn(),
    onZoomOut: () => canvasRef.current?.zoomOut(),
  })

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
      updateObject(updated)
      edit.cancelEditing()
    } catch (error) {
      setEditError(error instanceof Error ? error.message : 'Unable to save edits')
    }
  }

  const handleSidebarSelect = (key: string) => {
    setSelectedObjectKey(key)
    const obj = result.objects.find((o) => objectKey(o) === key)
    if (obj && canvasRef.current) {
      canvasRef.current.centerOnObject(obj)
    }
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
          onSelectObject={setSelectedObjectKey}
          onSetReviewStatus={setReviewStatus}
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
          fitKey={String(resultRunId)}
        />
        {editError && (
          <div className="absolute bottom-6 left-6 max-w-sm text-xs text-[var(--danger)] bg-[var(--bg-secondary)] border border-[var(--border-muted)] px-3 py-2 rounded-lg">
            {editError}
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
          onSetReviewStatus={setReviewStatus}
          selectedObjectKey={selectedObjectKey}
          onSelectObject={handleSidebarSelect}
          onExport={handleExport}
        />
      </div>
    </div>
  )
}
