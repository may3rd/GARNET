import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, RotateCcw, RotateCw, X } from 'lucide-react'
import type {
  DetectedObject,
  PipelineArtifact,
  PipelineReviewBucket,
  PipelineReviewDecision,
  PipelineReviewItem,
  PipelineReviewState,
} from '@/types'
import { CanvasView, type CanvasViewHandle } from '@/components/CanvasView'
import { ObjectSidebar } from '@/components/ObjectSidebar'
import { objectKey } from '@/lib/objectKey'
import { getPipelineReviewState, putPipelineReviewState } from '@/lib/api'

type WorkspaceDraft = {
  Object: string
  Left: number
  Top: number
  Width: number
  Height: number
  Text: string
}

type PipelineHitlReviewViewProps = {
  jobId: string
  activeBucket: PipelineReviewBucket
  itemsByBucket: Record<PipelineReviewBucket, PipelineReviewItem[]>
  imageArtifacts: PipelineArtifact[]
  initialReviewDecisions: Record<string, PipelineReviewDecision>
  onApply: (decisions: Record<string, PipelineReviewDecision>) => void
  onClose: () => void
}

type WorkspaceSnapshot = {
  bucketStates: Record<PipelineReviewBucket, DetectedObject[]>
  reviewDecisions: Record<string, PipelineReviewDecision>
  workspaceBucket: PipelineReviewBucket
  selectedObjectKey: string | null
}

const PIPELINE_WORKSPACE_STORAGE_PREFIX = 'garnet-pipeline-workspace'

function workspaceKey(jobId: string, bucket: PipelineReviewBucket) {
  return `${PIPELINE_WORKSPACE_STORAGE_PREFIX}:${jobId}:${bucket}`
}

function bucketObjectLabel(bucket: PipelineReviewBucket) {
  switch (bucket) {
    case 'stage4_line_number':
    case 'stage12_line_attachment':
      return 'line_number'
    case 'stage4_instrument':
      return 'instrument_semantic'
    case 'stage12_instrument_attachment':
      return 'instrument_attachment'
  }
}

function seedObjects(bucket: PipelineReviewBucket, items: PipelineReviewItem[]): DetectedObject[] {
  return items.map((item, index) => {
    const bbox = item.bbox as Record<string, number> | undefined
    const left = bbox?.x_min ?? 0
    const top = bbox?.y_min ?? 0
    const width = bbox ? Math.max(1, (bbox.x_max ?? 0) - (bbox.x_min ?? 0)) : 1
    const height = bbox ? Math.max(1, (bbox.y_max ?? 0) - (bbox.y_min ?? 0)) : 1
    const objectId = index + 1
    return {
      Index: objectId,
      Object: bucketObjectLabel(bucket),
      CategoryID: bucket.startsWith('stage4') ? 4 : 12,
      ObjectID: objectId,
      Left: left,
      Top: top,
      Width: width,
      Height: height,
      Score: 1,
      Text: item.text || item.normalizedText || item.title,
      ReviewStatus: null,
    }
  })
}

function pickBaseImageUrl(imageArtifacts: PipelineArtifact[]): string {
  const preferred = [
    'stage1_gray.png',
    'stage1_gray_equalized.png',
  ]
  for (const name of preferred) {
    const match = imageArtifacts.find((artifact) => artifact.name === name)
    if (match) return match.url
  }
  const firstNonOverlay = imageArtifacts.find((artifact) => !artifact.name.includes('overlay'))
  return firstNonOverlay?.url ?? imageArtifacts[0]?.url ?? ''
}

export function PipelineHitlReviewView({
  jobId,
  activeBucket,
  itemsByBucket,
  imageArtifacts,
  initialReviewDecisions,
  onApply,
  onClose,
}: PipelineHitlReviewViewProps) {
  const canvasRef = useRef<CanvasViewHandle>(null)
  const selectedItemRef = useRef<HTMLLIElement | null>(null)
  const historyRef = useRef<WorkspaceSnapshot[]>([])
  const futureRef = useRef<WorkspaceSnapshot[]>([])
  const [bucketStates, setBucketStates] = useState<Record<PipelineReviewBucket, DetectedObject[]>>({
    stage4_line_number: [],
    stage4_instrument: [],
    stage12_line_attachment: [],
    stage12_instrument_attachment: [],
  })
  const [workspaceBucket, setWorkspaceBucket] = useState<PipelineReviewBucket>(activeBucket)
  const [draftReviewDecisions, setDraftReviewDecisions] = useState<Record<string, PipelineReviewDecision>>(initialReviewDecisions)
  const [reviewStatus, setReviewStatus] = useState<Record<string, 'accepted' | 'rejected'>>({})
  const [selectedObjectKey, setSelectedObjectKey] = useState<string | null>(null)
  const [hiddenClasses, setHiddenClasses] = useState<Set<string>>(new Set())
  const [confidenceFilter, setConfidenceFilter] = useState(0)
  const [isCreating, setIsCreating] = useState(false)
  const [createDraft, setCreateDraft] = useState<WorkspaceDraft | null>(null)
  const [editDraft, setEditDraft] = useState<WorkspaceDraft | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [fitKey, setFitKey] = useState(`bucket:${activeBucket}`)
  const [historyVersion, setHistoryVersion] = useState(0)
  const [isSaving, setIsSaving] = useState(false)
  const [workspaceError, setWorkspaceError] = useState<string | null>(null)

  const snapshotState = (
    overrides?: Partial<Pick<WorkspaceSnapshot, 'bucketStates' | 'reviewDecisions' | 'workspaceBucket' | 'selectedObjectKey'>>
  ): WorkspaceSnapshot => ({
    bucketStates: overrides?.bucketStates ?? bucketStates,
    reviewDecisions: overrides?.reviewDecisions ?? draftReviewDecisions,
    workspaceBucket: overrides?.workspaceBucket ?? workspaceBucket,
    selectedObjectKey: overrides?.selectedObjectKey ?? selectedObjectKey,
  })

  const pushHistory = (snapshot: WorkspaceSnapshot) => {
    historyRef.current.push({
      bucketStates: structuredClone(snapshot.bucketStates),
      reviewDecisions: { ...snapshot.reviewDecisions },
      workspaceBucket: snapshot.workspaceBucket,
      selectedObjectKey: snapshot.selectedObjectKey,
    })
    if (historyRef.current.length > 50) {
      historyRef.current.shift()
    }
    futureRef.current = []
    setHistoryVersion((value) => value + 1)
  }

  const restoreSnapshot = (snapshot: WorkspaceSnapshot) => {
    setBucketStates(structuredClone(snapshot.bucketStates))
    setDraftReviewDecisions({ ...snapshot.reviewDecisions })
    setWorkspaceBucket(snapshot.workspaceBucket)
    setSelectedObjectKey(snapshot.selectedObjectKey)
    setFitKey(`bucket:${snapshot.workspaceBucket}`)
  }

  useEffect(() => {
    setWorkspaceBucket(activeBucket)
    setFitKey(`bucket:${activeBucket}`)
  }, [activeBucket])

  useEffect(() => {
    setDraftReviewDecisions(initialReviewDecisions)
  }, [initialReviewDecisions])

  useEffect(() => {
    let active = true
    const load = async () => {
      setWorkspaceError(null)
      let remoteState: PipelineReviewState | null = null
      try {
        remoteState = await getPipelineReviewState(jobId)
      } catch (error) {
        setWorkspaceError(error instanceof Error ? error.message : 'Failed to load review state')
      }

      const nextStates: Record<PipelineReviewBucket, DetectedObject[]> = {
        stage4_line_number: [],
        stage4_instrument: [],
        stage12_line_attachment: [],
        stage12_instrument_attachment: [],
      }
      ;(Object.keys(nextStates) as PipelineReviewBucket[]).forEach((bucket) => {
        const remoteWorkspace = remoteState?.workspace_objects?.[bucket]
        if (Array.isArray(remoteWorkspace) && remoteWorkspace.length > 0) {
          nextStates[bucket] = remoteWorkspace as unknown as DetectedObject[]
          return
        }
        const stored = window.localStorage.getItem(workspaceKey(jobId, bucket))
        if (stored) {
          try {
            const parsed = JSON.parse(stored) as DetectedObject[]
            if (parsed.length > 0 || itemsByBucket[bucket].length === 0) {
              nextStates[bucket] = parsed
              return
            }
          } catch {
            // fall through to seed from review items
          }
        }
        nextStates[bucket] = seedObjects(bucket, itemsByBucket[bucket])
      })
      if (!active) return
      setBucketStates(nextStates)
      if (remoteState) {
        const nextDecisions: Record<string, PipelineReviewDecision> = {}
        for (const item of remoteState.items) {
          nextDecisions[`${item.bucket}:${item.entity_id ?? item.item_id}`] = item.decision
        }
        setDraftReviewDecisions(nextDecisions)
      }
    }
    void load()
    return () => {
      active = false
    }
  }, [itemsByBucket, jobId])

  useEffect(() => {
    const nextStatus: Record<string, 'accepted' | 'rejected'> = {}
    itemsByBucket[workspaceBucket].forEach((item, index) => {
      const obj = bucketStates[workspaceBucket][index]
      if (!obj) return
      const decision = draftReviewDecisions[`${workspaceBucket}:${item.id}`]
      if (decision === 'accepted' || decision === 'rejected') {
        nextStatus[objectKey(obj)] = decision
      }
    })
    setReviewStatus(nextStatus)
  }, [bucketStates, itemsByBucket, draftReviewDecisions, workspaceBucket])

  const objects = bucketStates[workspaceBucket] ?? []
  const visibleObjects = useMemo(
    () => objects.filter((obj) => obj.Score >= confidenceFilter),
    [objects, confidenceFilter]
  )
  const selectedObject = useMemo(
    () => objects.find((obj) => objectKey(obj) === selectedObjectKey) ?? null,
    [objects, selectedObjectKey]
  )
  const imageUrl = useMemo(() => pickBaseImageUrl(imageArtifacts), [imageArtifacts])

  const selectAndCenter = (key: string | null) => {
    setSelectedObjectKey(key)
    if (!key || !canvasRef.current) return
    const obj = objects.find((item) => objectKey(item) === key)
    if (obj) {
      canvasRef.current.centerOnObject(obj)
    }
  }

  const setBucketObjects = (updater: (current: DetectedObject[]) => DetectedObject[]) => {
    pushHistory(snapshotState())
    setBucketStates((current) => ({
      ...current,
      [workspaceBucket]: updater(current[workspaceBucket] ?? []),
    }))
  }

  const handleSetReviewStatus = (key: string, status: 'accepted' | 'rejected' | null) => {
    pushHistory(snapshotState())
    setReviewStatus((current) => {
      const next = { ...current }
      if (status) next[key] = status
      else delete next[key]
      return next
    })
    const index = objects.findIndex((obj) => objectKey(obj) === key)
    const item = itemsByBucket[workspaceBucket][index]
    if (!item) return
    setDraftReviewDecisions((current) => ({
      ...current,
      [`${workspaceBucket}:${item.id}`]: status ?? 'deferred',
    }))
  }

  const handleDeleteSelected = () => {
    if (!selectedObject) return
    const key = objectKey(selectedObject)
    setBucketObjects((current) => current.filter((obj) => objectKey(obj) !== key))
    setSelectedObjectKey(null)
  }

  const handleSaveEdit = () => {
    if (!selectedObject || !editDraft) return
    const key = objectKey(selectedObject)
    pushHistory(snapshotState())
    setBucketObjects((current) =>
      current.map((obj) => (objectKey(obj) === key ? { ...obj, ...editDraft } : obj))
    )
    setIsEditing(false)
    setEditDraft(null)
  }

  const handleCreateSave = () => {
    if (!createDraft) return
    const nextId = objects.reduce((max, obj) => Math.max(max, obj.ObjectID), 0) + 1
    const created: DetectedObject = {
      Index: nextId,
      Object: createDraft.Object,
      CategoryID: workspaceBucket.startsWith('stage4') ? 4 : 12,
      ObjectID: nextId,
      Left: createDraft.Left,
      Top: createDraft.Top,
      Width: createDraft.Width,
      Height: createDraft.Height,
      Score: 1,
      Text: createDraft.Text,
      ReviewStatus: null,
    }
    pushHistory(snapshotState())
    setBucketObjects((current) => [...current, created])
    setIsCreating(false)
    setCreateDraft(null)
    setSelectedObjectKey(objectKey(created))
  }

  const handleUndo = () => {
    const snapshot = historyRef.current.pop()
    if (!snapshot) return
    futureRef.current.push(snapshotState())
    restoreSnapshot(snapshot)
    setHistoryVersion((value) => value + 1)
  }

  const handleRedo = () => {
    const snapshot = futureRef.current.pop()
    if (!snapshot) return
    historyRef.current.push(snapshotState())
    restoreSnapshot(snapshot)
    setHistoryVersion((value) => value + 1)
  }

  const handleApply = () => {
    ;(Object.keys(bucketStates) as PipelineReviewBucket[]).forEach((bucket) => {
      window.localStorage.setItem(workspaceKey(jobId, bucket), JSON.stringify(bucketStates[bucket]))
    })
    const payload = {
      items: Object.entries(draftReviewDecisions).map(([key, decision]) => {
        const [bucket, entityId] = key.split(':', 2) as [PipelineReviewBucket, string]
        return {
          item_id: key,
          bucket,
          entity_id: entityId,
          decision,
        }
      }),
      workspace_objects: bucketStates as unknown as PipelineReviewState['workspace_objects'],
    }
    setIsSaving(true)
    setWorkspaceError(null)
    void putPipelineReviewState(jobId, payload)
      .then((saved) => {
        const nextDecisions: Record<string, PipelineReviewDecision> = {}
        for (const item of saved.items) {
          nextDecisions[`${item.bucket}:${item.entity_id ?? item.item_id}`] = item.decision
        }
        onApply(nextDecisions)
        onClose()
      })
      .catch((error) => {
        setWorkspaceError(error instanceof Error ? error.message : 'Failed to save review state')
      })
      .finally(() => setIsSaving(false))
  }

  useEffect(() => {
    if (!selectedObjectKey) return
    const timeoutId = window.setTimeout(() => {
      selectedItemRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }, 40)
    return () => window.clearTimeout(timeoutId)
  }, [selectedObjectKey, workspaceBucket, visibleObjects.length])

  return (
    <div className="flex h-full flex-col bg-[var(--bg-primary)]">
      <div className="flex items-center justify-between border-b border-[var(--border-muted)] bg-[var(--bg-secondary)] px-6 py-4">
        <div>
          <div className="text-lg font-semibold">Pipeline HITL Review</div>
          <div className="text-xs text-[var(--text-secondary)]">Dedicated review workspace with the detection-mode layout.</div>
          {workspaceError ? <div className="mt-1 text-xs text-[var(--danger)]">{workspaceError}</div> : null}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleUndo}
            disabled={historyRef.current.length === 0}
            className="inline-flex items-center gap-2 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm font-semibold text-[var(--text-primary)] disabled:opacity-40"
          >
            <RotateCcw className="h-4 w-4" />
            Undo
          </button>
          <button
            type="button"
            onClick={handleRedo}
            disabled={futureRef.current.length === 0}
            className="inline-flex items-center gap-2 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm font-semibold text-[var(--text-primary)] disabled:opacity-40"
          >
            <RotateCw className="h-4 w-4" />
            Redo
          </button>
          <button
            type="button"
            onClick={onClose}
            className="inline-flex items-center gap-2 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm font-semibold text-[var(--text-primary)]"
          >
            <X className="h-4 w-4" />
            Cancel
          </button>
          <button
            type="button"
            onClick={handleApply}
            disabled={isSaving}
            className="inline-flex items-center gap-2 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-sm font-semibold text-emerald-700 disabled:opacity-40"
          >
            <Check className="h-4 w-4" />
            {isSaving ? 'Saving...' : 'OK'}
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 border-b border-[var(--border-muted)] bg-[var(--bg-secondary)] px-6 py-3">
        {([
          ['stage4_line_number', 'Stage 4 Line Numbers'],
          ['stage4_instrument', 'Stage 4 Instruments'],
          ['stage12_line_attachment', 'Stage 12 Line Attachments'],
          ['stage12_instrument_attachment', 'Stage 12 Instrument Attachments'],
        ] as Array<[PipelineReviewBucket, string]>).map(([bucket, label]) => (
          <button
            key={bucket}
            type="button"
            onClick={() => {
              setWorkspaceBucket(bucket)
              setFitKey(`bucket:${bucket}`)
              setSelectedObjectKey(null)
              setIsCreating(false)
              setCreateDraft(null)
              setIsEditing(false)
              setEditDraft(null)
            }}
            className={`rounded-full border px-3 py-1 text-xs font-semibold ${
              workspaceBucket === bucket
                ? 'border-[var(--accent)] bg-[var(--accent)]/10 text-[var(--accent)]'
                : 'border-[var(--border-muted)] bg-[var(--bg-primary)] text-[var(--text-secondary)]'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="flex h-full relative">
        <div className="flex-1 relative h-[calc(100vh-170px)] min-h-[720px]">
          <CanvasView
            ref={canvasRef}
            imageUrl={imageUrl}
            objects={visibleObjects}
            selectedObjectKey={selectedObjectKey}
            selectedObject={selectedObject}
            reviewStatus={reviewStatus}
            onSelectObject={(key) => {
              if (isCreating) return
              setSelectedObjectKey(key)
            }}
            onSetReviewStatus={handleSetReviewStatus}
            isEditing={isEditing}
            editDraft={editDraft}
            onStartEdit={(obj) => {
              setIsEditing(true)
              setEditDraft({
                Object: obj.Object,
                Left: obj.Left,
                Top: obj.Top,
                Width: obj.Width,
                Height: obj.Height,
                Text: obj.Text,
              })
            }}
            onCancelEdit={() => {
              setIsEditing(false)
              setEditDraft(null)
            }}
            onChangeEdit={(field, value) => {
              setEditDraft((current) => {
                if (!current) return current
                if (field === 'Object' || field === 'Text') {
                  return { ...current, [field]: value }
                }
                const parsed = Number(value)
                return { ...current, [field]: Number.isFinite(parsed) ? parsed : current[field] }
              })
            }}
            onReplaceEditDraft={(draft) => setEditDraft(draft)}
            onSaveEdit={handleSaveEdit}
            onDeleteSelected={handleDeleteSelected}
            onNavigatePrevious={() => {
              if (!selectedObjectKey) return
              const keys = visibleObjects.map((obj) => objectKey(obj))
              const index = keys.indexOf(selectedObjectKey)
              if (index > 0) selectAndCenter(keys[index - 1])
            }}
            onNavigateNext={() => {
              if (!selectedObjectKey) return
              const keys = visibleObjects.map((obj) => objectKey(obj))
              const index = keys.indexOf(selectedObjectKey)
              if (index >= 0 && index < keys.length - 1) selectAndCenter(keys[index + 1])
            }}
            isCreating={isCreating}
            createDraft={createDraft}
            onCreateDraftChange={setCreateDraft}
            fitKey={fitKey}
          />
        </div>
        <div className="w-[320px] border-l border-[var(--border-muted)] bg-[var(--bg-secondary)] overflow-y-auto">
          <ObjectSidebar
            objects={objects.map((obj) => {
              if (objectKey(obj) !== selectedObjectKey) return obj
              return {
                ...obj,
                Text: `${obj.Text}`,
              }
            })}
            visibleObjects={visibleObjects}
            hiddenClasses={hiddenClasses}
            confidenceFilter={confidenceFilter}
            onToggleClass={(classKey) => {
              setHiddenClasses((current) => {
                const next = new Set(current)
                if (next.has(classKey)) next.delete(classKey)
                else next.add(classKey)
                return next
              })
            }}
            onConfidenceChange={setConfidenceFilter}
            reviewStatus={reviewStatus}
            onSetReviewStatus={handleSetReviewStatus}
            selectedObjectKey={selectedObjectKey}
            onSelectObject={selectAndCenter}
            isCreating={isCreating}
            createDraft={createDraft}
            onStartCreate={() => {
              setIsCreating(true)
              setCreateDraft({
                Object: bucketObjectLabel(workspaceBucket),
                Left: 0,
                Top: 0,
                Width: 0,
                Height: 0,
                Text: '',
              })
              setSelectedObjectKey(null)
              setIsEditing(false)
              setEditDraft(null)
            }}
            onCancelCreate={() => {
              setIsCreating(false)
              setCreateDraft(null)
            }}
            onUpdateCreateDraft={(field, value) => {
              setCreateDraft((current) => {
                if (!current) return current
                if (field === 'Object' || field === 'Text') {
                  return { ...current, [field]: value }
                }
                const parsed = Number(value)
                return { ...current, [field]: Number.isFinite(parsed) ? parsed : current[field] }
              })
            }}
            onSaveCreate={handleCreateSave}
            onExport={(format) => {
              const payload = JSON.stringify(objects, null, 2)
              const blob = new Blob([payload], { type: 'application/json' })
              const url = URL.createObjectURL(blob)
              const link = document.createElement('a')
              link.href = url
              link.download = `${workspaceBucket}-${format}.json`
              link.click()
              URL.revokeObjectURL(url)
            }}
          />
          <ul className="hidden">
            {visibleObjects.map((obj) => {
              const key = objectKey(obj)
              return (
                <li key={key} ref={key === selectedObjectKey ? selectedItemRef : undefined}>
                  {obj.Text}
                </li>
              )
            })}
          </ul>
        </div>
      </div>
    </div>
  )
}
