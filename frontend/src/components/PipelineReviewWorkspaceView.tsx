import { useEffect, useMemo, useState } from 'react'
import { Layers, RefreshCw, SlidersHorizontal } from 'lucide-react'
import type { PipelineArtifact, PipelineJob, PipelineReviewWorkspaceState } from '@/types'
import { commitPipelineReviewWorkspace, getPipelineReviewWorkspace, recomputePipelineReviewWorkspace } from '@/lib/api'
import { ReviewCanvasLayers } from '@/components/ReviewCanvasLayers'

type PipelineReviewWorkspaceViewProps = {
  job: PipelineJob
  imageArtifacts: PipelineArtifact[]
  onOpenDetails: () => void
}

function pickBaseImageUrl(imageArtifacts: PipelineArtifact[]): string {
  for (const name of ['stage1_gray.png', 'stage1_gray_equalized.png']) {
    const match = imageArtifacts.find((artifact) => artifact.name === name)
    if (match) return match.url
  }
  const firstNonOverlay = imageArtifacts.find((artifact) => !artifact.name.includes('overlay'))
  return firstNonOverlay?.url ?? imageArtifacts[0]?.url ?? ''
}

const LAYERS = [
  ['equipment', 'Equipment'],
  ['objects', 'Objects'],
  ['ports', 'Ports'],
  ['traces', 'Traced paths'],
  ['branches', 'Branches'],
  ['lineAssociations', 'Line numbers'],
] as const

const LAYER_ARTIFACTS = [
  'stage5_connection_ports.json',
  'stage5b_trace_results.json',
  'stage5b_branch_trace_results.json',
  'stage6_trace_associations.json',
  'stage6_line_number_review.json',
] as const

function layerKeyFromArtifactName(name: string): string {
  return name.replace(/\.json$/i, '')
}

type EditableCollection = 'equipment' | 'objects'
type SelectedEntity = { collection: EditableCollection; id: string }
type RecomputeState = 'idle' | 'scheduled' | 'running' | 'succeeded' | 'failed'

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

function numericValue(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function entityId(entity: Record<string, unknown>, fallback: string): string {
  return String(entity.id ?? entity.Text ?? fallback)
}

function entityBbox(entity: Record<string, unknown>) {
  const bbox = asRecord(entity.bbox) ?? entity
  const xMin = numericValue(bbox.x_min)
  const yMin = numericValue(bbox.y_min)
  const xMax = numericValue(bbox.x_max)
  const yMax = numericValue(bbox.y_max)
  if (xMin === null || yMin === null || xMax === null || yMax === null) {
    return { x_min: 0, y_min: 0, x_max: 0, y_max: 0 }
  }
  return { x_min: xMin, y_min: yMin, x_max: xMax, y_max: yMax }
}

function withBbox(entity: Record<string, unknown>, bbox: { x_min: number; y_min: number; x_max: number; y_max: number }) {
  return { ...entity, bbox }
}

function nextEntityId(prefix: string, items: Array<Record<string, unknown>>): string {
  const used = new Set(items.map((item, index) => entityId(item, `${prefix}_${index + 1}`)))
  for (let index = items.length + 1; index < items.length + 10000; index += 1) {
    const id = `${prefix}_${String(index).padStart(prefix === 'equip' ? 3 : 6, '0')}`
    if (!used.has(id)) return id
  }
  return `${prefix}_${Date.now()}`
}

async function loadLayerPayloads(artifacts: PipelineArtifact[], signal: AbortSignal): Promise<Record<string, Record<string, unknown>>> {
  const entries = await Promise.all(
    LAYER_ARTIFACTS.map(async (name) => {
      const artifact = artifacts.find((item) => item.name === name)
      if (!artifact) return null
      const response = await fetch(artifact.url, { signal })
      if (!response.ok) return null
      const payload = await response.json()
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return null
      return [layerKeyFromArtifactName(name), payload as Record<string, unknown>] as const
    })
  )
  return Object.fromEntries(entries.filter((entry): entry is readonly [string, Record<string, unknown>] => entry !== null))
}

export function PipelineReviewWorkspaceView({ job, imageArtifacts, onOpenDetails }: PipelineReviewWorkspaceViewProps) {
  const [workspace, setWorkspace] = useState<PipelineReviewWorkspaceState | null>(null)
  const [layerPayloads, setLayerPayloads] = useState<Record<string, Record<string, unknown>>>({})
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null)
  const [selectedEntity, setSelectedEntity] = useState<SelectedEntity | null>(null)
  const [isDirty, setIsDirty] = useState(false)
  const [recomputeState, setRecomputeState] = useState<RecomputeState>('idle')
  const [isCommitting, setIsCommitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isLoadingLayers, setIsLoadingLayers] = useState(true)
  const [visibleLayers, setVisibleLayers] = useState<Set<string>>(() => new Set(LAYERS.map(([key]) => key)))

  const imageUrl = useMemo(() => pickBaseImageUrl(imageArtifacts), [imageArtifacts])

  useEffect(() => {
    let active = true
    setIsLoading(true)
    setError(null)
    void getPipelineReviewWorkspace(job.job_id)
      .then((payload) => {
        if (!active) return
        setWorkspace(payload.workspace)
      })
      .catch((loadError) => {
        if (!active) return
        setError(loadError instanceof Error ? loadError.message : 'Failed to load review workspace')
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [job.job_id])

  useEffect(() => {
    const controller = new AbortController()
    setIsLoadingLayers(true)
    void loadLayerPayloads(job.artifacts, controller.signal)
      .then((payloads) => {
        setLayerPayloads(payloads)
      })
      .catch((loadError) => {
        if (controller.signal.aborted) return
        setError(loadError instanceof Error ? loadError.message : 'Failed to load review layers')
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoadingLayers(false)
      })
    return () => {
      controller.abort()
    }
  }, [job.artifacts])

  const loadedLayerCount = Object.keys(layerPayloads).length

  const selectedRecord = useMemo(() => {
    if (!workspace || !selectedEntity) return null
    return workspace[selectedEntity.collection].find((item, index) => entityId(item, `${selectedEntity.collection}_${index}`) === selectedEntity.id) ?? null
  }, [selectedEntity, workspace])

  const updateWorkspaceEntity = (collection: EditableCollection, id: string, updater: (entity: Record<string, unknown>) => Record<string, unknown>) => {
    setWorkspace((current) => {
      if (!current) return current
      return {
        ...current,
        [collection]: current[collection].map((item, index) => {
          const currentId = entityId(item, `${collection}_${index}`)
          return currentId === id ? updater(item) : item
        }),
      }
    })
    setIsDirty(true)
    setRecomputeState('scheduled')
  }

  const updateSelectedBbox = (field: 'x_min' | 'y_min' | 'x_max' | 'y_max', value: number) => {
    if (!selectedEntity) return
    updateWorkspaceEntity(selectedEntity.collection, selectedEntity.id, (entity) => {
      const bbox = entityBbox(entity)
      return withBbox(entity, { ...bbox, [field]: value })
    })
  }

  const updateSelectedText = (field: 'id' | 'class_name' | 'Object', value: string) => {
    if (!selectedEntity) return
    const currentSelection = selectedEntity
    updateWorkspaceEntity(currentSelection.collection, currentSelection.id, (entity) => ({ ...entity, [field]: value }))
    if (field === 'id' && value.trim()) {
      setSelectedEntity({ ...currentSelection, id: value.trim() })
    }
  }

  const addEntity = (collection: EditableCollection) => {
    if (!workspace) return
    const prefix = collection === 'equipment' ? 'equip' : 'obj'
    const id = nextEntityId(prefix, workspace[collection])
    const width = 160
    const height = 110
    const x = Math.max(0, Math.round((imageSize?.width ?? 900) / 2 - width / 2))
    const y = Math.max(0, Math.round((imageSize?.height ?? 700) / 2 - height / 2))
    const entity = {
      id,
      class_name: collection === 'equipment' ? 'vessel' : 'object',
      bbox: { x_min: x, y_min: y, x_max: x + width, y_max: y + height },
      source: 'hitl',
      review_state: 'accepted',
    }
    setWorkspace((current) => current ? { ...current, [collection]: [...current[collection], entity] } : current)
    setSelectedEntity({ collection, id })
    setIsDirty(true)
    setRecomputeState('scheduled')
  }

  const rejectSelected = () => {
    if (!selectedEntity) return
    updateWorkspaceEntity(selectedEntity.collection, selectedEntity.id, (entity) => ({ ...entity, review_state: 'rejected' }))
  }

  const runRecompute = async (workspaceToRecompute = workspace) => {
    if (!workspaceToRecompute) return
    setRecomputeState('running')
    setError(null)
    try {
      const response = await recomputePipelineReviewWorkspace(job.job_id, workspaceToRecompute)
      setWorkspace(response.workspace)
      setLayerPayloads(response.layers)
      setIsDirty(false)
      setRecomputeState('succeeded')
    } catch (recomputeError) {
      setRecomputeState('failed')
      setError(recomputeError instanceof Error ? recomputeError.message : 'Failed to recompute reviewed layers')
    }
  }

  const commitWorkspace = async () => {
    if (!workspace) return
    setIsCommitting(true)
    setError(null)
    try {
      const response = await commitPipelineReviewWorkspace(job.job_id, workspace)
      setWorkspace(response.workspace)
      setIsDirty(false)
    } catch (commitError) {
      setError(commitError instanceof Error ? commitError.message : 'Failed to commit reviewed workspace')
    } finally {
      setIsCommitting(false)
    }
  }

  useEffect(() => {
    if (!isDirty || !workspace) return
    const timeoutId = window.setTimeout(() => {
      void runRecompute(workspace)
    }, 500)
    return () => window.clearTimeout(timeoutId)
  }, [isDirty, workspace])

  const toggleLayer = (key: string) => {
    setVisibleLayers((current) => {
      const next = new Set(current)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  return (
    <div className="flex h-full flex-col bg-[#f4f1e8] text-[#1f2a24]">
      <div className="flex items-center justify-between border-b border-[#d6cdb8] bg-[#fffaf0] px-5 py-3">
        <div>
          <div className="text-lg font-semibold tracking-tight">Pipeline Review Workspace</div>
          <div className="text-xs text-[#6f6757]">
            Edit objects, equipment, ports, traced paths, and Stage 6 associations in one review surface.
          </div>
          {error ? <div className="mt-1 text-xs text-red-700">{error}</div> : null}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={!workspace || recomputeState === 'running'}
            onClick={() => {
              void runRecompute()
            }}
            className="inline-flex items-center gap-2 rounded-lg border border-[#d6cdb8] bg-[#f6ead2] px-3 py-2 text-sm font-semibold text-[#8a6f25] disabled:opacity-60"
          >
            <RefreshCw className={`h-4 w-4 ${recomputeState === 'running' ? 'animate-spin' : ''}`} />
            Recompute now
          </button>
          <button
            type="button"
            onClick={onOpenDetails}
            className="rounded-lg border border-[#1f2a24] bg-[#1f2a24] px-3 py-2 text-sm font-semibold text-[#fffaf0]"
          >
            Artifacts / QA
          </button>
          <button
            type="button"
            disabled={!workspace || recomputeState === 'running' || isCommitting}
            onClick={() => {
              void commitWorkspace()
            }}
            className="rounded-lg border border-[#2f6f4e] bg-[#2f6f4e] px-3 py-2 text-sm font-semibold text-white disabled:opacity-60"
          >
            {isCommitting ? 'Committing...' : 'Commit review'}
          </button>
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[260px_minmax(0,1fr)_340px]">
        <aside className="border-r border-[#d6cdb8] bg-[#fbf4e4] p-4">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Layers className="h-4 w-4" />
            Layers
          </div>
          <div className="mt-4 space-y-2">
            {LAYERS.map(([key, label]) => (
              <label key={key} className="flex cursor-pointer items-center justify-between rounded-xl border border-[#e2d8c2] bg-[#fffaf0] px-3 py-2 text-sm">
                <span>{label}</span>
                <input
                  type="checkbox"
                  checked={visibleLayers.has(key)}
                  onChange={() => toggleLayer(key)}
                  className="h-4 w-4 accent-[#2f6f4e]"
                />
              </label>
            ))}
          </div>
          <div className="mt-5 rounded-xl border border-[#e2d8c2] bg-[#fffaf0] p-3 text-xs text-[#6f6757]">
            {isLoading ? 'Loading review workspace...' : `${workspace?.objects.length ?? 0} objects, ${workspace?.equipment.length ?? 0} equipment boxes`}
            <div className="mt-2">
              {isLoadingLayers ? 'Loading rendered layers...' : `${loadedLayerCount} JSON layers loaded`}
            </div>
          </div>
        </aside>

        <main className="relative min-h-0 overflow-auto bg-[#e8e1d1] p-5">
          <div className="relative mx-auto inline-block min-w-[720px] rounded-2xl border border-[#c8baa0] bg-white p-2 shadow-[0_24px_80px_rgba(71,55,26,0.20)]">
            {imageUrl ? (
              <>
                <img
                  src={imageUrl}
                  alt="Pipeline review base drawing"
                  className="block max-h-[calc(100vh-150px)] max-w-none select-none rounded-xl"
                  onLoad={(event) => {
                    const image = event.currentTarget
                    setImageSize({ width: image.naturalWidth, height: image.naturalHeight })
                  }}
                />
                <ReviewCanvasLayers
                  workspace={workspace}
                  layers={layerPayloads}
                  visibleLayers={visibleLayers}
                  imageSize={imageSize}
                  selectedEntity={selectedEntity}
                  onSelectEntity={setSelectedEntity}
                />
              </>
            ) : (
              <div className="flex h-[520px] w-[760px] items-center justify-center rounded-xl bg-[#fffaf0] text-sm text-[#6f6757]">
                No image artifact available.
              </div>
            )}
            <div className="pointer-events-none absolute left-4 top-4 rounded-full border border-[#d6cdb8] bg-[#fffaf0]/90 px-3 py-1 text-xs font-semibold text-[#4a4030]">
              {imageSize ? `${imageSize.width} x ${imageSize.height}px` : 'Loading image'}
            </div>
          </div>
        </main>

        <aside className="border-l border-[#d6cdb8] bg-[#fffaf0] p-4">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <SlidersHorizontal className="h-4 w-4" />
            Inspector
          </div>
          <div className="mt-4 rounded-xl border border-[#e2d8c2] bg-[#fbf4e4] p-4 text-sm text-[#6f6757]">
            <div className="flex gap-2">
              <button type="button" onClick={() => addEntity('equipment')} className="rounded-lg bg-[#2f6f4e] px-3 py-2 text-xs font-semibold text-white">
                Add equipment
              </button>
              <button type="button" onClick={() => addEntity('objects')} className="rounded-lg bg-[#8a5a12] px-3 py-2 text-xs font-semibold text-white">
                Add object
              </button>
            </div>
            <div className="mt-3 text-xs">
              Recompute: <span className="font-semibold">{recomputeState}</span>
              {isDirty ? <span className="ml-2 rounded-full bg-amber-200 px-2 py-0.5 text-amber-900">dirty</span> : null}
            </div>
          </div>
          {selectedEntity && selectedRecord ? (
            <div className="mt-4 rounded-xl border border-[#e2d8c2] bg-[#fbf4e4] p-4">
              <div className="text-xs uppercase tracking-wide text-[#6f6757]">Selected {selectedEntity.collection}</div>
              <label className="mt-3 block text-xs font-semibold text-[#4a4030]">
                ID
                <input
                  value={String(selectedRecord.id ?? selectedEntity.id)}
                  onChange={(event) => updateSelectedText('id', event.target.value)}
                  className="mt-1 w-full rounded-lg border border-[#d6cdb8] bg-white px-2 py-1 font-mono text-xs"
                />
              </label>
              <label className="mt-3 block text-xs font-semibold text-[#4a4030]">
                Class
                <input
                  value={String(selectedRecord.class_name ?? selectedRecord.Object ?? '')}
                  onChange={(event) => updateSelectedText(selectedEntity.collection === 'objects' ? 'Object' : 'class_name', event.target.value)}
                  className="mt-1 w-full rounded-lg border border-[#d6cdb8] bg-white px-2 py-1 text-xs"
                />
              </label>
              <div className="mt-3 grid grid-cols-2 gap-2">
                {(['x_min', 'y_min', 'x_max', 'y_max'] as const).map((field) => {
                  const bbox = entityBbox(selectedRecord)
                  return (
                    <label key={field} className="text-xs font-semibold text-[#4a4030]">
                      {field}
                      <input
                        type="number"
                        value={bbox[field]}
                        onChange={(event) => updateSelectedBbox(field, Number(event.target.value))}
                        className="mt-1 w-full rounded-lg border border-[#d6cdb8] bg-white px-2 py-1 font-mono text-xs"
                      />
                    </label>
                  )
                })}
              </div>
              <button
                type="button"
                onClick={rejectSelected}
                className="mt-4 w-full rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs font-semibold text-red-700"
              >
                Reject / remove from pipeline
              </button>
            </div>
          ) : (
            <div className="mt-4 rounded-xl border border-[#e2d8c2] bg-[#fbf4e4] p-4 text-sm text-[#6f6757]">
              Select an equipment or object box to edit its reviewed ID, class, or bbox.
            </div>
          )}
          <div className="mt-4 rounded-xl border border-[#e2d8c2] bg-[#fbf4e4] p-4">
            <div className="text-xs uppercase tracking-wide text-[#6f6757]">Job</div>
            <div className="mt-1 break-all font-mono text-xs">{job.job_id}</div>
            <div className="mt-3 text-xs uppercase tracking-wide text-[#6f6757]">Status</div>
            <div className="mt-1 text-sm font-semibold capitalize">{job.status}</div>
          </div>
        </aside>
      </div>
    </div>
  )
}
