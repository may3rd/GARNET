import { useEffect, useMemo, useState } from 'react'
import { Check, Layers, RefreshCw } from 'lucide-react'
import type { DetectedObject, PipelineArtifact, PipelineJob, PipelineManualPort, PipelineReviewWorkspaceState } from '@/types'
import { commitPipelineReviewWorkspace, getPipelineReviewWorkspace, recomputePipelineReviewWorkspace } from '@/lib/api'
import { ReviewCanvasLayers } from '@/components/ReviewCanvasLayers'
import { CanvasView } from '@/components/CanvasView'
import { ObjectSidebar } from '@/components/ObjectSidebar'
import { objectKey } from '@/lib/objectKey'
import type { ExportFormat } from '@/lib/exportFormats'

type PipelineReviewWorkspaceViewProps = {
  job: PipelineJob
  imageArtifacts: PipelineArtifact[]
  onOpenDetails: () => void
  onCommitComplete?: () => void
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

function pickPipeMaskUrl(imageArtifacts: PipelineArtifact[]): string | null {
  return imageArtifacts.find((artifact) => artifact.name === 'stage5_pipe_mask.png')?.url ?? null
}

type EditableCollection = 'equipment' | 'objects'
type SelectedEntity = { collection: EditableCollection; id: string }
type RecomputeState = 'idle' | 'scheduled' | 'running' | 'succeeded' | 'failed'
type WorkspaceDraft = {
  Object: string
  Left: number
  Top: number
  Width: number
  Height: number
  Text: string
}
type BBox = { x_min: number; y_min: number; x_max: number; y_max: number }
type PortCandidate = Pick<PipelineManualPort, 'x' | 'y' | 'direction'>

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

function isEquipmentCollection(entity: SelectedEntity | null): entity is SelectedEntity {
  return entity?.collection === 'equipment'
}

function makePortId(ownerId: string, ports: PipelineManualPort[]): string {
  const prefix = `${ownerId}:port_`
  const used = new Set(ports.map((port) => port.port_id))
  for (let index = 1; index < 1000; index += 1) {
    const id = `${prefix}${String(index).padStart(2, '0')}`
    if (!used.has(id)) return id
  }
  return `${prefix}${Date.now()}`
}

function normalizeManualPort(value: Record<string, unknown>): PipelineManualPort | null {
  const x = numericValue(value.x)
  const y = numericValue(value.y)
  const direction = String(value.direction ?? '').toUpperCase()
  if (x === null || y === null || !['UP', 'DOWN', 'LEFT', 'RIGHT'].includes(direction)) return null
  const ownerId = String(value.owner_id ?? value.source_obj_id ?? value.object_id ?? '')
  if (!ownerId) return null
  return {
    port_id: String(value.port_id ?? value.id ?? `${ownerId}:port`),
    owner_id: ownerId,
    owner_type: String(value.owner_type ?? 'equipment'),
    x,
    y,
    direction: direction as PipelineManualPort['direction'],
    source: typeof value.source === 'string' ? value.source : undefined,
    review_state: typeof value.review_state === 'string' ? value.review_state : undefined,
  }
}

function portMatchesCandidate(port: PipelineManualPort, ownerId: string, candidate: PortCandidate, tolerance = 8) {
  return (
    port.owner_id === ownerId
    && port.review_state !== 'rejected'
    && port.direction === candidate.direction
    && Math.abs(port.x - candidate.x) <= tolerance
    && Math.abs(port.y - candidate.y) <= tolerance
  )
}

async function loadImagePixels(url: string): Promise<{ width: number; height: number; data: Uint8ClampedArray }> {
  const image = new Image()
  image.crossOrigin = 'anonymous'
  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve()
    image.onerror = () => reject(new Error('Failed to load image for port detection'))
    image.src = url
  })
  const canvas = document.createElement('canvas')
  canvas.width = image.naturalWidth
  canvas.height = image.naturalHeight
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Cannot create image canvas for port detection')
  ctx.drawImage(image, 0, 0)
  return {
    width: canvas.width,
    height: canvas.height,
    data: ctx.getImageData(0, 0, canvas.width, canvas.height).data,
  }
}

function pixelBrightness(pixels: { width: number; height: number; data: Uint8ClampedArray }, x: number, y: number): number {
  const px = Math.max(0, Math.min(pixels.width - 1, Math.round(x)))
  const py = Math.max(0, Math.min(pixels.height - 1, Math.round(y)))
  const index = (py * pixels.width + px) * 4
  return (pixels.data[index] + pixels.data[index + 1] + pixels.data[index + 2]) / 3
}

function detectPortCandidatesFromImage(
  pixels: { width: number; height: number; data: Uint8ClampedArray },
  bbox: BBox,
  maskMode: boolean,
): PortCandidate[] {
  const isPipePixel = (x: number, y: number) => {
    const brightness = pixelBrightness(pixels, x, y)
    return maskMode ? brightness > 24 : brightness < 170
  }
  const hasOutwardRun = (x: number, y: number, dx: number, dy: number) => {
    let hits = 0
    for (let offset = 1; offset <= 16; offset += 1) {
      if (isPipePixel(x + dx * offset, y + dy * offset)) hits += 1
    }
    return hits >= 3
  }
  const edges = [
    { direction: 'UP' as const, start: bbox.x_min, end: bbox.x_max, fixed: bbox.y_min, axis: 'x' as const, dx: 0, dy: -1 },
    { direction: 'DOWN' as const, start: bbox.x_min, end: bbox.x_max, fixed: bbox.y_max, axis: 'x' as const, dx: 0, dy: 1 },
    { direction: 'LEFT' as const, start: bbox.y_min, end: bbox.y_max, fixed: bbox.x_min, axis: 'y' as const, dx: -1, dy: 0 },
    { direction: 'RIGHT' as const, start: bbox.y_min, end: bbox.y_max, fixed: bbox.x_max, axis: 'y' as const, dx: 1, dy: 0 },
  ]
  const candidates: PortCandidate[] = []
  for (const edge of edges) {
    let runStart: number | null = null
    for (let position = Math.round(edge.start); position <= Math.round(edge.end); position += 1) {
      const x = edge.axis === 'x' ? position : edge.fixed
      const y = edge.axis === 'x' ? edge.fixed : position
      const active = isPipePixel(x, y) && hasOutwardRun(x, y, edge.dx, edge.dy)
      if (active && runStart === null) {
        runStart = position
      }
      if ((!active || position === Math.round(edge.end)) && runStart !== null) {
        const runEnd = active && position === Math.round(edge.end) ? position : position - 1
        if (runEnd - runStart >= 2) {
          const center = Math.round((runStart + runEnd) / 2)
          candidates.push({
            x: edge.axis === 'x' ? center : edge.fixed,
            y: edge.axis === 'x' ? edge.fixed : center,
            direction: edge.direction,
          })
        }
        runStart = null
      }
    }
  }
  return candidates
}

function entityClassName(entity: Record<string, unknown>, fallback: string): string {
  return String(entity.class_name ?? entity.Object ?? fallback)
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

export function PipelineReviewWorkspaceView({ job, imageArtifacts, onOpenDetails, onCommitComplete }: PipelineReviewWorkspaceViewProps) {
  const [workspace, setWorkspace] = useState<PipelineReviewWorkspaceState | null>(null)
  const [layerPayloads, setLayerPayloads] = useState<Record<string, Record<string, unknown>>>({})
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null)
  const [selectedEntity, setSelectedEntity] = useState<SelectedEntity | null>(null)
  const [selectedObjectKey, setSelectedObjectKey] = useState<string | null>(null)
  const [selectedPortId, setSelectedPortId] = useState<string | null>(null)
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null)
  const [selectedBranchId, setSelectedBranchId] = useState<string | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [editDraft, setEditDraft] = useState<WorkspaceDraft | null>(null)
  const [isCreating, setIsCreating] = useState(false)
  const [createDraft, setCreateDraft] = useState<WorkspaceDraft | null>(null)
  const [createCollection, setCreateCollection] = useState<EditableCollection>('objects')
  const [isDirty, setIsDirty] = useState(false)
  const [recomputeState, setRecomputeState] = useState<RecomputeState>('idle')
  const [isCommitting, setIsCommitting] = useState(false)
  const [hiddenClasses, setHiddenClasses] = useState<Set<string>>(new Set())
  const [confidenceFilter, setConfidenceFilter] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isLoadingLayers, setIsLoadingLayers] = useState(true)
  const [visibleLayers, setVisibleLayers] = useState<Set<string>>(() => new Set(LAYERS.map(([key]) => key)))

  const imageUrl = useMemo(() => pickBaseImageUrl(imageArtifacts), [imageArtifacts])
  const pipeMaskUrl = useMemo(() => pickPipeMaskUrl(imageArtifacts), [imageArtifacts])

  useEffect(() => {
    if (!imageUrl) {
      setImageSize(null)
      return
    }
    let active = true
    const image = new Image()
    image.onload = () => {
      if (!active) return
      setImageSize({ width: image.naturalWidth, height: image.naturalHeight })
    }
    image.src = imageUrl
    return () => {
      active = false
    }
  }, [imageUrl])

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
  const manualPorts = useMemo(
    () => (workspace?.manual_ports ?? []).flatMap((item) => {
      const port = normalizeManualPort(item)
      return port ? [port] : []
    }),
    [workspace]
  )

  const { canvasObjects, entityByObjectKey } = useMemo(() => {
    const nextObjects: DetectedObject[] = []
    const nextMap = new Map<string, SelectedEntity>()
    const append = (collection: EditableCollection, items: Array<Record<string, unknown>>, categoryId: number) => {
      items.forEach((item, index) => {
        const bbox = entityBbox(item)
        const id = entityId(item, `${collection}_${index + 1}`)
        const obj: DetectedObject = {
          Index: index + 1,
          Object: entityClassName(item, collection === 'equipment' ? 'equipment' : 'object'),
          CategoryID: categoryId,
          ObjectID: index + 1,
          Left: bbox.x_min,
          Top: bbox.y_min,
          Width: Math.max(1, bbox.x_max - bbox.x_min),
          Height: Math.max(1, bbox.y_max - bbox.y_min),
          Score: 1,
          Text: id,
          ReviewStatus: item.review_state === 'rejected' ? 'rejected' : 'accepted',
        }
        nextObjects.push(obj)
        nextMap.set(objectKey(obj), { collection, id })
      })
    }
    append('equipment', workspace?.equipment ?? [], 3)
    append('objects', workspace?.objects ?? [], 4)
    return { canvasObjects: nextObjects, entityByObjectKey: nextMap }
  }, [workspace])

  const selectedCanvasObject = useMemo(
    () => canvasObjects.find((item) => objectKey(item) === selectedObjectKey) ?? null,
    [canvasObjects, selectedObjectKey]
  )

  const selectedEquipmentRecord = useMemo(() => {
    if (!workspace || !isEquipmentCollection(selectedEntity)) return null
    return workspace.equipment.find((item, index) => entityId(item, `equipment_${index + 1}`) === selectedEntity.id) ?? null
  }, [selectedEntity, workspace])

  const selectedEquipmentPorts = useMemo(
    () => manualPorts.filter((port) => port.owner_id === selectedEntity?.id && port.review_state !== 'rejected'),
    [manualPorts, selectedEntity]
  )

  const visibleCanvasObjects = useMemo(
    () =>
      canvasObjects.filter((obj) => {
        const classKey = obj.Object.toLowerCase().replace(/_/g, ' ').trim()
        return obj.Score >= confidenceFilter && !hiddenClasses.has(classKey)
      }),
    [canvasObjects, confidenceFilter, hiddenClasses]
  )

  const reviewStatus = useMemo(() => {
    const status: Record<string, 'accepted' | 'rejected'> = {}
    canvasObjects.forEach((item) => {
      status[objectKey(item)] = item.ReviewStatus === 'rejected' ? 'rejected' : 'accepted'
    })
    return status
  }, [canvasObjects])

  const updateWorkspaceEntity = (collection: EditableCollection, id: string, updater: (entity: Record<string, unknown>) => Record<string, unknown>) => {
    setWorkspace((current) => {
      if (!current) return current
      return {
        ...current,
        [collection]: current[collection].map((item, index) => {
          const currentId = entityId(item, `${collection}_${index + 1}`)
          return currentId === id ? updater(item) : item
        }),
      }
    })
    setIsDirty(true)
    setRecomputeState('scheduled')
  }

  const addEntity = (collection: EditableCollection, className?: string) => {
    const width = 160
    const height = 110
    const x = Math.max(0, Math.round((imageSize?.width ?? 900) / 2 - width / 2))
    const y = Math.max(0, Math.round((imageSize?.height ?? 700) / 2 - height / 2))
    setCreateCollection(collection)
    setIsCreating(true)
    setCreateDraft({
      Object: className ?? (collection === 'equipment' ? 'vessel' : 'object'),
      Left: x,
      Top: y,
      Width: width,
      Height: height,
      Text: '',
    })
    setSelectedEntity(null)
    setSelectedPortId(null)
    setSelectedTraceId(null)
    setSelectedBranchId(null)
    setSelectedObjectKey(null)
    setIsEditing(false)
    setEditDraft(null)
  }

  const rejectSelected = () => {
    if (!selectedEntity) return
    updateWorkspaceEntity(selectedEntity.collection, selectedEntity.id, (entity) => ({ ...entity, review_state: 'rejected' }))
  }

  const selectCanvasObject = (key: string | null) => {
    setSelectedObjectKey(key)
    setSelectedPortId(null)
    setSelectedTraceId(null)
    setSelectedBranchId(null)
    setIsEditing(false)
    setEditDraft(null)
    if (!key) {
      setSelectedEntity(null)
      return
    }
    setSelectedEntity(entityByObjectKey.get(key) ?? null)
  }

  const setCanvasReviewStatus = (key: string, status: 'accepted' | 'rejected' | null) => {
    const entity = entityByObjectKey.get(key)
    if (!entity) return
    updateWorkspaceEntity(entity.collection, entity.id, (current) => ({ ...current, review_state: status ?? 'accepted' }))
  }

  const startCanvasEdit = (obj: DetectedObject) => {
    selectCanvasObject(objectKey(obj))
    setIsEditing(true)
    setEditDraft({
      Object: obj.Object,
      Left: obj.Left,
      Top: obj.Top,
      Width: obj.Width,
      Height: obj.Height,
      Text: obj.Text,
    })
  }

  const saveCanvasEdit = () => {
    if (!selectedEntity || !editDraft) return
    const nextId = editDraft.Text.trim() || selectedEntity.id
    updateWorkspaceEntity(selectedEntity.collection, selectedEntity.id, (entity) => ({
      ...entity,
      id: nextId,
      class_name: editDraft.Object,
      Object: editDraft.Object,
      bbox: {
        x_min: Math.round(editDraft.Left),
        y_min: Math.round(editDraft.Top),
        x_max: Math.round(editDraft.Left + editDraft.Width),
        y_max: Math.round(editDraft.Top + editDraft.Height),
      },
    }))
    setSelectedEntity({ ...selectedEntity, id: nextId })
    setIsEditing(false)
    setEditDraft(null)
  }

  const saveCanvasCreate = () => {
    if (!workspace || !createDraft) return
    const prefix = createCollection === 'equipment' ? 'equip' : 'obj'
    const id = createDraft.Text.trim() || nextEntityId(prefix, workspace[createCollection])
    const entity = {
      id,
      class_name: createDraft.Object || (createCollection === 'equipment' ? 'vessel' : 'object'),
      bbox: {
        x_min: Math.round(createDraft.Left),
        y_min: Math.round(createDraft.Top),
        x_max: Math.round(createDraft.Left + createDraft.Width),
        y_max: Math.round(createDraft.Top + createDraft.Height),
      },
      source: 'hitl',
      review_state: 'accepted',
    }
    setWorkspace((current) => current ? { ...current, [createCollection]: [...current[createCollection], entity] } : current)
    setSelectedEntity({ collection: createCollection, id })
    setIsCreating(false)
    setCreateDraft(null)
    setIsDirty(true)
    setRecomputeState('scheduled')
    if (createCollection === 'equipment') {
      void addPortsFromBorderCrossings(id, entity.bbox)
    }
  }

  const deleteCanvasSelected = () => {
    rejectSelected()
  }

  const navigateCanvas = (direction: -1 | 1) => {
    if (!selectedObjectKey || canvasObjects.length === 0) return
    const keys = canvasObjects.map((item) => objectKey(item))
    const index = keys.indexOf(selectedObjectKey)
    const nextIndex = index + direction
    if (nextIndex < 0 || nextIndex >= keys.length) return
    selectCanvasObject(keys[nextIndex])
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
      onCommitComplete?.()
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

  const appendManualPorts = (ownerId: string, candidates: PortCandidate[]) => {
    if (!candidates.length) return 0
    let addedCount = 0
    setWorkspace((current) => {
      if (!current) return current
      const existing = current.manual_ports.flatMap((item) => {
        const port = normalizeManualPort(item)
        return port ? [port] : []
      })
      const nextPorts = [...current.manual_ports]
      for (const candidate of candidates) {
        if (existing.some((port) => portMatchesCandidate(port, ownerId, candidate))) {
          continue
        }
        const port: PipelineManualPort = {
          port_id: makePortId(ownerId, [...existing, ...nextPorts.flatMap((item) => {
            const normalized = normalizeManualPort(item)
            return normalized ? [normalized] : []
          })]),
          owner_id: ownerId,
          owner_type: 'equipment',
          x: Math.round(candidate.x),
          y: Math.round(candidate.y),
          direction: candidate.direction,
          source: 'hitl_border_crossing',
          review_state: 'accepted',
        }
        existing.push(port)
        nextPorts.push(port)
        addedCount += 1
      }
      return { ...current, manual_ports: nextPorts }
    })
    if (addedCount > 0) {
      setIsDirty(true)
      setRecomputeState('scheduled')
    }
    return addedCount
  }

  const addPortsFromBorderCrossings = async (ownerId = selectedEntity?.id, bbox = selectedEquipmentRecord ? entityBbox(selectedEquipmentRecord) : null) => {
    if (!ownerId || !bbox) return
    const detectionUrl = pipeMaskUrl ?? imageUrl
    if (!detectionUrl) return
    try {
      const pixels = await loadImagePixels(detectionUrl)
      const candidates = detectPortCandidatesFromImage(pixels, bbox, Boolean(pipeMaskUrl))
      const addedCount = appendManualPorts(ownerId, candidates)
      if (addedCount === 0) {
        setError('No new equipment ports found on pipe crossings at the selected box border.')
      } else {
        setError(null)
      }
    } catch (portError) {
      setError(portError instanceof Error ? portError.message : 'Failed to detect equipment ports')
    }
  }

  const rejectPort = (portId: string) => {
    setWorkspace((current) => {
      if (!current) return current
      return {
        ...current,
        manual_ports: current.manual_ports.map((item) =>
          String(item.port_id ?? item.id) === portId
            ? { ...item, review_state: 'rejected' }
            : item
        ),
      }
    })
    setSelectedPortId((current) => current === portId ? null : current)
    setIsDirty(true)
    setRecomputeState('scheduled')
  }

  const rejectTracePath = (kind: 'trace' | 'branch', id: string) => {
    setWorkspace((current) => {
      if (!current) return current
      const remaining = current.trace_overrides.filter((item) => {
        const targetKind = String(item.target_type ?? item.kind ?? '')
        const targetId = String(item.target_id ?? item.id ?? '')
        return !(targetKind === kind && targetId === id)
      })
      return {
        ...current,
        trace_overrides: [
          ...remaining,
          {
            target_type: kind,
            target_id: id,
            review_state: 'rejected',
            source: 'hitl',
          },
        ],
      }
    })
    if (kind === 'trace') setSelectedTraceId(null)
    else setSelectedBranchId(null)
    setIsDirty(true)
    setRecomputeState('scheduled')
  }

  const updateCreateDraft = (field: 'Object' | 'Left' | 'Top' | 'Width' | 'Height' | 'Text', value: string) => {
    setCreateDraft((current) => {
      if (!current) return current
      if (field === 'Object' || field === 'Text') {
        return { ...current, [field]: value }
      }
      const parsed = Number(value)
      return { ...current, [field]: Number.isFinite(parsed) ? parsed : current[field] }
    })
  }

  const exportWorkspaceObjects = (format: ExportFormat, filter: 'all' | 'accepted' | 'rejected' | 'visible') => {
    const source = filter === 'visible'
      ? visibleCanvasObjects
      : canvasObjects.filter((obj) => {
          const status = reviewStatus[objectKey(obj)]
          if (filter === 'accepted') return status === 'accepted'
          if (filter === 'rejected') return status === 'rejected'
          return true
        })
    const payload = JSON.stringify(source, null, 2)
    const blob = new Blob([payload], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `stage5b-review-${filter}-${format}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-[var(--bg-primary)] text-[var(--text-primary)]">
      <div className="shrink-0 flex items-center justify-between border-b border-[var(--border-muted)] bg-[var(--bg-secondary)] px-6 py-4">
        <div>
          <div className="text-lg font-semibold">Pipeline HITL Review</div>
          <div className="text-xs text-[var(--text-secondary)]">
            Stage 5b traced-path review using the detection-mode layout.
          </div>
          {error ? <div className="mt-1 text-xs text-[var(--danger)]">{error}</div> : null}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={!workspace || recomputeState === 'running'}
            onClick={() => {
              void runRecompute()
            }}
            className="inline-flex items-center gap-2 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm font-semibold text-[var(--text-primary)] disabled:opacity-40"
          >
            <RefreshCw className={`h-4 w-4 ${recomputeState === 'running' ? 'animate-spin' : ''}`} />
            Recompute now
          </button>
          <button
            type="button"
            onClick={onOpenDetails}
            className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm font-semibold text-[var(--text-primary)]"
          >
            Summary / artifacts
          </button>
          <button
            type="button"
            disabled={!workspace || isDirty || recomputeState === 'scheduled' || recomputeState === 'running' || isCommitting}
            onClick={() => {
              void commitWorkspace()
            }}
            className="inline-flex items-center gap-2 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-sm font-semibold text-emerald-700 disabled:opacity-40"
          >
            <Check className="h-4 w-4" />
            {isCommitting ? 'Committing...' : isDirty || recomputeState === 'scheduled' ? 'Waiting for recompute' : 'Commit review & continue'}
          </button>
        </div>
      </div>

      <div className="shrink-0 border-b border-[var(--border-muted)] bg-[var(--bg-secondary)] px-6 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => addEntity('equipment')}
            disabled={isCreating}
            className="rounded-full border border-[var(--accent)] bg-[var(--accent)]/10 px-3 py-1 text-xs font-semibold text-[var(--accent)] disabled:opacity-40"
          >
            Add equipment
          </button>
          <button
            type="button"
            onClick={() => addEntity('objects', 'connection')}
            disabled={isCreating}
            className="rounded-full border border-[var(--accent)] bg-[var(--accent)]/10 px-3 py-1 text-xs font-semibold text-[var(--accent)] disabled:opacity-40"
          >
            Add connection
          </button>
          <button
            type="button"
            onClick={() => addEntity('objects')}
            disabled={isCreating}
            className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-1 text-xs font-semibold text-[var(--text-secondary)] disabled:opacity-40"
          >
            Add object
          </button>
          <div className="mx-2 h-5 w-px bg-[var(--border-muted)]" />
          <button
            type="button"
            onClick={() => {
              void addPortsFromBorderCrossings()
            }}
            disabled={!selectedEquipmentRecord}
            className="rounded-full border border-cyan-500/40 bg-cyan-500/10 px-3 py-1 text-xs font-semibold text-cyan-700 disabled:opacity-40"
          >
            Add ports from border crossings
          </button>
          {selectedEquipmentRecord ? (
            <div className="flex flex-wrap items-center gap-1">
              {selectedEquipmentPorts.length ? selectedEquipmentPorts.map((port) => (
                <button
                  key={port.port_id}
                  type="button"
                  onClick={() => setSelectedPortId((current) => current === port.port_id ? null : port.port_id)}
                  className={`rounded-full border px-2 py-1 font-mono text-[11px] font-semibold ${
                    selectedPortId === port.port_id
                      ? 'border-red-500 bg-red-500/10 text-red-700'
                      : 'border-cyan-500/30 bg-[var(--bg-primary)] text-cyan-700'
                  }`}
                  title={`${port.direction} (${Math.round(port.x)}, ${Math.round(port.y)})`}
                >
                  {port.port_id.split(':').at(-1)}:{port.direction}
                </button>
              )) : (
                <span className="text-xs text-[var(--text-secondary)]">No reviewed ports for selected equipment</span>
              )}
              {selectedPortId ? (
                <button
                  type="button"
                  onClick={() => rejectPort(selectedPortId)}
                  className="rounded-full border border-red-500/40 bg-red-500/10 px-3 py-1 text-xs font-semibold text-red-700"
                >
                  Remove selected port
                </button>
              ) : null}
            </div>
          ) : null}
          <div className="mx-2 h-5 w-px bg-[var(--border-muted)]" />
          {selectedTraceId ? (
            <button
              type="button"
              onClick={() => rejectTracePath('trace', selectedTraceId)}
              className="rounded-full border border-red-500/40 bg-red-500/10 px-3 py-1 text-xs font-semibold text-red-700"
            >
              Delete trace {selectedTraceId}
            </button>
          ) : null}
          {selectedBranchId ? (
            <button
              type="button"
              onClick={() => rejectTracePath('branch', selectedBranchId)}
              className="rounded-full border border-red-500/40 bg-red-500/10 px-3 py-1 text-xs font-semibold text-red-700"
            >
              Delete branch {selectedBranchId}
            </button>
          ) : null}
          {(selectedTraceId || selectedBranchId) ? <div className="mx-2 h-5 w-px bg-[var(--border-muted)]" /> : null}
          <div className="flex items-center gap-2 text-xs font-semibold text-[var(--text-secondary)]">
            <Layers className="h-4 w-4" />
            Layers
          </div>
          <div className="flex flex-wrap gap-2">
            {LAYERS.map(([key, label]) => (
              <label key={key} className="flex cursor-pointer items-center gap-2 rounded-full border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-1 text-xs font-semibold text-[var(--text-secondary)]">
                <input
                  type="checkbox"
                  checked={visibleLayers.has(key)}
                  onChange={() => toggleLayer(key)}
                  className="h-3.5 w-3.5 accent-[var(--accent)]"
                />
                <span>{label}</span>
              </label>
            ))}
          </div>
          <div className="ml-auto text-xs text-[var(--text-secondary)]">
            {isLoading ? 'Loading workspace...' : `${workspace?.objects.length ?? 0} objects, ${workspace?.equipment.length ?? 0} equipment`}
            <span className="mx-2">·</span>
            {isLoadingLayers ? 'Loading layers...' : `${loadedLayerCount} layers`}
            <span className="mx-2">·</span>
            Recompute: <span className="font-semibold">{recomputeState}</span>
            {isDirty ? <span className="ml-2 rounded-full bg-amber-200 px-2 py-0.5 text-amber-900">dirty</span> : null}
          </div>
        </div>
      </div>

      <div className="relative flex min-h-0 flex-1">
        <main className="relative min-h-0 flex-1 bg-[var(--bg-canvas)]">
          {imageUrl ? (
            <CanvasView
              imageUrl={imageUrl}
              objects={visibleCanvasObjects}
              selectedObjectKey={selectedObjectKey}
              selectedObject={selectedCanvasObject}
              reviewStatus={reviewStatus}
              onSelectObject={selectCanvasObject}
              onSetReviewStatus={setCanvasReviewStatus}
              isEditing={isEditing}
              editDraft={editDraft}
              onStartEdit={startCanvasEdit}
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
              onReplaceEditDraft={setEditDraft}
              onSaveEdit={saveCanvasEdit}
              onDeleteSelected={deleteCanvasSelected}
              onNavigatePrevious={() => navigateCanvas(-1)}
              onNavigateNext={() => navigateCanvas(1)}
              isCreating={isCreating}
              createDraft={createDraft}
              onCreateDraftChange={setCreateDraft}
              fitKey={`workspace:${job.job_id}`}
              imageOverlay={
                <ReviewCanvasLayers
                  workspace={workspace}
                  layers={layerPayloads}
                  visibleLayers={visibleLayers}
                  imageSize={imageSize}
                  selectedPortId={selectedPortId}
                  onSelectPort={(portId) => {
                    setSelectedPortId((current) => current === portId ? null : portId)
                    setSelectedTraceId(null)
                    setSelectedBranchId(null)
                  }}
                  selectedTraceId={selectedTraceId}
                  onSelectTrace={(traceId) => {
                    setSelectedTraceId((current) => current === traceId ? null : traceId)
                    setSelectedBranchId(null)
                    setSelectedPortId(null)
                    setSelectedObjectKey(null)
                    setSelectedEntity(null)
                  }}
                  selectedBranchId={selectedBranchId}
                  onSelectBranch={(branchId) => {
                    setSelectedBranchId((current) => current === branchId ? null : branchId)
                    setSelectedTraceId(null)
                    setSelectedPortId(null)
                    setSelectedObjectKey(null)
                    setSelectedEntity(null)
                  }}
                  embedded
                  showBoxes={false}
                />
              }
            />
          ) : (
            <div className="flex h-full min-h-[520px] w-full items-center justify-center bg-[var(--bg-primary)] text-sm text-[var(--text-secondary)]">
                No image artifact available.
            </div>
          )}
          <div className="pointer-events-none absolute left-4 top-12 rounded-full border border-[var(--border-muted)] bg-[var(--bg-secondary)]/90 px-3 py-1 text-xs font-semibold text-[var(--text-secondary)]">
            {imageSize ? `${imageSize.width} x ${imageSize.height}px` : 'Loading image'}
          </div>
        </main>

        <div className="min-h-0 w-[320px] shrink-0 border-l border-[var(--border-muted)] bg-[var(--bg-secondary)]">
          <ObjectSidebar
            objects={canvasObjects}
            visibleObjects={visibleCanvasObjects}
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
            onSetReviewStatus={setCanvasReviewStatus}
            selectedObjectKey={selectedObjectKey}
            onSelectObject={selectCanvasObject}
            isCreating={isCreating}
            createDraft={createDraft}
            onStartCreate={() => addEntity('objects')}
            onCancelCreate={() => {
              setIsCreating(false)
              setCreateDraft(null)
            }}
            onUpdateCreateDraft={updateCreateDraft}
            onSaveCreate={saveCanvasCreate}
            onExport={exportWorkspaceObjects}
          />
        </div>
      </div>
    </div>
  )
}
