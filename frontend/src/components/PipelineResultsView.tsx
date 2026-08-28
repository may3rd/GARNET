import { useEffect, useMemo, useRef, useState } from 'react'
import { CheckCircle2, ChevronDown, Clock, Loader2, Maximize2, PanelLeftClose, PanelLeftOpen, RotateCcw, X } from 'lucide-react'
import type { DetectedObject, PipelineArtifact, PipelineJob, PipelineReviewBucket, PipelineReviewDecision, PipelineReviewItem, PipelineStageManifest } from '@/types'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { PipelineArtifactCanvas } from '@/components/PipelineArtifactCanvas'
import { PipelineHitlReviewView } from '@/components/PipelineHitlReviewView'
import { PipelineReviewWorkspaceView } from '@/components/PipelineReviewWorkspaceView'
import { GraphQaReviewView } from '@/components/GraphQaReviewView'
import { Stage6LineAssociationReview } from '@/components/Stage6LineAssociationReview'
import { getPipelineJob, getPipelineReviewedGraph, getPipelineReviewedQa, getPipelineStageStatus, putPipelineArtifact, resumePipelineFromStage } from '@/lib/api'
import { useAppStore } from '@/stores/appStore'

type JsonValue = string | number | boolean | null | JsonObject | JsonValue[]
type JsonObject = Record<string, JsonValue>
type ReviewBucket = PipelineReviewBucket
type ReviewDecision = PipelineReviewDecision

const REVIEW_STORAGE_PREFIX = 'garnet-pipeline-review'
const EQUIPMENT_CLASSES = new Set(['pump', 'heat exchanger', 'tank', 'vessel', 'column', 'compressor', 'blower', 'fan'])
const PRE_STAGE5_REVIEW_BUCKETS: ReviewBucket[] = ['stage3_equipment', 'stage4_object', 'stage4_line_number']

type HitlGateId = 'object' | 'trace' | 'stage6' | 'qa'

const HITL_GATES: Array<{ id: HitlGateId; after: string; name: string }> = [
  { id: 'object', after: 'stage4_instrument_tag_fusion', name: 'Equipment & object review' },
  { id: 'trace', after: 'stage5b_pipe_trace', name: 'Trace review' },
  { id: 'stage6', after: 'stage6_trace_associations', name: 'Stage 6 line review' },
  { id: 'qa', after: 'stage8_graph_qa', name: 'Graph QA review' },
]

type ReviewRow =
  | { kind: 'stage'; stage: PipelineStageManifest }
  | { kind: 'gate'; gateId: HitlGateId }

// Canonical pipeline stage order. Re-runs (rework/resume) can leave duplicate
// entries in a job's manifest, so stages are deduplicated by name and sorted by
// this order before rendering.
const PIPELINE_STAGE_ORDER = [
  'stage1_input_normalization',
  'stage2_ocr_discovery',
  'stage4_object_detection',
  'stage4_line_number_fusion',
  'stage4_instrument_tag_fusion',
  'stage5_pipe_mask',
  'stage5b_pipe_trace',
  'stage6_trace_associations',
  'stage7_geometric_graph_assembly',
  'stage7c_page_connector_labeling',
  'stage7b_graph_export',
  'stage8_graph_qa',
  'stage9_apply_review_decisions',
  'stage10_process_exports',
  'stage11_connection_overlay',
]

function dedupeStages(stages: PipelineStageManifest[]): PipelineStageManifest[] {
  const byName = new Map<string, PipelineStageManifest>()
  for (const stage of stages) {
    if (stage && typeof stage.name === 'string') byName.set(stage.name, stage)
  }
  return Array.from(byName.values()).sort(
    (a, b) => PIPELINE_STAGE_ORDER.indexOf(a.name) - PIPELINE_STAGE_ORDER.indexOf(b.name)
  )
}

function normalizedClassName(value: string | undefined): string {
  return (value ?? '').toLowerCase().replace(/[_-]+/g, ' ').trim()
}

function isLineNumberClass(className: string | undefined): boolean {
  const normalized = normalizedClassName(className)
  return normalized === 'line number' || normalized === 'line_number'
}

function isEquipmentClass(className: string | undefined): boolean {
  return EQUIPMENT_CLASSES.has(normalizedClassName(className))
}

function reviewStorageKey(jobId: string) {
  return `${REVIEW_STORAGE_PREFIX}:${jobId}`
}

function pickBaseImageUrl(imageArtifacts: { name: string; url: string }[]): string {
  for (const name of ['stage1_gray.png', 'stage1_gray_equalized.png']) {
    const match = imageArtifacts.find((artifact) => artifact.name === name)
    if (match) return match.url
  }
  const firstNonOverlay = imageArtifacts.find((artifact) => !artifact.name.includes('overlay'))
  return firstNonOverlay?.url ?? imageArtifacts[0]?.url ?? ''
}

function toNumber(value: JsonValue | undefined): number | undefined {
  return typeof value === 'number' ? value : undefined
}

function toStringValue(value: JsonValue | undefined): string | undefined {
  return typeof value === 'string' ? value : undefined
}

function toJsonObject(value: JsonValue | undefined): JsonObject | undefined {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as JsonObject) : undefined
}

function buildReviewItems(
  payloadName: string,
  payload: JsonObject | undefined
): PipelineReviewItem[] {
  if (!payload) return []
  const items: JsonObject[] =
    payloadName === 'stage4_line_numbers.json'
      ? ((payload.line_numbers as JsonObject[] | undefined) ?? [])
      : payloadName === 'stage4_instrument_tags.json'
        ? ((payload.instrument_tags as JsonObject[] | undefined) ?? [])
        : payloadName === 'stage12_text_attachments.json'
          ? ([...(((payload.accepted as JsonObject[] | undefined) ?? [])), ...(((payload.rejected as JsonObject[] | undefined) ?? []))] as JsonObject[])
          : ([...(((payload.accepted as JsonObject[] | undefined) ?? [])), ...(((payload.rejected as JsonObject[] | undefined) ?? []))] as JsonObject[])

  return items.map((item, index) => {
    const bucket: ReviewBucket =
      payloadName === 'stage4_line_numbers.json'
        ? 'stage4_line_number'
        : payloadName === 'stage4_instrument_tags.json'
          ? 'stage4_instrument'
          : payloadName === 'stage12_text_attachments.json'
            ? 'stage12_line_attachment'
            : 'stage12_instrument_attachment'

    const id =
      toStringValue(item.id) ??
      toStringValue(item.region_id) ??
      toStringValue(item.source_object_id) ??
      `${bucket}_${index + 1}`

    const text = toStringValue(item.text) ?? ''
    const normalizedText = toStringValue(item.normalized_text) ?? ''
    const reviewState = toStringValue(item.review_state)
    const ocrSource = toStringValue(item.ocr_source)
    const distancePx = toNumber(item.distance_px)
    const thresholdPx = toNumber(item.threshold_px)
    const edgeId = toStringValue(item.edge_id)
    const sourceObjectId = toStringValue(item.source_object_id)

    const artifactName =
      bucket === 'stage4_line_number'
        ? 'stage4_line_number_overlay.png'
        : bucket === 'stage4_instrument'
          ? 'stage4_instrument_tag_overlay.png'
          : 'stage12_text_attachment_overlay.png'

    const title =
      text || normalizedText || edgeId || sourceObjectId || `${bucket.replaceAll('_', ' ')} ${index + 1}`

    const subtitle =
      bucket === 'stage4_line_number'
        ? `Stage 4 line number`
        : bucket === 'stage4_instrument'
          ? `Stage 4 instrument semantic`
          : bucket === 'stage12_line_attachment'
            ? `Stage 12 line attachment`
            : `Stage 12 instrument attachment`

    const statusHint = [reviewState, ocrSource, edgeId].filter(Boolean).join(' • ')

    return {
      bucket,
      id,
      title,
      subtitle,
      text,
      normalizedText,
      artifactName,
      statusHint,
      bbox: toJsonObject(item.bbox),
      ocrSource,
      reviewState,
      distancePx,
      thresholdPx,
      edgeId,
      sourceObjectId,
    }
  })
}

function buildStage3EquipmentItems(
  stage3Payload: JsonObject | undefined,
  stage4Payload: JsonObject | undefined
): PipelineReviewItem[] {
  const reviewedEquipment = (stage3Payload?.equipment as JsonObject[] | undefined) ?? []
  const sourceItems = reviewedEquipment.length
    ? reviewedEquipment
    : ((stage4Payload?.objects as JsonObject[] | undefined) ?? []).filter((item) => {
      return isEquipmentClass(toStringValue(item.class_name))
    })

  return sourceItems.map((item, index) => {
    const id = toStringValue(item.id) ?? `equip_${String(index + 1).padStart(3, '0')}`
    const className = toStringValue(item.class_name) ?? 'vessel'
    const bbox = toJsonObject(item.bbox)
    return {
      bucket: 'stage3_equipment',
      id,
      title: className,
      subtitle: reviewedEquipment.length ? 'Stage 3 reviewed equipment' : 'Stage 4 equipment fallback',
      text: id,
      normalizedText: id,
      artifactName: 'stage4_objects_overlay.png',
      statusHint: reviewedEquipment.length ? 'reviewed equipment bbox' : 'fallback from Stage 4 object detection',
      bbox,
      sourceObjectId: id,
      reviewState: toStringValue(item.review_state),
    }
  })
}

function buildStage4ObjectItems(stage4Payload: JsonObject | undefined): PipelineReviewItem[] {
  return ((stage4Payload?.objects as JsonObject[] | undefined) ?? [])
    .filter((item) => {
      const className = toStringValue(item.class_name)
      return !isEquipmentClass(className) && !isLineNumberClass(className)
    })
    .map((item, index) => {
    const id = toStringValue(item.id) ?? `obj_${String(index + 1).padStart(6, '0')}`
    const className = toStringValue(item.class_name) ?? 'object'
    const confidence = toNumber(item.confidence)
    const source = toStringValue(item.source)
    const reviewState = toStringValue(item.review_state)
    return {
      bucket: 'stage4_object',
      id,
      title: className,
      subtitle: `Stage 4 object ${id}`,
      text: id,
      normalizedText: id,
      artifactName: 'stage4_objects_overlay.png',
      statusHint: [source, reviewState, confidence !== undefined ? `score ${confidence.toFixed(3)}` : undefined].filter(Boolean).join(' • '),
      bbox: toJsonObject(item.bbox),
      reviewState,
      sourceObjectId: id,
    }
  })
}

function buildStage6LineAssociationItems(
  tracePayload: JsonObject | undefined,
  reviewPayload: JsonObject | undefined
): PipelineReviewItem[] {
  const accepted = new Map<string, JsonObject>()
  ;((reviewPayload?.accepted as JsonObject[] | undefined) ?? []).forEach((item) => {
    const traceId = toStringValue(item.trace_id)
    if (traceId) accepted.set(traceId, item)
  })
  const rejected = new Set<string>()
  ;((reviewPayload?.needs_review as JsonObject[] | undefined) ?? []).forEach((item) => {
    const traceId = toStringValue(item.trace_id) ?? toStringValue(item.id)
    if (traceId) rejected.add(traceId)
  })

  return ((tracePayload?.trace_edges as JsonObject[] | undefined) ?? []).map((edge, index) => {
    const traceId = toStringValue(edge.trace_id) ?? `trace_${index + 1}`
    const attachments = toJsonObject(edge.attachments)
    const lineNumbers = (attachments?.line_numbers as JsonObject[] | undefined) ?? []
    const firstLine = accepted.get(traceId) ?? lineNumbers[0]
    const text = toStringValue(firstLine?.text) ?? ''
    const normalizedText = toStringValue(firstLine?.normalized_text) ?? text
    const reviewState = accepted.has(traceId) ? 'accepted' : rejected.has(traceId) ? 'needs_review' : normalizedText ? 'system' : 'missing_line_number'
    return {
      bucket: 'stage6_line_association',
      id: traceId,
      title: normalizedText || traceId,
      subtitle: `Stage 6 ${toStringValue(edge.trace_kind) || 'trace'} line association`,
      text,
      normalizedText,
      artifactName: 'stage6_trace_association_overlay.png',
      statusHint: reviewState,
      reviewState,
      sourceObjectId: toStringValue(edge.source_obj_id),
    }
  })
}

function equipmentObjectsToStage3Artifact(objects: DetectedObject[]) {
  const acceptedEquipment = objects.filter((obj) => obj.ReviewStatus !== 'rejected')
  return {
    equipment: acceptedEquipment.map((obj, index) => ({
      id: obj.Text?.trim() || `equip_${String(index + 1).padStart(3, '0')}`,
      class_name: obj.Object || 'vessel',
      bbox: {
        x_min: Math.round(obj.Left),
        y_min: Math.round(obj.Top),
        x_max: Math.round(obj.Left + obj.Width),
        y_max: Math.round(obj.Top + obj.Height),
      },
      source: 'hitl',
      review_state: obj.ReviewStatus ?? 'accepted',
    })),
  }
}

function detectedObjectsToStage4Artifact(objects: DetectedObject[], imageId?: string) {
  const acceptedObjects = objects.filter((obj) => obj.ReviewStatus !== 'rejected')
  return {
    ...(imageId ? { image_id: imageId } : {}),
    objects: acceptedObjects.map((obj, index) => ({
      id: obj.Text?.trim() || `obj_${String(index + 1).padStart(6, '0')}`,
      class_name: obj.Object || 'object',
      bbox: {
        x_min: Math.round(obj.Left),
        y_min: Math.round(obj.Top),
        x_max: Math.round(obj.Left + obj.Width),
        y_max: Math.round(obj.Top + obj.Height),
      },
      confidence: Number.isFinite(obj.Score) ? obj.Score : 1,
      source: 'hitl',
      review_state: obj.ReviewStatus ?? 'accepted',
    })),
  }
}

function stageStatusClass(status: PipelineStageManifest['status']) {
  if (status === 'completed') return 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700'
  if (status === 'stale') return 'border-amber-500/40 bg-amber-500/10 text-amber-700'
  if (status === 'failed') return 'border-red-500/40 bg-red-500/10 text-red-700'
  if (status === 'started' || status === 'running') return 'border-blue-500/40 bg-blue-500/10 text-blue-700'
  return 'border-[var(--border-muted)] bg-[var(--bg-primary)] text-[var(--text-secondary)]'
}

// Stop point used when reworking from an earlier stage: the run pauses at the
// next HITL gate after the reworked stage (object review, trace review, graph QA).
function reworkStopAfter(stageName: string): number | undefined {
  if (/^stage(1|2|4)_/.test(stageName)) return 4
  if (stageName === 'stage5_pipe_mask' || stageName === 'stage5b_pipe_trace') return 5
  if (stageName === 'stage6_trace_associations') return 6
  if (/^stage(7|8)_/.test(stageName)) return 8
  return undefined
}

function SummaryCard({ title, entries }: { title: string; entries: Array<[string, JsonValue | undefined]> }) {
  const visibleEntries = entries.filter(([, value]) => value !== undefined)
  if (!visibleEntries.length) return null
  return (
    <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
      <div className="text-sm font-semibold">{title}</div>
      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {visibleEntries.map(([label, value]) => (
          <div key={label} className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
            <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">{label}</div>
            <div className="mt-1 text-sm font-semibold">{String(value)}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function PipelineResultsView({ job }: { job: PipelineJob }) {
  const [liveJob, setLiveJob] = useState(job)
  const activeJob = liveJob
  const [jsonSummaries, setJsonSummaries] = useState<Record<string, JsonObject>>({})
  const [jsonDetails, setJsonDetails] = useState<Record<string, JsonObject>>({})
  const [stageStatuses, setStageStatuses] = useState<PipelineStageManifest[]>(job.manifest?.stages ?? [])
  const [activeArtifactName, setActiveArtifactName] = useState<string | null>(null)
  const [expandedArtifactName, setExpandedArtifactName] = useState<string | null>(null)
  const [activeReviewBucket, setActiveReviewBucket] = useState<ReviewBucket>('stage3_equipment')
  const [reviewDecisions, setReviewDecisions] = useState<Record<string, ReviewDecision>>({})
  const [workspaceOpen, setWorkspaceOpen] = useState(false)
  const [preStage5ReviewActive, setPreStage5ReviewActive] = useState(false)
  const [preStage5ReviewDismissed, setPreStage5ReviewDismissed] = useState(false)
  const [isResuming, setIsResuming] = useState(false)
  const [isSavingStage6, setIsSavingStage6] = useState(false)
  const [isSavingStage8, setIsSavingStage8] = useState(false)
  const [showArtifactDetails, setShowArtifactDetails] = useState(false)
  const [pipelineActionError, setPipelineActionError] = useState<string | null>(null)
  const [graphMode, setGraphMode] = useState<'raw' | 'reviewed'>('raw')
  const [reviewedGraphSummary, setReviewedGraphSummary] = useState<JsonObject | null>(null)
  const [reviewedQaSummary, setReviewedQaSummary] = useState<JsonObject | null>(null)
  const stages = dedupeStages(stageStatuses.length ? stageStatuses : activeJob.manifest?.stages ?? [])
  const imageArtifacts = useMemo(
    () => activeJob.artifacts.filter((artifact) => /\.(png|jpg|jpeg|webp)$/i.test(artifact.name)),
    [activeJob.artifacts]
  )
  const jsonArtifacts = useMemo(
    () => activeJob.artifacts.filter((artifact) => artifact.name.endsWith('.json')),
    [activeJob.artifacts]
  )
  const route = activeJob.manifest?.ocr_route ?? activeJob.ocr_route
  const spotlightImageArtifacts = useMemo(
    () =>
      imageArtifacts.filter((artifact) =>
        [
          'stage4_line_number_overlay.png',
          'stage6_trace_association_overlay.png',
          'stage12_text_attachment_overlay.png',
        ].includes(artifact.name)
      ),
    [imageArtifacts]
  )
  const summaryArtifacts = useMemo(
    () =>
      jsonArtifacts.filter((artifact) =>
        [
          'stage2_ocr_summary.json',
          'stage4_objects_summary.json',
          'stage4_line_number_summary.json',
          'stage6_trace_association_summary.json',
          'stage6_line_number_review_summary.json',
          'stage7_graph_summary.json',
          'stage7_graph_qa_summary.json',
          'stage5_pipe_mask_summary.json',
          'stage6_pipe_mask_sealed_summary.json',
          'stage7_pipe_skeleton_summary.json',
          'stage8_node_summary.json',
          'stage9_node_cluster_summary.json',
          'stage10_pipe_edge_summary.json',
          'stage11_junction_review_summary.json',
          'stage12_equipment_attachment_summary.json',
          'stage12_text_attachment_summary.json',
          'stage12_graph_summary.json',
          'stage8_graph_qa_summary.json',
        ].includes(artifact.name)
      ),
    [jsonArtifacts]
  )
  const detailArtifacts = useMemo(
    () =>
      jsonArtifacts.filter((artifact) =>
        [
          'stage4_line_numbers.json',
          'stage4_instrument_tags.json',
          'stage3_equipment_bboxes.json',
          'stage4_objects.json',
          'stage6_trace_associations.json',
          'stage6_line_number_review.json',
          'stage8_review_items.json',
          'stage8_review_decisions.json',
          'stage12_text_attachments.json',
          'stage12_instrument_tag_attachments.json',
        ].includes(artifact.name)
      ),
    [jsonArtifacts]
  )

  useEffect(() => {
    let active = true
    const load = async () => {
      const results = await Promise.all(
        summaryArtifacts.map(async (artifact) => {
          try {
            const response = await fetch(artifact.url)
            if (!response.ok) return [artifact.name, null] as const
            const payload = (await response.json()) as JsonObject
            return [artifact.name, payload] as const
          } catch {
            return [artifact.name, null] as const
          }
        })
      )
      if (!active) return
      setJsonSummaries(
        Object.fromEntries(results.filter(([, payload]) => payload !== null)) as Record<string, JsonObject>
      )
    }
    void load()
    return () => {
      active = false
    }
  }, [summaryArtifacts])

  useEffect(() => {
    let active = true
    const load = async () => {
      const results = await Promise.all(
        detailArtifacts.map(async (artifact) => {
          try {
            const response = await fetch(artifact.url)
            if (!response.ok) return [artifact.name, null] as const
            const payload = (await response.json()) as JsonObject
            return [artifact.name, payload] as const
          } catch {
            return [artifact.name, null] as const
          }
        })
      )
      if (!active) return
      setJsonDetails(
        Object.fromEntries(results.filter(([, payload]) => payload !== null)) as Record<string, JsonObject>
      )
    }
    void load()
    return () => {
      active = false
    }
  }, [detailArtifacts])

  useEffect(() => {
    const raw = window.localStorage.getItem(reviewStorageKey(activeJob.job_id))
    if (!raw) {
      setReviewDecisions({})
      return
    }
    try {
      setReviewDecisions(JSON.parse(raw) as Record<string, ReviewDecision>)
    } catch {
      setReviewDecisions({})
    }
  }, [activeJob.job_id])

  useEffect(() => {
    window.localStorage.setItem(reviewStorageKey(activeJob.job_id), JSON.stringify(reviewDecisions))
  }, [activeJob.job_id, reviewDecisions])

  useEffect(() => {
    let active = true
    const load = async () => {
      try {
        const [graph, qa] = await Promise.all([
          getPipelineReviewedGraph(activeJob.job_id),
          getPipelineReviewedQa(activeJob.job_id),
        ])
        if (!active) return
        setReviewedGraphSummary(graph.summary as JsonObject)
        setReviewedQaSummary(qa.summary as JsonObject)
      } catch {
        if (!active) return
        setReviewedGraphSummary(null)
        setReviewedQaSummary(null)
      }
    }
    void load()
    return () => {
      active = false
    }
  }, [activeJob.job_id, reviewDecisions])

  useEffect(() => {
    setLiveJob(job)
    setStageStatuses(job.manifest?.stages ?? [])
    setPreStage5ReviewActive(false)
    setPreStage5ReviewDismissed(false)
    setSelectedStageName(null)
    setCanvasOverrideName(null)
  }, [job])

  useEffect(() => {
    let active = true
    void getPipelineStageStatus(activeJob.job_id)
      .then((payload) => {
        if (active) setStageStatuses(payload.stages)
      })
      .catch(() => {
        if (active) setStageStatuses(activeJob.manifest?.stages ?? [])
      })
    return () => {
      active = false
    }
  }, [activeJob.job_id, activeJob.manifest])

  useEffect(() => {
    if (!imageArtifacts.length) {
      setActiveArtifactName(null)
      return
    }
    if (activeArtifactName && imageArtifacts.some((artifact) => artifact.name === activeArtifactName)) {
      return
    }
    setActiveArtifactName((spotlightImageArtifacts[0] ?? imageArtifacts[0])?.name ?? null)
  }, [activeArtifactName, imageArtifacts, spotlightImageArtifacts])

  const expandedArtifact = expandedArtifactName
    ? (imageArtifacts.find((artifact) => artifact.name === expandedArtifactName) ?? null)
    : null
  const reviewItems = useMemo(
    () => ({
      stage3_equipment: buildStage3EquipmentItems(jsonDetails['stage3_equipment_bboxes.json'], jsonDetails['stage4_objects.json']),
      stage4_object: buildStage4ObjectItems(jsonDetails['stage4_objects.json']),
      stage4_line_number: buildReviewItems('stage4_line_numbers.json', jsonDetails['stage4_line_numbers.json']),
      stage4_instrument: buildReviewItems('stage4_instrument_tags.json', jsonDetails['stage4_instrument_tags.json']),
      stage6_line_association: buildStage6LineAssociationItems(jsonDetails['stage6_trace_associations.json'], jsonDetails['stage6_line_number_review.json']),
      stage12_line_attachment: buildReviewItems('stage12_text_attachments.json', jsonDetails['stage12_text_attachments.json']),
      stage12_instrument_attachment: buildReviewItems('stage12_instrument_tag_attachments.json', jsonDetails['stage12_instrument_tag_attachments.json']),
    }),
    [jsonDetails]
  )

  const reviewCounts = useMemo(() => {
    const counts: Record<ReviewBucket, Record<ReviewDecision, number>> = {
      stage3_equipment: { accepted: 0, rejected: 0, deferred: 0 },
      stage4_object: { accepted: 0, rejected: 0, deferred: 0 },
      stage4_line_number: { accepted: 0, rejected: 0, deferred: 0 },
      stage4_instrument: { accepted: 0, rejected: 0, deferred: 0 },
      stage6_line_association: { accepted: 0, rejected: 0, deferred: 0 },
      stage12_line_attachment: { accepted: 0, rejected: 0, deferred: 0 },
      stage12_instrument_attachment: { accepted: 0, rejected: 0, deferred: 0 },
    }
    ;(Object.keys(reviewItems) as ReviewBucket[]).forEach((bucket) => {
      reviewItems[bucket].forEach((item) => {
        const decision = reviewDecisions[`${bucket}:${item.id}`] ?? 'deferred'
        counts[bucket][decision] += 1
      })
    })
    return counts
  }, [reviewDecisions, reviewItems])

  const staleFromStage7 = stages.some((stage) => stage.name === 'stage7_geometric_graph_assembly' && stage.status === 'stale')
    || stages.some((stage) => stage.status === 'stale' && (stage.num ?? 0) >= 7)
  const stage5Started = stages.some((stage) => (stage.num ?? 0) >= 5 && stage.status !== 'pending')
  const requiresPreStage5Review = activeJob.status === 'completed' && !stage5Started && (activeJob.stop_after ?? 4) <= 4
  const stage5bComplete = stages.some((stage) => stage.name === 'stage5b_pipe_trace' && stage.status === 'completed')
  const stage6Complete = stages.some((stage) => stage.name === 'stage6_trace_associations' && stage.status === 'completed')
  const requiresTraceReview = activeJob.status === 'completed' && stage5bComplete && !stage6Complete && (activeJob.stop_after ?? 5) <= 5
  const stage7Complete = stages.some((stage) => stage.name === 'stage7_geometric_graph_assembly' && stage.status === 'completed')
  const requiresStage6Review = activeJob.status === 'completed' && stage6Complete && !stage7Complete && (activeJob.stop_after ?? 6) <= 6
  const stage8Complete = stages.some((stage) => stage.name === 'stage8_graph_qa' && stage.status === 'completed')
  const stage9Complete = stages.some((stage) => stage.name === 'stage9_apply_review_decisions' && stage.status === 'completed')
  const requiresGraphQaReview = activeJob.status === 'completed' && stage8Complete && !stage9Complete && (activeJob.stop_after ?? 8) <= 8

  const gateStatus = (gateId: HitlGateId): 'completed' | 'awaiting' | 'pending' => {
    if (gateId === 'object') return stage5Started ? 'completed' : requiresPreStage5Review ? 'awaiting' : 'pending'
    if (gateId === 'trace') return stage6Complete ? 'completed' : requiresTraceReview ? 'awaiting' : 'pending'
    if (gateId === 'stage6') return stage7Complete ? 'completed' : requiresStage6Review ? 'awaiting' : 'pending'
    return stage9Complete ? 'completed' : requiresGraphQaReview ? 'awaiting' : 'pending'
  }
  const enterGate = (gateId: HitlGateId) => {
    if (gateStatus(gateId) !== 'awaiting') return
    setStageOutputActive(false)
    if (gateId !== 'object') return
    setPreStage5ReviewActive(true)
    setPreStage5ReviewDismissed(false)
    setActiveReviewBucket('stage3_equipment')
    setWorkspaceOpen(true)
  }
  const gateStatusClass = (status: 'completed' | 'awaiting' | 'pending') => {
    if (status === 'awaiting') return 'border-amber-500/40 bg-amber-500/10 text-amber-700'
    if (status === 'completed') return 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700'
    return 'border-[var(--border-muted)] bg-[var(--bg-primary)] text-[var(--text-secondary)]'
  }
  const reviewRows: ReviewRow[] = useMemo(() => {
    const rows: ReviewRow[] = []
    for (const stage of stages) {
      rows.push({ kind: 'stage', stage })
      const gate = HITL_GATES.find((gate) => gate.after === stage.name)
      if (gate) rows.push({ kind: 'gate', gateId: gate.id })
    }
    return rows
  }, [stages])

  const [selectedStageName, setSelectedStageName] = useState<string | null>(null)
  const [canvasOverrideName, setCanvasOverrideName] = useState<string | null>(null)
  const selectedStage = useMemo(() => {
    if (selectedStageName) {
      const match = stages.find((stage) => stage.name === selectedStageName)
      if (match) return match
    }
    const current = stages.find((stage) => ['started', 'running', 'stale', 'failed'].includes(stage.status))
    if (current) return current
    const completed = stages.filter((stage) => stage.status === 'completed')
    return completed[completed.length - 1] ?? stages[0] ?? null
  }, [selectedStageName, stages])
  const stageArtifactsByStage = useMemo(() => {
    const byName = new Map(activeJob.artifacts.map((artifact) => [artifact.name, artifact]))
    const map = new Map<string, { images: PipelineArtifact[]; jsons: PipelineArtifact[] }>()
    stages.forEach((stage) => {
      const images: PipelineArtifact[] = []
      const jsons: PipelineArtifact[] = []
      for (const name of stage.artifacts ?? []) {
        const artifact = byName.get(name)
        if (!artifact) continue
        if (/\.(png|jpe?g|webp)$/i.test(name)) images.push(artifact)
        else if (name.endsWith('.json')) jsons.push(artifact)
      }
      map.set(stage.name, { images, jsons })
    })
    return map
  }, [activeJob.artifacts, stages])
  const selectedStageArtifacts = selectedStage
    ? stageArtifactsByStage.get(selectedStage.name) ?? { images: [], jsons: [] }
    : { images: [], jsons: [] }
  const canvasArtifact = useMemo(() => {
    const images = selectedStageArtifacts.images
    const fallback = images.find((artifact) => artifact.name.includes('overlay')) ?? images[0] ?? null
    if (!canvasOverrideName) return fallback
    return images.find((artifact) => artifact.name === canvasOverrideName) ?? fallback
  }, [canvasOverrideName, selectedStageArtifacts])
  const isStageReworkable = (stage: PipelineStageManifest | null) =>
    !!stage && ['completed', 'stale', 'failed'].includes(stage.status)

  const reworkStage = (stage: PipelineStageManifest) => {
    if (stage.name.startsWith('stage4_')) {
      setPreStage5ReviewActive(true)
      setPreStage5ReviewDismissed(false)
      setActiveReviewBucket('stage3_equipment')
      setStageOutputActive(false)
      setWorkspaceOpen(true)
      return
    }
    const confirmed = window.confirm(
      `Rework from ${stage.name}? All later stages will be marked stale and re-run from here.`
    )
    if (!confirmed) return
    void resumeFromStageName(
      stage.name,
      reworkStopAfter(stage.name),
      stage.name === 'stage5_pipe_mask' ? { continueStage5bForTraceReview: true } : {}
    )
  }

  // Gate workspaces render inside the Review shell so the stage rail stays
  // visible; stageOutputActive lets the user peek at a stage's artifacts.
  const [stageOutputActive, setStageOutputActive] = useState(false)
  const [stagesRailOpen, setStagesRailOpen] = useState(true)
  const activeGate: 'hitl' | 'trace' | 'stage6' | 'qa' | null = (() => {
    if (showArtifactDetails) return null
    if (workspaceOpen) return 'hitl'
    if (requiresTraceReview) return 'trace'
    if (requiresStage6Review) return 'stage6'
    if (requiresGraphQaReview) return 'qa'
    return null
  })()
  const gateWorkspaceActive = activeGate !== null && !stageOutputActive
  const gateLabels: Record<'hitl' | 'trace' | 'stage6' | 'qa', string> = {
    hitl: 'HITL review',
    trace: 'trace review',
    stage6: 'Stage 6 line review',
    qa: 'graph QA review',
  }

  useEffect(() => {
    if (!requiresPreStage5Review || preStage5ReviewDismissed || workspaceOpen || isResuming) return
    setPreStage5ReviewActive(true)
    setActiveReviewBucket('stage3_equipment')
    setStageOutputActive(false)
    setWorkspaceOpen(true)
  }, [isResuming, preStage5ReviewDismissed, requiresPreStage5Review, workspaceOpen])

  // When a resume completes, reveal the awaiting HITL gate (trace / stage 6 /
  // graph QA) instead of leaving the stage-output view active. The resume loop
  // sets stageOutputActive(true) to show the canvas while it runs.
  const prevIsResumingRef = useRef(isResuming)
  useEffect(() => {
    const wasResuming = prevIsResumingRef.current
    prevIsResumingRef.current = isResuming
    if (!wasResuming || isResuming) return
    if (requiresTraceReview || requiresStage6Review || requiresGraphQaReview) {
      setStageOutputActive(false)
    }
  }, [isResuming, requiresTraceReview, requiresStage6Review, requiresGraphQaReview])

  const saveStage3Equipment = async (objects: DetectedObject[]) => {
    setPipelineActionError(null)
    const payload = equipmentObjectsToStage3Artifact(objects)
    const response = await putPipelineArtifact(activeJob.job_id, 'stage3_equipment_bboxes.json', payload)
    setStageStatuses(response.stages)
    const refreshedJob = await getPipelineJob(activeJob.job_id)
    setLiveJob(refreshedJob)
  }

  const saveStage4Objects = async (objects: DetectedObject[]) => {
    setPipelineActionError(null)
    const payload = detectedObjectsToStage4Artifact(
      objects,
      toStringValue(jsonDetails['stage4_objects.json']?.image_id)
    )
    const response = await putPipelineArtifact(activeJob.job_id, 'stage4_objects.json', payload)
    setStageStatuses(response.stages)
    const refreshedJob = await getPipelineJob(activeJob.job_id)
    setLiveJob(refreshedJob)
  }

  const resumeFromStage7 = async () => {
    await resumeFromStageName('stage7_geometric_graph_assembly', 8)
  }

  const resumeFromStageName = async (
    stageName: string,
    stopAfter?: number,
    options: { continueStage5bForTraceReview?: boolean } = {}
  ) => {
    setIsResuming(true)
    setShowArtifactDetails(false)
    setWorkspaceOpen(false)
    setPreStage5ReviewActive(false)
    setPreStage5ReviewDismissed(true)
    setPipelineActionError(null)
    useAppStore.setState({
      isProcessing: true,
      processingMode: 'pipeline',
      pipelineJob: activeJob,
      progress: {
        step: `Resuming from ${stageName.replaceAll('_', ' ')}`,
        percent: 8,
      },
    })
    try {
      await resumePipelineFromStage(activeJob.job_id, stageName, { stopAfter })
      let continuedStage5bForTraceReview = false
      while (true) {
        const nextJob = await getPipelineJob(activeJob.job_id)
        setLiveJob(nextJob)
        const totalStages = Math.max(nextJob.manifest?.stages.length ?? nextJob.manifest?.stop_after ?? 1, 1)
        const completedStages = nextJob.manifest?.stages.filter((stage) => stage.status === 'completed').length ?? 0
        const percent = nextJob.status === 'completed'
          ? 100
          : Math.min(95, Math.max(10, Math.round((completedStages / totalStages) * 100)))
        useAppStore.setState({
          pipelineJob: nextJob,
          progress: {
            step: (nextJob.current_stage ?? 'Resuming pipeline').replaceAll('_', ' '),
            percent,
          },
        })
        let latestStages: PipelineStageManifest[] = []
        try {
          const statusPayload = await getPipelineStageStatus(activeJob.job_id)
          setStageStatuses(statusPayload.stages)
          latestStages = statusPayload.stages
        } catch {
          latestStages = nextJob.manifest?.stages ?? []
          setStageStatuses(latestStages)
        }
        // Follow the most recently completed stage so the canvas shows its
        // artifact as the pipeline advances.
        const lastCompleted = latestStages.filter((stage) => stage.status === 'completed').pop()
        if (lastCompleted) {
          setSelectedStageName(lastCompleted.name)
          setCanvasOverrideName(null)
          setStageOutputActive(true)
        }
        if (
          nextJob.status === 'completed'
          && options.continueStage5bForTraceReview
          && !continuedStage5bForTraceReview
          && !nextJob.manifest?.stages.some((stage) => stage.name === 'stage5b_pipe_trace' && stage.status === 'completed')
        ) {
          continuedStage5bForTraceReview = true
          await resumePipelineFromStage(activeJob.job_id, 'stage5b_pipe_trace', { stopAfter: 5 })
          useAppStore.setState({
            pipelineJob: nextJob,
            progress: {
              step: 'Resuming from stage5b pipe trace',
              percent: 10,
            },
          })
          continue
        }
        if (nextJob.status === 'completed') {
          useAppStore.setState({
            isProcessing: false,
            pipelineJob: nextJob,
            progress: { step: 'Pipeline complete', percent: 100 },
          })
          break
        }
        if (nextJob.status === 'failed') {
          setPipelineActionError(nextJob.error || 'Pipeline resume failed')
          useAppStore.setState({
            isProcessing: false,
            pipelineJob: nextJob,
            progress: null,
          })
          break
        }
        await new Promise((resolve) => window.setTimeout(resolve, 500))
      }
    } catch (error) {
      setPipelineActionError(error instanceof Error ? error.message : 'Pipeline resume failed')
      useAppStore.setState({
        isProcessing: false,
        progress: null,
      })
    } finally {
      setIsResuming(false)
    }
  }

  const saveStage6LineReview = async (payload: JsonObject) => {
    setIsSavingStage6(true)
    setPipelineActionError(null)
    try {
      const response = await putPipelineArtifact(activeJob.job_id, 'stage6_line_number_review.json', payload)
      setStageStatuses(response.stages)
      const refreshedJob = await getPipelineJob(activeJob.job_id)
      setLiveJob(refreshedJob)
    } catch (error) {
      setPipelineActionError(error instanceof Error ? error.message : 'Failed to save Stage 6 line review')
    } finally {
      setIsSavingStage6(false)
    }
  }

  const saveGraphQaDecisions = async (payload: JsonObject) => {
    setIsSavingStage8(true)
    setPipelineActionError(null)
    try {
      const response = await putPipelineArtifact(activeJob.job_id, 'stage8_review_decisions.json', payload)
      setStageStatuses(response.stages)
      const refreshedJob = await getPipelineJob(activeJob.job_id)
      setLiveJob(refreshedJob)
    } catch (error) {
      setPipelineActionError(error instanceof Error ? error.message : 'Failed to save Stage 8 QA review')
    } finally {
      setIsSavingStage8(false)
    }
  }

  const resumeFromStage9 = async () => {
    await resumeFromStageName('stage9_apply_review_decisions')
  }

  const handleReviewBucketSaved = (bucket: ReviewBucket) => {
    if (!preStage5ReviewActive) {
      setWorkspaceOpen(false)
      return
    }
    if (bucket === 'stage3_equipment') {
      setActiveReviewBucket('stage4_object')
      return
    }
    if (bucket === 'stage4_object') {
      setActiveReviewBucket('stage4_line_number')
      return
    }
    if (bucket === 'stage4_line_number') {
      setPreStage5ReviewActive(false)
      setPreStage5ReviewDismissed(true)
      setWorkspaceOpen(false)
      setShowArtifactDetails(false)
      void resumeFromStageName('stage5_pipe_mask', 5, { continueStage5bForTraceReview: true })
      return
    }
    setWorkspaceOpen(false)
  }

  const closeReviewWorkspace = () => {
    if (preStage5ReviewActive || requiresPreStage5Review) {
      setPreStage5ReviewActive(false)
      setPreStage5ReviewDismissed(true)
    }
    setWorkspaceOpen(false)
  }

  if (expandedArtifact) {
    return (
      <div className="h-full overflow-hidden bg-[var(--bg-canvas)]">
        <div className="flex h-full flex-col p-4">
          <div className="mb-3 flex items-center justify-between rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] px-4 py-3">
            <div>
              <div className="text-sm font-semibold">Artifact Viewer</div>
              <div className="mt-0.5 text-xs text-[var(--text-secondary)]">{expandedArtifact.name}</div>
            </div>
            <button
              type="button"
              onClick={() => setExpandedArtifactName(null)}
              className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] p-2 text-[var(--text-secondary)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
              aria-label="Close artifact viewer"
            >
              <X size={18} />
            </button>
          </div>
          <div className="min-h-0 flex-1">
            <PipelineArtifactCanvas imageUrl={expandedArtifact.url} title={expandedArtifact.name} />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg-canvas)]">
      <div className="shrink-0 border-b border-[var(--border-muted)] bg-[var(--bg-secondary)] px-6 py-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            {!showArtifactDetails ? (
              <button
                type="button"
                onClick={() => setStagesRailOpen((open) => !open)}
                title={stagesRailOpen ? 'Hide stages' : 'Show stages'}
                aria-label={stagesRailOpen ? 'Hide stages panel' : 'Show stages panel'}
                aria-pressed={stagesRailOpen}
                className="mt-0.5 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] p-2 text-[var(--text-secondary)] transition hover:border-[var(--accent)] hover:text-[var(--accent)]"
              >
                {stagesRailOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
              </button>
            ) : null}
            <div className="min-w-0">
              <div className="text-lg font-semibold">Pipeline Review</div>
            <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-[var(--text-secondary)]">
              <span className="font-mono">{activeJob.job_id}</span>
              <span>·</span>
              <span className="capitalize">{activeJob.status}</span>
              <span>·</span>
              <span>{activeJob.current_stage ?? 'Queued'}</span>
              <span>·</span>
              <span className="uppercase">OCR {route}</span>
            </div>
            {pipelineActionError ? <div className="mt-1 text-xs text-red-600">{pipelineActionError}</div> : null}
          </div>
          </div>
          <div className="flex items-center gap-2">
            {requiresPreStage5Review || requiresTraceReview ? (
              <button
                type="button"
                onClick={() => {
                  if (requiresTraceReview) {
                    setShowArtifactDetails(false)
                  } else {
                    setPreStage5ReviewActive(true)
                    setPreStage5ReviewDismissed(false)
                    setActiveReviewBucket('stage3_equipment')
                    setWorkspaceOpen(true)
                  }
                  setStageOutputActive(false)
                }}
                className="rounded-lg border border-[var(--accent)] bg-[var(--accent)]/10 px-3 py-2 text-sm font-semibold text-[var(--accent)]"
              >
                {requiresTraceReview ? 'Back to Trace Review' : 'Continue Pre-Stage-5 Review'}
              </button>
            ) : null}
            <div className="flex rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] p-1 text-xs font-semibold">
              <button
                type="button"
                onClick={() => setShowArtifactDetails(false)}
                className={`rounded-md px-3 py-1.5 transition ${
                  !showArtifactDetails
                    ? 'bg-[var(--accent)]/10 text-[var(--accent)]'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                Review
              </button>
              <button
                type="button"
                onClick={() => setShowArtifactDetails(true)}
                className={`rounded-md px-3 py-1.5 transition ${
                  showArtifactDetails
                    ? 'bg-[var(--accent)]/10 text-[var(--accent)]'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                Summary
              </button>
            </div>
          </div>
        </div>
      </div>

      {showArtifactDetails ? (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 py-6">
            <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
              <div>
                <div className="text-lg font-semibold">Pipeline Artifacts / QA</div>
                <div className="mt-1 text-sm text-[var(--text-secondary)]">
                  Staged pipeline review. The default workflow pauses after Stage 4 object detection for equipment, object, and line-number review before Stage 5 pipe tracing starts.
                </div>
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-3">
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
              <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Job</div>
              <div className="mt-1 font-mono text-xs">{activeJob.job_id}</div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
              <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Status</div>
              <div className="mt-1 text-sm font-semibold capitalize">{activeJob.status}</div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
              <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Current Stage</div>
              <div className="mt-1 text-sm font-semibold">{activeJob.current_stage ?? 'Queued'}</div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
              <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">OCR Route</div>
              <div className="mt-1 text-sm font-semibold uppercase">{route}</div>
            </div>
          </div>
        </div>

        <SummaryCard
          title="OCR Summary"
          entries={[
            ['Route', jsonSummaries['stage2_ocr_summary.json']?.route],
            ['Tiles', jsonSummaries['stage2_ocr_summary.json']?.tile_count],
            ['Raw Regions', jsonSummaries['stage2_ocr_summary.json']?.raw_detection_count],
            ['Merged Regions', jsonSummaries['stage2_ocr_summary.json']?.merged_region_count],
            ['Exceptions', jsonSummaries['stage2_ocr_summary.json']?.exception_candidate_count],
            ['Framework', jsonSummaries['stage2_ocr_summary.json']?.framework],
            ['Recognition', jsonSummaries['stage2_ocr_summary.json']?.recognition_level],
          ]}
        />

        <SummaryCard
          title="Line Numbers"
          entries={[
            ['Detected Tags', jsonSummaries['stage4_line_number_summary.json']?.line_number_object_count],
            ['OCR Confirmed', jsonSummaries['stage4_line_number_summary.json']?.ocr_confirmed_line_number_count],
            ['Detection Only', jsonSummaries['stage4_line_number_summary.json']?.od_only_line_number_count],
            ['Rejected Tags', jsonSummaries['stage4_line_number_summary.json']?.rejected_line_number_count],
            ['Attach Candidates', jsonSummaries['stage12_text_attachment_summary.json']?.candidate_count],
          ]}
        />

        <SummaryCard
          title="Line Number Provenance"
          entries={[
            ['Sheet OCR', jsonSummaries['stage4_line_number_summary.json']?.sheet_ocr_line_number_count],
            ['Crop OCR', jsonSummaries['stage4_line_number_summary.json']?.crop_ocr_line_number_count],
            ['Rotated Crop OCR', jsonSummaries['stage4_line_number_summary.json']?.rotated_crop_ocr_line_number_count],
            ['Detection Only', jsonSummaries['stage4_line_number_summary.json']?.od_only_line_number_count],
          ]}
        />

        <SummaryCard
          title="Instrumentation"
          entries={[
            ['Detected Tags', jsonSummaries['stage4_instrument_tag_summary.json']?.instrument_semantic_object_count],
            ['OCR Confirmed', jsonSummaries['stage4_instrument_tag_summary.json']?.ocr_confirmed_instrument_semantic_count],
            ['Detection Only', jsonSummaries['stage4_instrument_tag_summary.json']?.detection_only_instrument_semantic_count],
            ['Rejected Tags', jsonSummaries['stage4_instrument_tag_summary.json']?.rejected_instrument_semantic_count],
            ['Attached Tags', jsonSummaries['stage12_instrument_tag_attachment_summary.json']?.accepted_attachment_count],
          ]}
        />

        <SummaryCard
          title="Instrumentation Provenance"
          entries={[
            ['Sheet OCR', jsonSummaries['stage4_instrument_tag_summary.json']?.sheet_ocr_instrument_semantic_count],
            ['Crop OCR', jsonSummaries['stage4_instrument_tag_summary.json']?.crop_ocr_instrument_semantic_count],
            ['Rotated Crop OCR', jsonSummaries['stage4_instrument_tag_summary.json']?.rotated_crop_ocr_instrument_semantic_count],
            ['Detection Only', jsonSummaries['stage4_instrument_tag_summary.json']?.detection_only_instrument_semantic_count],
          ]}
        />

        <SummaryCard
          title="Attachments"
          entries={[
            ['Equipment Attached', jsonSummaries['stage12_equipment_attachment_summary.json']?.accepted_attachment_count],
            ['Equipment Rejected', jsonSummaries['stage12_equipment_attachment_summary.json']?.rejected_attachment_count],
            ['Text Attached', jsonSummaries['stage12_text_attachment_summary.json']?.accepted_attachment_count],
            ['Text Rejected', jsonSummaries['stage12_text_attachment_summary.json']?.rejected_attachment_count],
          ]}
        />

        <SummaryCard
          title="Graph Summary"
          entries={[
            ['Nodes', graphMode === 'reviewed' ? reviewedGraphSummary?.node_count : jsonSummaries['stage12_graph_summary.json']?.node_count],
            ['Edges', graphMode === 'reviewed' ? reviewedGraphSummary?.edge_count : jsonSummaries['stage12_graph_summary.json']?.edge_count],
            ['Components', graphMode === 'reviewed' ? reviewedGraphSummary?.connected_component_count : jsonSummaries['stage12_graph_summary.json']?.connected_component_count],
            ['Unresolved Junctions', graphMode === 'reviewed' ? reviewedGraphSummary?.unresolved_junction_count : jsonSummaries['stage12_graph_summary.json']?.unresolved_junction_count],
          ]}
        />

        <SummaryCard
          title="QA Summary"
          entries={[
            ['QA Components', graphMode === 'reviewed' ? reviewedQaSummary?.connected_component_count : jsonSummaries['stage8_graph_qa_summary.json']?.connected_component_count],
            ['Articulation Points', graphMode === 'reviewed' ? reviewedQaSummary?.articulation_point_count : jsonSummaries['stage8_graph_qa_summary.json']?.articulation_point_count],
            ['Isolated Nodes', graphMode === 'reviewed' ? reviewedQaSummary?.isolated_node_count : jsonSummaries['stage8_graph_qa_summary.json']?.isolated_node_count],
            ['Review Queue', graphMode === 'reviewed' ? reviewedQaSummary?.review_queue_count : jsonSummaries['stage8_graph_qa_summary.json']?.review_queue_count],
          ]}
        />

        <SummaryCard
          title="HITL Review"
          entries={[
            ['S3 Equipment Boxes', reviewItems.stage3_equipment.length],
            ['S4 Object Boxes', reviewItems.stage4_object.length],
            ['S4 Line Accepted', reviewCounts.stage4_line_number.accepted],
            ['S4 Instrument Accepted', reviewCounts.stage4_instrument.accepted],
            ['S6 Line Accepted', reviewCounts.stage6_line_association.accepted],
            ['S12 Line Accepted', reviewCounts.stage12_line_attachment.accepted],
            ['S12 Instrument Accepted', reviewCounts.stage12_instrument_attachment.accepted],
          ]}
        />

            <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
              <div className="text-sm font-semibold">Review Flow</div>
              {requiresPreStage5Review ? (
                <div className="mt-2 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-800">
                  Stage 5 is waiting for HITL confirmation: Stage 3 equipment boxes, Stage 4 objects, then Stage 4 line numbers.
                </div>
              ) : null}
              <div className="mt-3">
                <button
                  type="button"
                  onClick={() => setWorkspaceOpen(true)}
                  className="rounded-lg border border-[var(--accent)] bg-[var(--accent)]/10 px-3 py-2 text-sm font-semibold text-[var(--accent)]"
                >
                  Open HITL Review
                </button>
              </div>
              <div className="mt-3 flex gap-2">
                {(['raw', 'reviewed'] as const).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setGraphMode(mode)}
                    className={`rounded-full border px-3 py-1 text-xs font-semibold ${
                      graphMode === mode
                        ? 'border-[var(--accent)] bg-[var(--accent)]/10 text-[var(--accent)]'
                        : 'border-[var(--border-muted)] bg-[var(--bg-primary)] text-[var(--text-secondary)]'
                    }`}
                  >
                    {mode === 'raw' ? 'Raw Outputs' : 'Reviewed Outputs'}
                  </button>
                ))}
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                {([
                  ['stage3_equipment', 'Stage 3 Equipment'],
                  ['stage4_object', 'Stage 4 Objects'],
                  ['stage4_line_number', 'Stage 4 Line Numbers'],
                  ['stage4_instrument', 'Stage 4 Instruments'],
                  ['stage6_line_association', 'Stage 6 Line Associations'],
                  ['stage12_line_attachment', 'Stage 12 Line Attachments'],
                  ['stage12_instrument_attachment', 'Stage 12 Instrument Attachments'],
                ] as Array<[ReviewBucket, string]>).map(([bucket, label]) => {
                  const isActive = bucket === activeReviewBucket
                  const counts = reviewCounts[bucket]
                  return (
                    <button
                      key={bucket}
                      type="button"
                      onClick={() => setActiveReviewBucket(bucket)}
                      className={`rounded-xl border p-3 text-left transition ${
                        isActive
                          ? 'border-[var(--accent)] bg-[var(--bg-primary)] ring-2 ring-[var(--accent)]/25'
                          : 'border-[var(--border-muted)] bg-[var(--bg-primary)] hover:border-[var(--accent)]/50'
                      }`}
                    >
                      <div className="text-sm font-semibold">{label}</div>
                      <div className="mt-2 text-xs text-[var(--text-secondary)]">
                        {counts.accepted} accepted • {counts.rejected} rejected • {counts.deferred} deferred
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>

            <Stage6LineAssociationReview
              tracePayload={jsonDetails['stage6_trace_associations.json']}
              reviewPayload={jsonDetails['stage6_line_number_review.json']}
              baseImageUrl={pickBaseImageUrl(imageArtifacts)}
              overlayUrl={imageArtifacts.find((artifact) => artifact.name === 'stage6_trace_association_overlay.png')?.url}
              stage7Stale={staleFromStage7}
              isSaving={isSavingStage6}
              isResuming={isResuming}
              onSave={saveStage6LineReview}
              onResumeStage7={resumeFromStage7}
            />

            <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
              <div className="text-sm font-semibold">Artifact Thumbnails</div>
              <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {imageArtifacts.map((artifact) => {
                  const isActive = artifact.name === activeArtifactName
                  return (
                    <div
                      key={artifact.name}
                      className={`rounded-xl border bg-[var(--bg-primary)] p-3 text-left transition ${
                        isActive
                          ? 'border-[var(--accent)] ring-2 ring-[var(--accent)]/25'
                          : 'border-[var(--border-muted)] hover:border-[var(--accent)]/50'
                      }`}
                    >
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <div className="min-w-0 truncate text-xs font-semibold text-[var(--text-secondary)]">{artifact.name}</div>
                        <button
                          type="button"
                          onClick={() => {
                            setActiveArtifactName(artifact.name)
                            setExpandedArtifactName(artifact.name)
                          }}
                          className="shrink-0 rounded-md border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-1.5 text-[var(--text-secondary)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
                          aria-label={`Open ${artifact.name} full page`}
                        >
                          <Maximize2 size={14} />
                        </button>
                      </div>
                      <button
                        type="button"
                        onClick={() => setActiveArtifactName(artifact.name)}
                        onDoubleClick={() => setExpandedArtifactName(artifact.name)}
                        className="block w-full"
                      >
                        <img src={artifact.url} alt={artifact.name} className="w-full rounded-lg border border-[var(--border-muted)]" />
                      </button>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
              <div className="text-sm font-semibold">JSON Artifacts</div>
              <div className="mt-4 space-y-2">
                {jsonArtifacts.map((artifact) => (
                  <a
                    key={artifact.name}
                    href={artifact.url}
                    target="_blank"
                    rel="noreferrer"
                    className="block rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm text-[var(--accent)]"
                  >
                    {artifact.name}
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1">
          {stagesRailOpen ? (
          <aside className="flex w-[320px] shrink-0 flex-col border-r border-[var(--border-muted)] bg-[var(--bg-secondary)]">
            <div className="shrink-0 border-b border-[var(--border-muted)] px-4 py-3 text-sm font-semibold">Stages</div>
            <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
              {stages.some((stage) => stage.status === 'stale') ? (
                <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-700">
                  Stale stages reset the workflow: everything after the rework point re-runs.
                </div>
              ) : null}
              {reviewRows.map((row) => {
                if (row.kind === 'gate') {
                  const gate = HITL_GATES.find((item) => item.id === row.gateId)!
                  const status = gateStatus(row.gateId)
                  return (
                    <div
                      key={`gate-${row.gateId}`}
                      onClick={() => enterGate(row.gateId)}
                      className={`rounded-xl border border-dashed p-3 transition ${gateStatusClass(status)} ${
                        status === 'awaiting' ? 'cursor-pointer hover:brightness-95' : ''
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="text-[10px] uppercase tracking-wide opacity-80">HITL gate</div>
                          <div className="truncate text-sm font-semibold">{gate.name}</div>
                        </div>
                        {status === 'completed' ? (
                          <CheckCircle2 size={15} className="shrink-0 opacity-80" />
                        ) : status === 'awaiting' ? (
                          <Clock size={15} className="shrink-0 opacity-80" />
                        ) : null}
                      </div>
                      <div className="mt-1 text-xs opacity-80">
                        {status === 'awaiting'
                          ? 'Awaiting review — click to open'
                          : status === 'completed'
                            ? 'Cleared'
                            : 'Pending'}
                      </div>
                    </div>
                  )
                }
                const stage = row.stage
                const isSelected = selectedStage?.name === stage.name
                const reworkable = isStageReworkable(stage)
                return (
                  <div
                    key={stage.name}
                    onClick={() => {
                      setSelectedStageName(stage.name)
                      setCanvasOverrideName(null)
                      setStageOutputActive(true)
                    }}
                    className={`cursor-pointer rounded-xl border p-3 transition ${stageStatusClass(stage.status)} ${
                      isSelected ? 'ring-2 ring-[var(--accent)]/40' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="text-xs uppercase tracking-wide opacity-80">Stage {stage.num}</div>
                        <div className="truncate text-sm font-semibold">{stage.name}</div>
                      </div>
                      {reworkable ? (
                        <button
                          type="button"
                          title={`Rework from ${stage.name}; later stages will be re-run`}
                          aria-label={`Rework from ${stage.name}`}
                          onClick={(event) => {
                            event.stopPropagation()
                            reworkStage(stage)
                          }}
                          className="shrink-0 rounded-md border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-1.5 text-[var(--text-secondary)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
                        >
                          <RotateCcw size={13} />
                        </button>
                      ) : null}
                    </div>
                    <div className="mt-1 text-xs opacity-80">
                      {stage.status}
                      {stage.duration_sec !== undefined ? ` • ${stage.duration_sec.toFixed(3)}s` : ''}
                      {stage.stale_source_artifact ? ` • from ${stage.stale_source_artifact}` : ''}
                    </div>
                  </div>
                )
              })}
            </div>
          </aside>
          ) : null}

          <main className="flex min-h-0 flex-1 flex-col">
            {gateWorkspaceActive ? (
              <div className="min-h-0 flex-1 overflow-hidden rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-primary)]">
                {activeGate === 'hitl' ? (
                  <PipelineHitlReviewView
                    jobId={activeJob.job_id}
                    activeBucket={activeReviewBucket}
                    itemsByBucket={reviewItems}
                    imageArtifacts={imageArtifacts}
                    initialReviewDecisions={reviewDecisions}
                    onApply={(decisions) => setReviewDecisions(decisions)}
                    onSaveStage3Equipment={saveStage3Equipment}
                    onSaveStage4Objects={saveStage4Objects}
                    onAfterBucketSave={handleReviewBucketSaved}
                    visibleBuckets={preStage5ReviewActive ? PRE_STAGE5_REVIEW_BUCKETS : undefined}
                    onClose={closeReviewWorkspace}
                  />
                ) : activeGate === 'trace' ? (
                  <PipelineReviewWorkspaceView
                    job={activeJob}
                    imageArtifacts={imageArtifacts}
                    onOpenDetails={() => setShowArtifactDetails(true)}
                    onCommitComplete={() => {
                      setShowArtifactDetails(false)
                      void resumeFromStageName('stage6_trace_associations', 6)
                    }}
                  />
                ) : activeGate === 'stage6' ? (
                  <Stage6LineAssociationReview
                    tracePayload={jsonDetails['stage6_trace_associations.json']}
                    reviewPayload={jsonDetails['stage6_line_number_review.json']}
                    baseImageUrl={pickBaseImageUrl(imageArtifacts)}
                    overlayUrl={imageArtifacts.find((artifact) => artifact.name === 'stage6_trace_association_overlay.png')?.url}
                    stage7Stale={staleFromStage7 || requiresStage6Review}
                    isSaving={isSavingStage6}
                    isResuming={isResuming}
                    layout="workspace"
                    onCancel={() => setShowArtifactDetails(true)}
                    onSave={saveStage6LineReview}
                    onResumeStage7={resumeFromStage7}
                  />
                ) : (
                  <GraphQaReviewView
                    reviewItemsPayload={jsonDetails['stage8_review_items.json']}
                    reviewDecisionsPayload={jsonDetails['stage8_review_decisions.json']}
                    baseImageUrl={pickBaseImageUrl(imageArtifacts)}
                    overlayUrl={imageArtifacts.find((artifact) => artifact.name === 'stage8_review_overlay.png')?.url}
                    stage9Stale={requiresGraphQaReview}
                    isSaving={isSavingStage8}
                    isResuming={isResuming}
                    layout="workspace"
                    onCancel={() => setShowArtifactDetails(true)}
                    onSave={saveGraphQaDecisions}
                    onResumeStage9={resumeFromStage9}
                  />
                )}
              </div>
            ) : (
              <>
            <div className="flex items-center justify-between gap-3 px-4 pt-4 pb-3">
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold">{selectedStage?.name ?? 'No stage selected'}</div>
                <div className="text-xs text-[var(--text-secondary)]">
                  {selectedStage
                    ? `${selectedStage.status}${selectedStage.duration_sec !== undefined ? ` • ${selectedStage.duration_sec.toFixed(3)}s` : ''}`
                    : 'Pick a stage from the list'}
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {activeGate && stageOutputActive ? (
                  <button
                    type="button"
                    onClick={() => setStageOutputActive(false)}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--accent)] bg-[var(--accent)]/10 px-3 py-2 text-xs font-semibold text-[var(--accent)]"
                  >
                    Back to {gateLabels[activeGate]}
                  </button>
                ) : null}
                {isStageReworkable(selectedStage) && selectedStage ? (
                  <button
                    type="button"
                    onClick={() => reworkStage(selectedStage)}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs font-semibold text-amber-700"
                  >
                    <RotateCcw size={14} />
                    Rework from here
                  </button>
                ) : null}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      type="button"
                      className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-xs font-semibold text-[var(--text-primary)]"
                    >
                      Open <ChevronDown size={14} />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="max-h-96 w-72 overflow-y-auto">
                    <DropdownMenuItem onSelect={() => setShowArtifactDetails(true)}>
                      Open Summary page
                    </DropdownMenuItem>
                    {selectedStageArtifacts.images.length ? (
                      <>
                        <DropdownMenuSeparator />
                        <DropdownMenuLabel>Stage images</DropdownMenuLabel>
                        {selectedStageArtifacts.images.map((artifact) => (
                          <DropdownMenuItem key={artifact.name} onSelect={() => setCanvasOverrideName(artifact.name)}>
                            {artifact.name}
                          </DropdownMenuItem>
                        ))}
                      </>
                    ) : null}
                    {selectedStageArtifacts.jsons.length ? (
                      <>
                        <DropdownMenuSeparator />
                        <DropdownMenuLabel>Stage JSON</DropdownMenuLabel>
                        {selectedStageArtifacts.jsons.map((artifact) => (
                          <DropdownMenuItem key={artifact.name} asChild>
                            <a href={artifact.url} target="_blank" rel="noreferrer">
                              {artifact.name}
                            </a>
                          </DropdownMenuItem>
                        ))}
                      </>
                    ) : null}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>

            <div className="relative min-h-0 flex-1 overflow-hidden">
              {canvasArtifact ? (
                <PipelineArtifactCanvas imageUrl={canvasArtifact.url} title={canvasArtifact.name} />
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
                  <div className="text-xs text-[var(--text-secondary)]">
                    {selectedStage
                      ? `${selectedStage.name} has no image artifacts.`
                      : 'Select a stage to view its output.'}
                  </div>
                  {selectedStageArtifacts.jsons.length ? (
                    <div className="w-full max-w-sm space-y-1">
                      {selectedStageArtifacts.jsons.map((artifact) => (
                        <a
                          key={artifact.name}
                          href={artifact.url}
                          target="_blank"
                          rel="noreferrer"
                          className="block truncate rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-1.5 text-xs text-[var(--accent)]"
                        >
                          {artifact.name}
                        </a>
                      ))}
                    </div>
                  ) : null}
                </div>
              )}
              {isResuming ? (
                <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 bg-[var(--bg-canvas)]/70 backdrop-blur-sm">
                  <Loader2 className="h-10 w-10 animate-spin text-[var(--accent)]" />
                  <div className="text-sm font-semibold">Resuming pipeline…</div>
                  <div className="text-xs text-[var(--text-secondary)]">
                    {(activeJob.current_stage ?? 'Preparing').replaceAll('_', ' ')}
                  </div>
                </div>
              ) : null}
            </div>

            {selectedStageArtifacts.images.length > 1 ? (
              <div className="flex flex-wrap gap-2 px-4">
                {selectedStageArtifacts.images.map((artifact) => (
                  <button
                    key={artifact.name}
                    type="button"
                    onClick={() => setCanvasOverrideName(artifact.name)}
                    className={`max-w-full truncate rounded-full border px-3 py-1 text-xs font-semibold ${
                      canvasArtifact?.name === artifact.name
                        ? 'border-[var(--accent)] bg-[var(--accent)]/10 text-[var(--accent)]'
                        : 'border-[var(--border-muted)] bg-[var(--bg-primary)] text-[var(--text-secondary)]'
                    }`}
                  >
                    {artifact.name}
                  </button>
                ))}
              </div>
            ) : null}
              </>
            )}
          </main>
        </div>
      )}
    </div>
  )
}
