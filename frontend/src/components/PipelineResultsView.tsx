import { useEffect, useMemo, useState } from 'react'
import { Maximize2, X } from 'lucide-react'
import type { DetectedObject, PipelineJob, PipelineReviewBucket, PipelineReviewDecision, PipelineReviewItem, PipelineStageManifest } from '@/types'
import { PipelineArtifactCanvas } from '@/components/PipelineArtifactCanvas'
import { PipelineHitlReviewView } from '@/components/PipelineHitlReviewView'
import { getPipelineJob, getPipelineReviewedGraph, getPipelineReviewedQa, getPipelineStageStatus, putPipelineArtifact, resumePipelineFromStage } from '@/lib/api'

type JsonValue = string | number | boolean | null | JsonObject | JsonValue[]
type JsonObject = Record<string, JsonValue>
type ReviewBucket = PipelineReviewBucket
type ReviewDecision = PipelineReviewDecision

const REVIEW_STORAGE_PREFIX = 'garnet-pipeline-review'
const EQUIPMENT_CLASSES = new Set(['pump', 'heat exchanger', 'tank', 'vessel', 'column', 'compressor', 'blower', 'fan'])

function reviewStorageKey(jobId: string) {
  return `${REVIEW_STORAGE_PREFIX}:${jobId}`
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
      const className = toStringValue(item.class_name)?.toLowerCase()
      return className ? EQUIPMENT_CLASSES.has(className) : false
    })

  return sourceItems.map((item, index) => {
    const id = toStringValue(item.id) ?? `equip_${String(index + 1).padStart(3, '0')}`
    const className = toStringValue(item.class_name) ?? 'vessel'
    const bbox = toJsonObject(item.bbox)
    return {
      bucket: 'stage3_equipment',
      id,
      title: `${className} ${id}`,
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

function equipmentObjectsToStage3Artifact(objects: DetectedObject[]) {
  return {
    equipment: objects.map((obj, index) => ({
      id: obj.Text?.trim() || `equip_${String(index + 1).padStart(3, '0')}`,
      class_name: obj.Object || 'vessel',
      bbox: {
        x_min: Math.round(obj.Left),
        y_min: Math.round(obj.Top),
        x_max: Math.round(obj.Left + obj.Width),
        y_max: Math.round(obj.Top + obj.Height),
      },
      source: 'hitl',
      review_state: 'accepted',
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
  const [isResuming, setIsResuming] = useState(false)
  const [pipelineActionError, setPipelineActionError] = useState<string | null>(null)
  const [graphMode, setGraphMode] = useState<'raw' | 'reviewed'>('raw')
  const [reviewedGraphSummary, setReviewedGraphSummary] = useState<JsonObject | null>(null)
  const [reviewedQaSummary, setReviewedQaSummary] = useState<JsonObject | null>(null)
  const stages = stageStatuses.length ? stageStatuses : activeJob.manifest?.stages ?? []
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
      stage4_line_number: buildReviewItems('stage4_line_numbers.json', jsonDetails['stage4_line_numbers.json']),
      stage4_instrument: buildReviewItems('stage4_instrument_tags.json', jsonDetails['stage4_instrument_tags.json']),
      stage12_line_attachment: buildReviewItems('stage12_text_attachments.json', jsonDetails['stage12_text_attachments.json']),
      stage12_instrument_attachment: buildReviewItems('stage12_instrument_tag_attachments.json', jsonDetails['stage12_instrument_tag_attachments.json']),
    }),
    [jsonDetails]
  )

  const reviewCounts = useMemo(() => {
    const counts: Record<ReviewBucket, Record<ReviewDecision, number>> = {
      stage3_equipment: { accepted: 0, rejected: 0, deferred: 0 },
      stage4_line_number: { accepted: 0, rejected: 0, deferred: 0 },
      stage4_instrument: { accepted: 0, rejected: 0, deferred: 0 },
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

  const staleFromStage5b = stages.some((stage) => stage.name === 'stage5b_pipe_trace' && stage.status === 'stale')
    || stages.some((stage) => stage.status === 'stale' && (stage.num ?? 0) >= 5)

  const saveStage3Equipment = async (objects: DetectedObject[]) => {
    setPipelineActionError(null)
    const payload = equipmentObjectsToStage3Artifact(objects)
    const response = await putPipelineArtifact(activeJob.job_id, 'stage3_equipment_bboxes.json', payload)
    setStageStatuses(response.stages)
    const refreshedJob = await getPipelineJob(activeJob.job_id)
    setLiveJob(refreshedJob)
  }

  const resumeFromStage5b = async () => {
    setIsResuming(true)
    setPipelineActionError(null)
    try {
      await resumePipelineFromStage(activeJob.job_id, 'stage5b_pipe_trace')
      while (true) {
        const nextJob = await getPipelineJob(activeJob.job_id)
        setLiveJob(nextJob)
        try {
          const statusPayload = await getPipelineStageStatus(activeJob.job_id)
          setStageStatuses(statusPayload.stages)
        } catch {
          setStageStatuses(nextJob.manifest?.stages ?? [])
        }
        if (nextJob.status === 'completed') break
        if (nextJob.status === 'failed') {
          setPipelineActionError(nextJob.error || 'Pipeline resume failed')
          break
        }
        await new Promise((resolve) => window.setTimeout(resolve, 500))
      }
    } catch (error) {
      setPipelineActionError(error instanceof Error ? error.message : 'Pipeline resume failed')
    } finally {
      setIsResuming(false)
    }
  }

  if (workspaceOpen) {
    return (
      <PipelineHitlReviewView
        jobId={activeJob.job_id}
        activeBucket={activeReviewBucket}
        itemsByBucket={reviewItems}
        imageArtifacts={imageArtifacts}
        initialReviewDecisions={reviewDecisions}
        onApply={(decisions) => setReviewDecisions(decisions)}
        onSaveStage3Equipment={saveStage3Equipment}
        onClose={() => setWorkspaceOpen(false)}
      />
    )
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
    <div className="h-full overflow-auto bg-[var(--bg-canvas)]">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 py-6">
        <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
          <div className="text-lg font-semibold">Pipeline Review</div>
          <div className="mt-1 text-sm text-[var(--text-secondary)]">
            Full staged review through Stage 11: normalization, OCR, object detection, pipe mask, path tracing, trace associations, graph assembly, QA, reviewed outputs, process exports, and final connection overlay.
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
            ['S4 Line Accepted', reviewCounts.stage4_line_number.accepted],
            ['S4 Instrument Accepted', reviewCounts.stage4_instrument.accepted],
            ['S12 Line Accepted', reviewCounts.stage12_line_attachment.accepted],
            ['S12 Instrument Accepted', reviewCounts.stage12_instrument_attachment.accepted],
          ]}
        />

        <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
          <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
            <div className="text-sm font-semibold">Stages</div>
            <div className="mt-4 space-y-3">
              {stages.map((stage) => (
                <div key={stage.name} className={`rounded-xl border p-3 ${stageStatusClass(stage.status)}`}>
                  <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Stage {stage.num}</div>
                  <div className="mt-1 text-sm font-semibold">{stage.name}</div>
                  <div className="mt-1 text-xs text-[var(--text-secondary)]">
                    {stage.status} {stage.duration_sec !== undefined ? `• ${stage.duration_sec.toFixed(3)}s` : ''}
                    {stage.stale_source_artifact ? ` • from ${stage.stale_source_artifact}` : ''}
                  </div>
                </div>
              ))}
            </div>
            {staleFromStage5b ? (
              <button
                type="button"
                onClick={resumeFromStage5b}
                disabled={isResuming}
                className="mt-4 w-full rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm font-semibold text-amber-700 disabled:opacity-40"
              >
                {isResuming ? 'Resuming...' : 'Resume from Stage 5b'}
              </button>
            ) : null}
            {pipelineActionError ? <div className="mt-3 text-xs text-red-600">{pipelineActionError}</div> : null}
          </div>

          <div className="space-y-6">
            <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
              <div className="text-sm font-semibold">Review Flow</div>
              <div className="mt-3">
                <button
                  type="button"
                  onClick={() => setWorkspaceOpen(true)}
                  className="rounded-lg border border-[var(--accent)] bg-[var(--accent)]/10 px-3 py-2 text-sm font-semibold text-[var(--accent)]"
                >
                  Open Full Review Workspace
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
                  ['stage4_line_number', 'Stage 4 Line Numbers'],
                  ['stage4_instrument', 'Stage 4 Instruments'],
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
      </div>
    </div>
  )
}
