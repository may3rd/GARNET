import { useEffect, useMemo, useState } from 'react'
import type { PipelineJob, PipelineReviewBucket, PipelineReviewDecision, PipelineReviewItem } from '@/types'
import { PipelineArtifactCanvas } from '@/components/PipelineArtifactCanvas'
import { PipelineHitlReviewView } from '@/components/PipelineHitlReviewView'

type JsonValue = string | number | boolean | null | JsonObject | JsonValue[]
type JsonObject = Record<string, JsonValue>
type ReviewFilter = 'all' | ReviewDecision | 'unresolved'

const REVIEW_STORAGE_PREFIX = 'garnet-pipeline-review'

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

function toHighlightBox(bbox: JsonObject | undefined) {
  if (!bbox) return null
  const xMin = toNumber(bbox.x_min)
  const yMin = toNumber(bbox.y_min)
  const xMax = toNumber(bbox.x_max)
  const yMax = toNumber(bbox.y_max)
  if ([xMin, yMin, xMax, yMax].some((value) => value === undefined)) return null
  return {
    xMin: xMin as number,
    yMin: yMin as number,
    xMax: xMax as number,
    yMax: yMax as number,
  }
}

function buildReviewItems(
  payloadName: string,
  payload: JsonObject | undefined
): ReviewItem[] {
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
  const [jsonSummaries, setJsonSummaries] = useState<Record<string, JsonObject>>({})
  const [jsonDetails, setJsonDetails] = useState<Record<string, JsonObject>>({})
  const [activeArtifactName, setActiveArtifactName] = useState<string | null>(null)
  const [activeReviewBucket, setActiveReviewBucket] = useState<ReviewBucket>('stage4_line_number')
  const [activeReviewFilter, setActiveReviewFilter] = useState<ReviewFilter>('all')
  const [selectedReviewItemId, setSelectedReviewItemId] = useState<string | null>(null)
  const [reviewDecisions, setReviewDecisions] = useState<Record<string, ReviewDecision>>({})
  const [workspaceOpen, setWorkspaceOpen] = useState(false)
  const stages = job.manifest?.stages ?? []
  const imageArtifacts = useMemo(
    () => job.artifacts.filter((artifact) => /\.(png|jpg|jpeg|webp)$/i.test(artifact.name)),
    [job.artifacts]
  )
  const jsonArtifacts = useMemo(
    () => job.artifacts.filter((artifact) => artifact.name.endsWith('.json')),
    [job.artifacts]
  )
  const route = job.manifest?.ocr_route ?? job.ocr_route
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
          'stage13_graph_qa_summary.json',
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
    const raw = window.localStorage.getItem(reviewStorageKey(job.job_id))
    if (!raw) {
      setReviewDecisions({})
      return
    }
    try {
      setReviewDecisions(JSON.parse(raw) as Record<string, ReviewDecision>)
    } catch {
      setReviewDecisions({})
    }
  }, [job.job_id])

  useEffect(() => {
    window.localStorage.setItem(reviewStorageKey(job.job_id), JSON.stringify(reviewDecisions))
  }, [job.job_id, reviewDecisions])

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

  const activeArtifact = imageArtifacts.find((artifact) => artifact.name === activeArtifactName) ?? imageArtifacts[0] ?? null
  const reviewItems = useMemo(
    () => ({
      stage4_line_number: buildReviewItems('stage4_line_numbers.json', jsonDetails['stage4_line_numbers.json']),
      stage4_instrument: buildReviewItems('stage4_instrument_tags.json', jsonDetails['stage4_instrument_tags.json']),
      stage12_line_attachment: buildReviewItems('stage12_text_attachments.json', jsonDetails['stage12_text_attachments.json']),
      stage12_instrument_attachment: buildReviewItems('stage12_instrument_tag_attachments.json', jsonDetails['stage12_instrument_tag_attachments.json']),
    }),
    [jsonDetails]
  )

  const activeReviewItems = reviewItems[activeReviewBucket]
  const filteredReviewItems = useMemo(() => {
    return activeReviewItems.filter((item) => {
      if (activeReviewFilter === 'all') return true
      if (activeReviewFilter === 'unresolved') {
        return item.reviewState === 'detection_only' || item.reviewState === 'rejected' || !item.reviewState
      }
      return (reviewDecisions[`${item.bucket}:${item.id}`] ?? 'deferred') === activeReviewFilter
    })
  }, [activeReviewFilter, activeReviewItems, reviewDecisions])
  const selectedReviewItem =
    filteredReviewItems.find((item) => item.id === selectedReviewItemId) ??
    filteredReviewItems[0] ??
    null

  useEffect(() => {
    if (!filteredReviewItems.length) {
      setSelectedReviewItemId(null)
      return
    }
    if (selectedReviewItemId && filteredReviewItems.some((item) => item.id === selectedReviewItemId)) {
      return
    }
    setSelectedReviewItemId(filteredReviewItems[0].id)
  }, [filteredReviewItems, selectedReviewItemId])

  useEffect(() => {
    if (!selectedReviewItem) return
    setActiveArtifactName(selectedReviewItem.artifactName)
  }, [selectedReviewItem])

  const reviewCounts = useMemo(() => {
    const counts: Record<ReviewBucket, Record<ReviewDecision, number>> = {
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

  const setDecision = (bucket: ReviewBucket, itemId: string, decision: ReviewDecision) => {
    setReviewDecisions((current) => ({
      ...current,
      [`${bucket}:${itemId}`]: decision,
    }))
  }

  if (workspaceOpen) {
    return (
      <PipelineHitlReviewView
        jobId={job.job_id}
        activeBucket={activeReviewBucket}
        itemsByBucket={reviewItems}
        imageArtifacts={imageArtifacts}
        initialReviewDecisions={reviewDecisions}
        onApply={(decisions) => setReviewDecisions(decisions)}
        onClose={() => setWorkspaceOpen(false)}
      />
    )
  }

  return (
    <div className="h-full overflow-auto bg-[var(--bg-canvas)]">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 py-6">
        <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
          <div className="text-lg font-semibold">Pipeline Review</div>
          <div className="mt-1 text-sm text-[var(--text-secondary)]">
            Full staged review through Stage 13: normalization, OCR, object detection, pipe mask, sealing, skeleton, nodes, clustering, edge tracing, junction review, graph assembly, and graph QA.
          </div>
          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
              <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Job</div>
              <div className="mt-1 font-mono text-xs">{job.job_id}</div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
              <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Status</div>
              <div className="mt-1 text-sm font-semibold capitalize">{job.status}</div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
              <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Current Stage</div>
              <div className="mt-1 text-sm font-semibold">{job.current_stage ?? 'Queued'}</div>
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
            ['Nodes', jsonSummaries['stage12_graph_summary.json']?.node_count],
            ['Edges', jsonSummaries['stage12_graph_summary.json']?.edge_count],
            ['Components', jsonSummaries['stage12_graph_summary.json']?.connected_component_count],
            ['Unresolved Junctions', jsonSummaries['stage12_graph_summary.json']?.unresolved_junction_count],
          ]}
        />

        <SummaryCard
          title="QA Summary"
          entries={[
            ['QA Components', jsonSummaries['stage13_graph_qa_summary.json']?.connected_component_count],
            ['Articulation Points', jsonSummaries['stage13_graph_qa_summary.json']?.articulation_point_count],
            ['Isolated Nodes', jsonSummaries['stage13_graph_qa_summary.json']?.isolated_node_count],
            ['Review Queue', jsonSummaries['stage13_graph_qa_summary.json']?.review_queue_count],
          ]}
        />

        <SummaryCard
          title="HITL Review"
          entries={[
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
                <div key={stage.name} className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
                  <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Stage {stage.num}</div>
                  <div className="mt-1 text-sm font-semibold">{stage.name}</div>
                  <div className="mt-1 text-xs text-[var(--text-secondary)]">
                    {stage.status} {stage.duration_sec !== undefined ? `• ${stage.duration_sec.toFixed(3)}s` : ''}
                  </div>
                </div>
              ))}
            </div>
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
              <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                {([
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

            {activeArtifact && (
              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
                <div className="text-sm font-semibold">Artifact Viewer</div>
                <div className="mt-1 text-xs text-[var(--text-secondary)]">
                  Blue = sheet OCR, green = crop OCR, cyan = rotated crop OCR, orange = detection only, red = rejected.
                </div>
                <div className="mt-4">
                  <PipelineArtifactCanvas
                    imageUrl={activeArtifact.url}
                    title={activeArtifact.name}
                    highlightBox={selectedReviewItem ? toHighlightBox(selectedReviewItem.bbox) : null}
                  />
                </div>
              </div>
            )}

            <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
                <div className="text-sm font-semibold">Review Items</div>
                <div className="mt-4 flex flex-wrap gap-2">
                  {([
                    ['all', 'All'],
                    ['accepted', 'Accepted'],
                    ['rejected', 'Rejected'],
                    ['deferred', 'Deferred'],
                    ['unresolved', 'Unresolved'],
                  ] as Array<[ReviewFilter, string]>).map(([filterKey, label]) => {
                    const isActive = filterKey === activeReviewFilter
                    return (
                      <button
                        key={filterKey}
                        type="button"
                        onClick={() => setActiveReviewFilter(filterKey)}
                        className={`rounded-full border px-3 py-1 text-xs font-semibold transition ${
                          isActive
                            ? 'border-[var(--accent)] bg-[var(--accent)]/10 text-[var(--accent)]'
                            : 'border-[var(--border-muted)] bg-[var(--bg-primary)] text-[var(--text-secondary)]'
                        }`}
                      >
                        {label}
                      </button>
                    )
                  })}
                </div>
                <div className="mt-4 max-h-[420px] space-y-2 overflow-auto pr-1">
                  {filteredReviewItems.map((item) => {
                    const isSelected = item.id === selectedReviewItem?.id
                    const decision = reviewDecisions[`${item.bucket}:${item.id}`] ?? 'deferred'
                    return (
                      <button
                        key={`${item.bucket}:${item.id}`}
                        type="button"
                        onClick={() => setSelectedReviewItemId(item.id)}
                        className={`w-full rounded-xl border p-3 text-left transition ${
                          isSelected
                            ? 'border-[var(--accent)] bg-[var(--bg-primary)] ring-2 ring-[var(--accent)]/25'
                            : 'border-[var(--border-muted)] bg-[var(--bg-primary)] hover:border-[var(--accent)]/50'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <div className="truncate text-sm font-semibold">{item.title}</div>
                            <div className="mt-1 text-xs text-[var(--text-secondary)]">{item.subtitle}</div>
                            {item.statusHint ? (
                              <div className="mt-1 truncate text-xs text-[var(--text-secondary)]">{item.statusHint}</div>
                            ) : null}
                          </div>
                          <div className="rounded-full border border-[var(--border-muted)] px-2 py-0.5 text-[10px] uppercase tracking-wide text-[var(--text-secondary)]">
                            {decision}
                          </div>
                        </div>
                      </button>
                    )
                  })}
                  {!filteredReviewItems.length ? (
                    <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-4 text-sm text-[var(--text-secondary)]">
                      No review items match the current filter.
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
                <div className="text-sm font-semibold">Selected Item</div>
                {selectedReviewItem ? (
                  <div className="mt-4 space-y-4">
                    <div>
                      <div className="text-sm font-semibold">{selectedReviewItem.title}</div>
                      <div className="mt-1 text-xs text-[var(--text-secondary)]">{selectedReviewItem.subtitle}</div>
                    </div>
                    <div className="grid gap-2 text-xs text-[var(--text-secondary)]">
                      <div><span className="font-semibold text-[var(--text-primary)]">Text:</span> {selectedReviewItem.text || 'n/a'}</div>
                      <div><span className="font-semibold text-[var(--text-primary)]">Normalized:</span> {selectedReviewItem.normalizedText || 'n/a'}</div>
                      <div><span className="font-semibold text-[var(--text-primary)]">OCR Source:</span> {selectedReviewItem.ocrSource || 'n/a'}</div>
                      <div><span className="font-semibold text-[var(--text-primary)]">Review State:</span> {selectedReviewItem.reviewState || 'n/a'}</div>
                      <div><span className="font-semibold text-[var(--text-primary)]">Source Object:</span> {selectedReviewItem.sourceObjectId || 'n/a'}</div>
                      <div><span className="font-semibold text-[var(--text-primary)]">Edge:</span> {selectedReviewItem.edgeId || 'n/a'}</div>
                      <div><span className="font-semibold text-[var(--text-primary)]">Distance:</span> {selectedReviewItem.distancePx ?? 'n/a'}</div>
                      <div><span className="font-semibold text-[var(--text-primary)]">Threshold:</span> {selectedReviewItem.thresholdPx ?? 'n/a'}</div>
                    </div>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => setDecision(selectedReviewItem.bucket, selectedReviewItem.id, 'accepted')}
                        className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-xs font-semibold text-emerald-600"
                      >
                        Accept
                      </button>
                      <button
                        type="button"
                        onClick={() => setDecision(selectedReviewItem.bucket, selectedReviewItem.id, 'rejected')}
                        className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs font-semibold text-red-600"
                      >
                        Reject
                      </button>
                      <button
                        type="button"
                        onClick={() => setDecision(selectedReviewItem.bucket, selectedReviewItem.id, 'deferred')}
                        className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-xs font-semibold text-[var(--text-secondary)]"
                      >
                        Defer
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="mt-4 text-sm text-[var(--text-secondary)]">Select an item to review.</div>
                )}
              </div>
            </div>

            <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
              <div className="text-sm font-semibold">Artifact Thumbnails</div>
              <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {imageArtifacts.map((artifact) => {
                  const isActive = artifact.name === activeArtifactName
                  return (
                    <button
                      key={artifact.name}
                      type="button"
                      onClick={() => setActiveArtifactName(artifact.name)}
                      className={`rounded-xl border bg-[var(--bg-primary)] p-3 text-left transition ${
                        isActive
                          ? 'border-[var(--accent)] ring-2 ring-[var(--accent)]/25'
                          : 'border-[var(--border-muted)] hover:border-[var(--accent)]/50'
                      }`}
                    >
                      <div className="mb-2 text-xs font-semibold text-[var(--text-secondary)]">{artifact.name}</div>
                      <img src={artifact.url} alt={artifact.name} className="w-full rounded-lg border border-[var(--border-muted)]" />
                    </button>
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
