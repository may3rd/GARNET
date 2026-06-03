import { Fragment, useEffect, useMemo, useState } from 'react'
import { Image as KonvaImage, Layer, Line, Stage, Text } from 'react-konva'
import type { PipelineReviewDecision } from '@/types'

type JsonValue = string | number | boolean | null | JsonObject | JsonValue[]
type JsonObject = Record<string, JsonValue>

type Stage6TraceEdge = {
  trace_id: string
  trace_kind?: string
  source_obj_id?: string
  source_obj_type?: string
  terminal_type?: string
  trace_length_px?: number
  segments?: JsonObject[]
  polyline?: number[][]
  attachments?: {
    line_numbers?: JsonObject[]
  }
}

type TraceDraft = {
  decision: PipelineReviewDecision
  lineText: string
}

type Stage6LineAssociationReviewProps = {
  tracePayload?: JsonObject
  reviewPayload?: JsonObject
  overlayUrl?: string
  stage7Stale: boolean
  isSaving: boolean
  isResuming: boolean
  onSave: (payload: JsonObject) => Promise<void>
  onResumeStage7: () => Promise<void>
}

function asObject(value: JsonValue | undefined): JsonObject | undefined {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as JsonObject) : undefined
}

function asString(value: JsonValue | undefined): string {
  return typeof value === 'string' ? value : ''
}

function asNumber(value: JsonValue | undefined): number | undefined {
  return typeof value === 'number' ? value : undefined
}

function asObjectArray(value: JsonValue | undefined): JsonObject[] {
  return Array.isArray(value) ? value.filter((item): item is JsonObject => Boolean(asObject(item as JsonValue))) : []
}

function traceEdgesFromPayload(payload?: JsonObject): Stage6TraceEdge[] {
  return asObjectArray(payload?.trace_edges).map((edge, index) => {
    const attachments = asObject(edge.attachments)
    return {
      trace_id: asString(edge.trace_id) || `trace_${index + 1}`,
      trace_kind: asString(edge.trace_kind),
      source_obj_id: asString(edge.source_obj_id),
      source_obj_type: asString(edge.source_obj_type),
      terminal_type: asString(edge.terminal_type),
      trace_length_px: asNumber(edge.trace_length_px),
      segments: asObjectArray(edge.segments),
      polyline: Array.isArray(edge.polyline)
        ? edge.polyline
          .filter((point): point is JsonValue[] => Array.isArray(point))
          .map((point) => [Number(point[0]), Number(point[1])])
          .filter((point) => Number.isFinite(point[0]) && Number.isFinite(point[1]))
        : [],
      attachments: {
        line_numbers: asObjectArray(attachments?.line_numbers),
      },
    }
  })
}

function edgePoints(edge: Stage6TraceEdge): number[] {
  const points = edge.polyline?.length
    ? edge.polyline
    : (edge.segments ?? []).flatMap((segment, index) => {
      const x1 = Number(segment.x1)
      const y1 = Number(segment.y1)
      const x2 = Number(segment.x2)
      const y2 = Number(segment.y2)
      if (![x1, y1, x2, y2].every(Number.isFinite)) return []
      return index === 0 ? [[x1, y1], [x2, y2]] : [[x2, y2]]
    })
  return points.flatMap((point) => [point[0], point[1]])
}

function firstLineText(edge: Stage6TraceEdge): string {
  const line = edge.attachments?.line_numbers?.[0]
  return asString(line?.normalized_text) || asString(line?.text) || asString(line?.id)
}

function reviewAcceptedByTrace(reviewPayload?: JsonObject): Map<string, JsonObject> {
  const result = new Map<string, JsonObject>()
  for (const item of asObjectArray(reviewPayload?.accepted)) {
    const traceId = asString(item.trace_id)
    if (traceId) result.set(traceId, item)
  }
  return result
}

function reviewRejectedTraceIds(reviewPayload?: JsonObject): Set<string> {
  const result = new Set<string>()
  for (const item of asObjectArray(reviewPayload?.needs_review)) {
    const traceId = asString(item.trace_id) || asString(item.id)
    if (traceId) result.add(traceId)
  }
  return result
}

function buildInitialDrafts(edges: Stage6TraceEdge[], reviewPayload?: JsonObject): Record<string, TraceDraft> {
  const accepted = reviewAcceptedByTrace(reviewPayload)
  const rejected = reviewRejectedTraceIds(reviewPayload)
  const drafts: Record<string, TraceDraft> = {}
  for (const edge of edges) {
    const reviewed = accepted.get(edge.trace_id)
    const lineText = reviewed
      ? (asString(reviewed.normalized_text) || asString(reviewed.text) || asString(reviewed.id))
      : firstLineText(edge)
    drafts[edge.trace_id] = {
      decision: reviewed ? 'accepted' : rejected.has(edge.trace_id) ? 'rejected' : lineText ? 'deferred' : 'deferred',
      lineText,
    }
  }
  return drafts
}

function draftColor(draft: TraceDraft | undefined): string {
  if (!draft?.lineText.trim()) return '#ef4444'
  if (draft.decision === 'accepted') return '#16a34a'
  if (draft.decision === 'rejected') return '#f97316'
  return '#eab308'
}

function lineAssociation(traceId: string, edge: Stage6TraceEdge, draft: TraceDraft): JsonObject {
  const text = draft.lineText.trim()
  return {
    id: text || `${traceId}:missing_line_number`,
    source_object_id: text || `${traceId}:missing_line_number`,
    class_name: 'line_number',
    text,
    normalized_text: text,
    trace_id: traceId,
    trace_kind: edge.trace_kind ?? '',
    source: 'hitl',
    review_state: draft.decision === 'accepted' ? 'accepted' : 'needs_review',
    review_source: 'human',
    review_required: draft.decision !== 'accepted',
  }
}

export function Stage6LineAssociationReview({
  tracePayload,
  reviewPayload,
  overlayUrl,
  stage7Stale,
  isSaving,
  isResuming,
  onSave,
  onResumeStage7,
}: Stage6LineAssociationReviewProps) {
  const edges = useMemo(() => traceEdgesFromPayload(tracePayload), [tracePayload])
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, TraceDraft>>({})
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null)
  const [imageSize, setImageSize] = useState({ width: 1200, height: 800 })

  useEffect(() => {
    setDrafts(buildInitialDrafts(edges, reviewPayload))
    setSelectedTraceId((current) => current && edges.some((edge) => edge.trace_id === current) ? current : edges[0]?.trace_id ?? null)
  }, [edges, reviewPayload])

  useEffect(() => {
    if (!overlayUrl) return
    const img = new window.Image()
    img.onload = () => {
      setImageEl(img)
      setImageSize({ width: img.naturalWidth, height: img.naturalHeight })
    }
    img.src = overlayUrl
  }, [overlayUrl])

  const selectedEdge = edges.find((edge) => edge.trace_id === selectedTraceId) ?? null
  const selectedDraft = selectedTraceId ? drafts[selectedTraceId] : undefined
  const counts = useMemo(() => {
    let accepted = 0
    let rejected = 0
    let missing = 0
    let deferred = 0
    for (const edge of edges) {
      const draft = drafts[edge.trace_id]
      if (!draft?.lineText.trim()) missing += 1
      if (draft?.decision === 'accepted') accepted += 1
      else if (draft?.decision === 'rejected') rejected += 1
      else deferred += 1
    }
    return { accepted, rejected, deferred, missing }
  }, [drafts, edges])

  const updateSelectedDraft = (patch: Partial<TraceDraft>) => {
    if (!selectedTraceId) return
    setDrafts((current) => ({
      ...current,
      [selectedTraceId]: {
        decision: current[selectedTraceId]?.decision ?? 'deferred',
        lineText: current[selectedTraceId]?.lineText ?? '',
        ...patch,
      },
    }))
  }

  const saveReview = async () => {
    const accepted: JsonObject[] = []
    const needsReview: JsonObject[] = []
    const tracesWithoutLineNumber: string[] = []

    for (const edge of edges) {
      const draft = drafts[edge.trace_id] ?? { decision: 'deferred' as PipelineReviewDecision, lineText: firstLineText(edge) }
      const association = lineAssociation(edge.trace_id, edge, draft)
      if (!draft.lineText.trim()) tracesWithoutLineNumber.push(edge.trace_id)
      if (draft.decision === 'accepted' && draft.lineText.trim()) {
        accepted.push(association)
      } else {
        needsReview.push({
          ...association,
          reason: draft.lineText.trim() ? 'not_accepted_by_reviewer' : 'missing_line_number',
        })
      }
    }

    await onSave({
      image_id: asString(tracePayload?.image_id),
      review_assumption: 'human_reviewed_stage6_line_associations',
      accepted,
      needs_review: needsReview,
      traces_without_line_number: tracesWithoutLineNumber,
    })
  }

  if (!tracePayload || !overlayUrl) {
    return (
      <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5 text-sm text-[var(--text-secondary)]">
        Stage 6 trace association artifacts are not available for this job.
      </div>
    )
  }

  return (
    <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="text-sm font-semibold">Stage 6 Line Association Review</div>
          <div className="mt-1 text-xs text-[var(--text-secondary)]">
            Click a traced path, edit its line number, then accept. Paths without line numbers stay visible as missing.
          </div>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="rounded-full bg-emerald-500/10 px-2 py-1 text-emerald-700">{counts.accepted} accepted</span>
          <span className="rounded-full bg-amber-500/10 px-2 py-1 text-amber-700">{counts.deferred} deferred</span>
          <span className="rounded-full bg-red-500/10 px-2 py-1 text-red-700">{counts.missing} missing</span>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="h-[68vh] min-h-[480px] overflow-auto rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)]">
          <Stage width={imageSize.width} height={imageSize.height}>
            <Layer>
              {imageEl ? <KonvaImage image={imageEl} width={imageSize.width} height={imageSize.height} /> : null}
              {edges.map((edge) => {
                const points = edgePoints(edge)
                if (points.length < 4) return null
                const draft = drafts[edge.trace_id]
                const selected = edge.trace_id === selectedTraceId
                const color = draftColor(draft)
                return (
                  <Fragment key={edge.trace_id}>
                    <Line
                      points={points}
                      stroke={color}
                      strokeWidth={selected ? 7 : 4}
                      lineCap="round"
                      lineJoin="round"
                      opacity={selected ? 1 : 0.72}
                    />
                    <Line
                      points={points}
                      stroke="rgba(0,0,0,0.01)"
                      strokeWidth={24}
                      lineCap="round"
                      lineJoin="round"
                      onClick={() => setSelectedTraceId(edge.trace_id)}
                      onTap={() => setSelectedTraceId(edge.trace_id)}
                    />
                    {points.length >= 2 && selected ? (
                      <Text
                        x={points[0] + 8}
                        y={points[1] - 24}
                        text={draft?.lineText || edge.trace_id}
                        fontSize={18}
                        fontStyle="bold"
                        fill={color}
                      />
                    ) : null}
                  </Fragment>
                )
              })}
            </Layer>
          </Stage>
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-4">
          {selectedEdge && selectedDraft ? (
            <>
              <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Selected Trace</div>
              <div className="mt-1 break-all font-mono text-sm font-semibold">{selectedEdge.trace_id}</div>
              <div className="mt-3 space-y-1 text-xs text-[var(--text-secondary)]">
                <div>Kind: {selectedEdge.trace_kind || 'unknown'}</div>
                <div>Source: {selectedEdge.source_obj_id || 'unknown'} {selectedEdge.source_obj_type ? `(${selectedEdge.source_obj_type})` : ''}</div>
                <div>Terminal: {selectedEdge.terminal_type || 'unknown'}</div>
                <div>Length: {selectedEdge.trace_length_px ?? 0}px</div>
              </div>

              <label className="mt-5 block text-xs font-semibold text-[var(--text-secondary)]">
                Line number
                <input
                  value={selectedDraft.lineText}
                  onChange={(event) => updateSelectedDraft({ lineText: event.target.value })}
                  className="mt-2 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent)]"
                  placeholder="e.g. 3”-Cul-25-002013-B1A2-NI"
                />
              </label>

              <div className="mt-4 grid grid-cols-3 gap-2">
                {(['accepted', 'rejected', 'deferred'] as PipelineReviewDecision[]).map((decision) => (
                  <button
                    key={decision}
                    type="button"
                    onClick={() => updateSelectedDraft({ decision })}
                    className={`rounded-lg border px-2 py-2 text-xs font-semibold capitalize ${
                      selectedDraft.decision === decision
                        ? 'border-[var(--accent)] bg-[var(--accent)]/10 text-[var(--accent)]'
                        : 'border-[var(--border-muted)] text-[var(--text-secondary)]'
                    }`}
                  >
                    {decision}
                  </button>
                ))}
              </div>
            </>
          ) : (
            <div className="text-sm text-[var(--text-secondary)]">Select a trace path to review.</div>
          )}

          <div className="mt-5 flex flex-col gap-2">
            <button
              type="button"
              onClick={saveReview}
              disabled={isSaving}
              className="rounded-lg border border-[var(--accent)] bg-[var(--accent)]/10 px-3 py-2 text-sm font-semibold text-[var(--accent)] disabled:opacity-40"
            >
              {isSaving ? 'Saving Stage 6 review...' : 'Save Stage 6 Line Review'}
            </button>
            {stage7Stale ? (
              <button
                type="button"
                onClick={onResumeStage7}
                disabled={isResuming}
                className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm font-semibold text-amber-700 disabled:opacity-40"
              >
                {isResuming ? 'Resuming...' : 'Resume from Stage 7'}
              </button>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}
