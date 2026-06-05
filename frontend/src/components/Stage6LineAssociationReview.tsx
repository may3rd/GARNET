import { Fragment, useEffect, useMemo, useState } from 'react'
import { CanvasView } from '@/components/CanvasView'
import type { DetectedObject, PipelineReviewDecision } from '@/types'

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
  baseImageUrl?: string
  overlayUrl?: string
  stage7Stale: boolean
  isSaving: boolean
  isResuming: boolean
  layout?: 'card' | 'workspace'
  onCancel?: () => void
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
    const reason = asString(item.reason)
    const reviewState = asString(item.review_state)
    if (traceId && (reason === 'rejected_by_reviewer' || reviewState === 'rejected')) result.add(traceId)
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
  if (draft?.decision === 'accepted') return '#16a34a'
  if (draft?.decision === 'rejected') return '#dc2626'
  if (!draft?.lineText.trim()) return '#ef4444'
  return '#a16207'
}

function labelPoint(points: number[]): { x: number; y: number } {
  if (points.length < 4) return { x: points[0] ?? 0, y: points[1] ?? 0 }
  let totalLength = 0
  const segments: Array<{ x1: number; y1: number; x2: number; y2: number; length: number }> = []
  for (let index = 0; index < points.length - 3; index += 2) {
    const x1 = points[index]
    const y1 = points[index + 1]
    const x2 = points[index + 2]
    const y2 = points[index + 3]
    const length = Math.hypot(x2 - x1, y2 - y1)
    if (!length) continue
    segments.push({ x1, y1, x2, y2, length })
    totalLength += length
  }
  const halfway = totalLength / 2
  let walked = 0
  for (const segment of segments) {
    if (walked + segment.length >= halfway) {
      const ratio = (halfway - walked) / segment.length
      return {
        x: segment.x1 + (segment.x2 - segment.x1) * ratio,
        y: segment.y1 + (segment.y2 - segment.y1) * ratio,
      }
    }
    walked += segment.length
  }
  return { x: points[0], y: points[1] }
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
    review_decision: draft.decision,
    review_source: 'human',
    review_required: draft.decision !== 'accepted',
  }
}

export function Stage6LineAssociationReview({
  tracePayload,
  reviewPayload,
  baseImageUrl,
  overlayUrl,
  stage7Stale,
  isSaving,
  isResuming,
  layout = 'card',
  onCancel,
  onSave,
  onResumeStage7,
}: Stage6LineAssociationReviewProps) {
  const edges = useMemo(() => traceEdgesFromPayload(tracePayload), [tracePayload])
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, TraceDraft>>({})
  const canvasObjects = useMemo<DetectedObject[]>(() => [], [])
  const canvasImageUrl = baseImageUrl || overlayUrl || ''

  useEffect(() => {
    setDrafts(buildInitialDrafts(edges, reviewPayload))
    setSelectedTraceId((current) => current && edges.some((edge) => edge.trace_id === current) ? current : edges[0]?.trace_id ?? null)
  }, [edges, reviewPayload])

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
          reason: draft.decision === 'rejected'
            ? 'rejected_by_reviewer'
            : draft.lineText.trim()
              ? 'not_accepted_by_reviewer'
              : 'missing_line_number',
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

  const actionButtons = (
    <div className="flex flex-wrap items-center gap-2">
      {onCancel ? (
        <button
          type="button"
          onClick={onCancel}
          className="inline-flex items-center gap-2 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm font-semibold text-[var(--text-primary)]"
        >
          Cancel
        </button>
      ) : null}
      <button
        type="button"
        onClick={saveReview}
        disabled={isSaving}
        className="inline-flex items-center justify-center rounded-lg border border-[var(--accent)] bg-[var(--accent)]/10 px-3 py-2 text-sm font-semibold text-[var(--accent)] disabled:opacity-40"
      >
        {isSaving ? 'Saving...' : 'Save review'}
      </button>
      {stage7Stale ? (
        <button
          type="button"
          onClick={onResumeStage7}
          disabled={isResuming}
          className="inline-flex items-center justify-center rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-sm font-semibold text-emerald-700 disabled:opacity-40"
        >
          {isResuming ? 'Resuming...' : 'Resume Stage 7'}
        </button>
      ) : null}
    </div>
  )

  if (!tracePayload || !canvasImageUrl) {
    if (layout === 'workspace') {
      return (
        <div className="flex h-full min-h-0 flex-col overflow-hidden bg-[var(--bg-primary)] text-[var(--text-primary)]">
          <div className="shrink-0 flex items-center justify-between border-b border-[var(--border-muted)] bg-[var(--bg-secondary)] px-6 py-4">
            <div>
              <div className="text-lg font-semibold">Pipeline HITL Review</div>
              <div className="text-xs text-[var(--text-secondary)]">Gate 3: Stage 6 association review.</div>
            </div>
            {actionButtons}
          </div>
          <div className="flex flex-1 items-center justify-center bg-[var(--bg-canvas)] text-sm text-[var(--text-secondary)]">
            Stage 6 trace association artifacts are not available for this job.
          </div>
        </div>
      )
    }
    return (
      <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5 text-sm text-[var(--text-secondary)]">
        Stage 6 trace association artifacts are not available for this job.
      </div>
    )
  }

  const reviewBody = (
    <>
      <main className="relative min-h-0 min-w-0 flex-1 overflow-hidden bg-[var(--bg-canvas)]">
        <CanvasView
          imageUrl={canvasImageUrl}
          objects={canvasObjects}
          selectedObjectKey={null}
          selectedObject={null}
          reviewStatus={{}}
          onSelectObject={(key) => {
            if (key === null) setSelectedTraceId(null)
          }}
          onSetReviewStatus={() => undefined}
          isEditing={false}
          editDraft={null}
          onStartEdit={() => undefined}
          onCancelEdit={() => undefined}
          onChangeEdit={() => undefined}
          onReplaceEditDraft={() => undefined}
          onSaveEdit={() => undefined}
          onDeleteSelected={() => undefined}
          onNavigatePrevious={() => undefined}
          onNavigateNext={() => undefined}
          isCreating={false}
          createDraft={null}
          onCreateDraftChange={() => undefined}
          fitKey={`stage6-association:${canvasImageUrl}`}
          imageOverlay={
            <svg className="pointer-events-none absolute inset-0 h-full w-full overflow-visible">
              {edges.map((edge) => {
                const points = edgePoints(edge)
                if (points.length < 4) return null
                const draft = drafts[edge.trace_id]
                const selected = edge.trace_id === selectedTraceId
                const color = draftColor(draft)
                const lineLabel = draft?.lineText.trim()
                const label = labelPoint(points)
                const labelWidth = lineLabel ? Math.max(96, lineLabel.length * 8.2 + 18) : 0
                const polylinePoints = points.reduce<string[]>((result, value, index) => {
                  if (index % 2 === 0) result.push(`${value},${points[index + 1]}`)
                  return result
                }, []).join(' ')
                return (
                  <Fragment key={edge.trace_id}>
                    {selected ? (
                      <polyline
                        points={polylinePoints}
                        fill="none"
                        stroke="#ffffff"
                        strokeWidth={13}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        opacity={0.9}
                      />
                    ) : null}
                    <polyline
                      points={polylinePoints}
                      fill="none"
                      stroke={color}
                      strokeWidth={selected ? 7 : 4}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      opacity={selected ? 1 : 0.72}
                    />
                    <polyline
                      className="pointer-events-auto cursor-pointer"
                      points={polylinePoints}
                      fill="none"
                      stroke="rgba(0,0,0,0.01)"
                      strokeWidth={24}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      onPointerDown={(event) => {
                        event.preventDefault()
                        event.stopPropagation()
                      }}
                      onPointerUp={(event) => {
                        event.preventDefault()
                        event.stopPropagation()
                      }}
                      onClick={(event) => {
                        event.preventDefault()
                        event.stopPropagation()
                        setSelectedTraceId(edge.trace_id)
                      }}
                    />
                    {lineLabel ? (
                      <g
                        className="pointer-events-auto cursor-pointer"
                        onPointerDown={(event) => {
                          event.preventDefault()
                          event.stopPropagation()
                        }}
                        onPointerUp={(event) => {
                          event.preventDefault()
                          event.stopPropagation()
                        }}
                        onClick={(event) => {
                          event.preventDefault()
                          event.stopPropagation()
                          setSelectedTraceId(edge.trace_id)
                        }}
                      >
                        <rect
                          x={label.x + 8}
                          y={label.y - 29}
                          width={labelWidth}
                          height={22}
                          rx={6}
                          fill={selected ? color : '#ffffff'}
                          stroke={color}
                          strokeWidth={selected ? 2 : 1.5}
                          opacity={selected ? 0.96 : 0.88}
                        />
                        <text
                          x={label.x + 17}
                          y={label.y - 13}
                          fontSize={13}
                          fontWeight={700}
                          fill={selected ? '#ffffff' : color}
                        >
                          {lineLabel}
                        </text>
                      </g>
                    ) : null}
                  </Fragment>
                )
              })}
            </svg>
          }
        />
      </main>

      <aside className={layout === 'workspace'
        ? 'min-h-0 w-[360px] shrink-0 overflow-auto border-l border-[var(--border-muted)] bg-[var(--bg-secondary)] p-6'
        : 'rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-4'}
      >
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
                className="mt-2 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent)]"
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

        {layout === 'card' ? (
          <div className="mt-5 flex flex-col gap-2">
            {actionButtons}
          </div>
        ) : null}
      </aside>
    </>
  )

  if (layout === 'workspace') {
    return (
      <div className="flex h-full min-h-0 flex-col overflow-hidden bg-[var(--bg-primary)] text-[var(--text-primary)]">
        <div className="shrink-0 flex items-center justify-between border-b border-[var(--border-muted)] bg-[var(--bg-secondary)] px-6 py-4">
          <div>
            <div className="text-lg font-semibold">Pipeline HITL Review</div>
            <div className="text-xs text-[var(--text-secondary)]">
              Gate 3: Stage 6 line association review using the detection-mode layout.
            </div>
          </div>
          {actionButtons}
        </div>
        <div className="shrink-0 border-b border-[var(--border-muted)] bg-[var(--bg-secondary)] px-6 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-[var(--accent)] bg-[var(--accent)]/10 px-3 py-1 text-xs font-semibold text-[var(--accent)]">
              Stage 6 Associations
            </span>
            <span className="ml-auto rounded-full bg-emerald-500/10 px-2 py-1 text-xs font-semibold text-emerald-700">{counts.accepted} accepted</span>
            <span className="rounded-full bg-amber-500/10 px-2 py-1 text-xs font-semibold text-amber-700">{counts.deferred} deferred</span>
            <span className="rounded-full bg-red-500/10 px-2 py-1 text-xs font-semibold text-red-700">{counts.missing} missing</span>
          </div>
        </div>
        <div className="relative flex min-h-0 flex-1">
          {reviewBody}
        </div>
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
        {reviewBody}
      </div>
    </div>
  )
}
