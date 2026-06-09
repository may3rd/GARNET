import { Fragment, useEffect, useMemo, useState } from 'react'
import { CanvasView } from '@/components/CanvasView'
import type { DetectedObject, GraphQaDecision } from '@/types'

type JsonValue = string | number | boolean | null | JsonObject | JsonValue[]
type JsonObject = Record<string, JsonValue>

type GraphQaItem = {
  id: string
  reviewItemType: string
  category: string
  severity: string
  priority: number
  message: string
  evidence: JsonObject
  geometry?: { x: number; y: number }
}

type ItemDraft = {
  decision: GraphQaDecision | null
  note?: string
}

type DraftState = Record<string, ItemDraft>

type GraphQaReviewViewProps = {
  reviewItemsPayload?: JsonObject
  reviewDecisionsPayload?: JsonObject
  overlayUrl?: string
  baseImageUrl?: string
  stage9Stale: boolean
  isSaving: boolean
  isResuming: boolean
  layout?: 'card' | 'workspace'
  onCancel?: () => void
  onSave: (payload: JsonObject) => Promise<void>
  onResumeStage9: () => Promise<void>
}

// ── decision constants ────────────────────────────────────────────────────────

const DECISION_LABELS: Record<GraphQaDecision, string> = {
  accept_as_is: 'Accept',
  false_positive: 'False Positive',
  defer: 'Defer',
  set_line_number: 'Set Line #',
}

// Which decisions apply for a given review_item_type
function decisionsForType(reviewItemType: string): GraphQaDecision[] {
  if (reviewItemType === 'line_number') {
    return ['accept_as_is', 'false_positive', 'defer', 'set_line_number']
  }
  return ['accept_as_is', 'false_positive', 'defer']
}

function decisionActiveClasses(decision: GraphQaDecision): string {
  if (decision === 'accept_as_is') return 'border-emerald-500 bg-emerald-500/10 text-emerald-700'
  if (decision === 'false_positive') return 'border-blue-500 bg-blue-500/10 text-blue-700'
  if (decision === 'set_line_number') return 'border-purple-500 bg-purple-500/10 text-purple-700'
  return 'border-amber-500 bg-amber-500/10 text-amber-700' // defer
}

function decisionShortLabel(decision: GraphQaDecision): string {
  if (decision === 'accept_as_is') return 'ACC'
  if (decision === 'false_positive') return 'FP'
  if (decision === 'set_line_number') return 'L#'
  return 'DEF'
}

// ── parse / build helpers ────────────────────────────────────────────────────

function parseGeometry(value: unknown): { x: number; y: number } | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  const r = value as Record<string, unknown>
  const x = typeof r.x === 'number' ? r.x : undefined
  const y = typeof r.y === 'number' ? r.y : undefined
  if (x === undefined || y === undefined) return undefined
  return { x, y }
}

function parseReviewItems(payload?: JsonObject): GraphQaItem[] {
  if (!payload) return []
  const raw = Array.isArray(payload.review_items) ? payload.review_items : []
  return raw.flatMap((item, index) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return []
    const r = item as JsonObject
    const id = typeof r.id === 'string' ? r.id : `item_${index + 1}`
    return [{
      id,
      reviewItemType: typeof r.review_item_type === 'string' ? r.review_item_type : 'unknown',
      category: typeof r.category === 'string' ? r.category : 'unknown',
      severity: typeof r.severity === 'string' ? r.severity : 'info',
      priority: typeof r.priority === 'number' ? r.priority : 0,
      message: typeof r.message === 'string' ? r.message : '',
      evidence: (r.evidence && typeof r.evidence === 'object' && !Array.isArray(r.evidence))
        ? r.evidence as JsonObject
        : {},
      geometry: parseGeometry(r.geometry),
    }]
  })
}

/** Rehydrate drafts from an existing stage8_review_decisions.json */
function buildInitialDrafts(payload?: JsonObject): DraftState {
  const drafts: DraftState = {}
  const decisions = Array.isArray(payload?.decisions) ? payload.decisions : []
  for (const d of decisions) {
    if (!d || typeof d !== 'object' || Array.isArray(d)) continue
    const item = d as JsonObject
    const id = typeof item.review_item_id === 'string' ? item.review_item_id : null
    const dec = typeof item.decision === 'string' ? item.decision as GraphQaDecision : null
    if (!id || !dec) continue
    drafts[id] = {
      decision: dec,
      note: typeof item.note === 'string' ? item.note : undefined,
    }
  }
  return drafts
}

/** Build the stage8_review_decisions.json payload from current drafts */
function buildDecisionsPayload(items: GraphQaItem[], drafts: DraftState): JsonObject {
  const decisions = items.flatMap((item) => {
    const draft = drafts[item.id]
    if (!draft?.decision) return []
    return [{
      review_item_id: item.id,
      decision: draft.decision,
      reviewer: 'human',
      ...(draft.note?.trim() ? { note: draft.note.trim() } : {}),
    }]
  })
  return { decisions }
}

// ── display helpers ───────────────────────────────────────────────────────────

function formatCategory(category: string): string {
  return category.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function severityClasses(severity: string, priority: number): string {
  if (severity === 'high' || priority >= 8) return 'bg-red-500/10 text-red-700 border-red-200'
  if (severity === 'review' || severity === 'medium' || priority >= 5) return 'bg-amber-500/10 text-amber-700 border-amber-200'
  return 'bg-yellow-500/10 text-yellow-700 border-yellow-200'
}

function markerColor(severity: string, priority: number, decision: GraphQaDecision | null | undefined): string {
  if (decision === 'accept_as_is') return '#16a34a'
  if (decision === 'false_positive') return '#2563eb'
  if (decision === 'set_line_number') return '#9333ea'
  if (decision === 'defer') return '#d97706'
  // undecided: severity color
  if (severity === 'high' || priority >= 8) return '#dc2626'
  if (severity === 'review' || severity === 'medium' || priority >= 5) return '#d97706'
  return '#ca8a04'
}

// ── component ────────────────────────────────────────────────────────────────

export function GraphQaReviewView({
  reviewItemsPayload,
  reviewDecisionsPayload,
  overlayUrl,
  baseImageUrl,
  stage9Stale,
  isSaving,
  isResuming,
  layout = 'card',
  onCancel,
  onSave,
  onResumeStage9,
}: GraphQaReviewViewProps) {
  const items = useMemo(() => parseReviewItems(reviewItemsPayload), [reviewItemsPayload])
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<DraftState>({})
  const canvasObjects = useMemo<DetectedObject[]>(() => [], [])
  const canvasImageUrl = overlayUrl || baseImageUrl || ''

  // Initialise selected item when items load
  useEffect(() => {
    setSelectedItemId((current) => {
      if (current && items.some((item) => item.id === current)) return current
      return items[0]?.id ?? null
    })
  }, [items])

  // Rehydrate drafts from existing decisions artifact on load/change
  useEffect(() => {
    setDrafts(buildInitialDrafts(reviewDecisionsPayload))
  }, [reviewDecisionsPayload])

  const selectedItem = items.find((item) => item.id === selectedItemId) ?? null
  const selectedDraft = selectedItemId ? (drafts[selectedItemId] ?? { decision: null }) : null

  const counts = useMemo(() => {
    let high = 0; let medium = 0; let info = 0; let decided = 0
    for (const item of items) {
      if (item.severity === 'high' || item.priority >= 8) high += 1
      else if (item.severity === 'review' || item.severity === 'medium' || item.priority >= 5) medium += 1
      else info += 1
      if (drafts[item.id]?.decision) decided += 1
    }
    return { total: items.length, high, medium, info, decided, undecided: items.length - decided }
  }, [items, drafts])

  const setDraft = (itemId: string, patch: Partial<ItemDraft>) => {
    setDrafts((current) => ({
      ...current,
      [itemId]: { decision: null, ...current[itemId], ...patch },
    }))
  }

  const saveReview = async () => {
    await onSave(buildDecisionsPayload(items, drafts))
  }

  const saveReviewAndResume = async () => {
    await onSave(buildDecisionsPayload(items, drafts))
    await onResumeStage9()
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
        disabled={isSaving || isResuming}
        className="inline-flex items-center justify-center rounded-lg border border-[var(--accent)] bg-[var(--accent)]/10 px-3 py-2 text-sm font-semibold text-[var(--accent)] disabled:opacity-40"
      >
        {isSaving ? 'Saving…' : 'Save review'}
      </button>
      {stage9Stale ? (
        <button
          type="button"
          onClick={saveReviewAndResume}
          disabled={isSaving || isResuming}
          className="inline-flex items-center justify-center rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-sm font-semibold text-emerald-700 disabled:opacity-40"
        >
          {isSaving ? 'Saving…' : isResuming ? 'Resuming…' : 'Save & Resume Stage 9'}
        </button>
      ) : null}
    </div>
  )

  // ── early-exit: no artifacts ───────────────────────────────────────────────

  if (!reviewItemsPayload && !canvasImageUrl) {
    if (layout === 'workspace') {
      return (
        <div className="flex h-full min-h-0 flex-col overflow-hidden bg-[var(--bg-primary)] text-[var(--text-primary)]">
          <div className="shrink-0 flex items-center justify-between border-b border-[var(--border-muted)] bg-[var(--bg-secondary)] px-6 py-4">
            <div>
              <div className="text-lg font-semibold">Pipeline HITL Review</div>
              <div className="text-xs text-[var(--text-secondary)]">Gate 4: Stage 8 graph QA review.</div>
            </div>
            {actionButtons}
          </div>
          <div className="flex flex-1 items-center justify-center bg-[var(--bg-canvas)] text-sm text-[var(--text-secondary)]">
            Stage 8 graph QA artifacts are not available for this job.
          </div>
        </div>
      )
    }
    return (
      <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5 text-sm text-[var(--text-secondary)]">
        Stage 8 graph QA artifacts are not available for this job.
      </div>
    )
  }

  // ── early-exit: no flagged items ──────────────────────────────────────────

  if (items.length === 0) {
    if (layout === 'workspace') {
      return (
        <div className="flex h-full min-h-0 flex-col overflow-hidden bg-[var(--bg-primary)] text-[var(--text-primary)]">
          <div className="shrink-0 flex items-center justify-between border-b border-[var(--border-muted)] bg-[var(--bg-secondary)] px-6 py-4">
            <div>
              <div className="text-lg font-semibold">Pipeline HITL Review</div>
              <div className="text-xs text-[var(--text-secondary)]">Gate 4: Stage 8 graph QA review.</div>
            </div>
            {actionButtons}
          </div>
          <div className="flex flex-1 flex-col items-center justify-center gap-3 bg-[var(--bg-canvas)]">
            <div className="text-base font-semibold text-[var(--text-primary)]">No graph QA items flagged</div>
            <div className="text-sm text-[var(--text-secondary)]">The graph passed QA checks. You can proceed to Stage 9.</div>
            <button
              type="button"
              onClick={saveReviewAndResume}
              disabled={isSaving || isResuming}
              className="mt-2 inline-flex items-center justify-center rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-700 disabled:opacity-40"
            >
              {isResuming ? 'Resuming…' : 'Resume Stage 9'}
            </button>
          </div>
        </div>
      )
    }
    return (
      <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
        <div className="text-sm font-semibold">Stage 8 Graph QA Review</div>
        <div className="mt-2 text-sm text-[var(--text-secondary)]">No graph QA items flagged. The graph passed QA checks.</div>
        <div className="mt-4 flex gap-2">{actionButtons}</div>
      </div>
    )
  }

  // ── main review body ─────────────────────────────────────────────────────

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
            if (key === null) setSelectedItemId(null)
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
          fitKey={`stage8-qa:${canvasImageUrl}`}
          imageOverlay={
            <svg className="pointer-events-none absolute inset-0 h-full w-full overflow-visible">
              {items.map((item) => {
                if (!item.geometry) return null
                const selected = item.id === selectedItemId
                const draft = drafts[item.id]
                const color = selected ? '#f97316' : markerColor(item.severity, item.priority, draft?.decision)
                return (
                  <Fragment key={item.id}>
                    <circle
                      cx={item.geometry.x}
                      cy={item.geometry.y}
                      r={selected ? 20 : 14}
                      fill={color}
                      fillOpacity={0.18}
                      stroke={color}
                      strokeWidth={selected ? 3 : 2}
                      className="pointer-events-auto cursor-pointer"
                      onClick={(event) => {
                        event.preventDefault()
                        event.stopPropagation()
                        setSelectedItemId(item.id)
                      }}
                    />
                    <circle
                      cx={item.geometry.x}
                      cy={item.geometry.y}
                      r={selected ? 7 : 5}
                      fill={color}
                      stroke="#ffffff"
                      strokeWidth={1.5}
                      pointerEvents="none"
                    />
                  </Fragment>
                )
              })}
            </svg>
          }
        />
      </main>

      <aside
        className={layout === 'workspace'
          ? 'min-h-0 w-[380px] shrink-0 overflow-auto border-l border-[var(--border-muted)] bg-[var(--bg-secondary)] p-4'
          : 'rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-4'}
      >
        {selectedItem && selectedDraft ? (
          <>
            {/* ── item meta ─────────────────────────────────────────────── */}
            <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Selected Item</div>
            <div className="mt-1 break-all font-mono text-xs text-[var(--text-secondary)]">{selectedItem.id}</div>
            <span className={`mt-2 inline-block rounded-full border px-2 py-0.5 text-xs font-semibold capitalize ${severityClasses(selectedItem.severity, selectedItem.priority)}`}>
              {selectedItem.severity}
            </span>
            <div className="mt-3 text-sm font-semibold">{formatCategory(selectedItem.category)}</div>
            <div className="mt-1 text-sm text-[var(--text-secondary)]">{selectedItem.message}</div>

            {/* ── evidence ──────────────────────────────────────────────── */}
            {Object.keys(selectedItem.evidence).length > 0 ? (
              <div className="mt-4">
                <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Evidence</div>
                <div className="mt-2 space-y-1">
                  {Object.entries(selectedItem.evidence).map(([key, val]) => (
                    <div key={key} className="flex gap-2 text-xs">
                      <span className="w-36 shrink-0 text-[var(--text-secondary)]">{key}</span>
                      <span className="min-w-0 break-all font-mono text-[var(--text-primary)]">
                        {Array.isArray(val) ? val.join(', ') : String(val ?? '')}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {/* ── decision chips ────────────────────────────────────────── */}
            <div className="mt-5">
              <div className="text-xs font-semibold uppercase tracking-wide text-[var(--text-secondary)]">Decision</div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {decisionsForType(selectedItem.reviewItemType).map((decision) => {
                  const active = selectedDraft.decision === decision
                  return (
                    <button
                      key={decision}
                      type="button"
                      onClick={() => setDraft(selectedItem.id, {
                        decision: active ? null : decision,
                      })}
                      className={`rounded-lg border px-2.5 py-1.5 text-xs font-semibold transition ${
                        active
                          ? decisionActiveClasses(decision)
                          : 'border-[var(--border-muted)] text-[var(--text-secondary)] hover:border-[var(--accent)]/50'
                      }`}
                    >
                      {DECISION_LABELS[decision]}
                    </button>
                  )
                })}
              </div>
              {selectedItem.reviewItemType === 'line_number' && selectedDraft.decision === 'set_line_number' ? (
                <div className="mt-2 rounded-lg border border-purple-200 bg-purple-500/5 px-3 py-2 text-xs text-purple-700">
                  Line number &amp; edge pickers will be added in the next phase. Save will record this decision.
                </div>
              ) : null}
            </div>

            {/* ── note field ────────────────────────────────────────────── */}
            <label className="mt-4 block">
              <span className="text-xs font-semibold uppercase tracking-wide text-[var(--text-secondary)]">Note (optional)</span>
              <textarea
                value={selectedDraft.note ?? ''}
                onChange={(event) => setDraft(selectedItem.id, { note: event.target.value })}
                rows={2}
                placeholder="Add a note for this decision…"
                className="mt-1.5 w-full resize-none rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-xs text-[var(--text-primary)] outline-none focus:border-[var(--accent)]"
              />
            </label>
          </>
        ) : (
          <div className="text-sm text-[var(--text-secondary)]">Select a flagged item to inspect evidence and set a decision.</div>
        )}

        {/* ── item list ─────────────────────────────────────────────────── */}
        <div className="mt-6">
          <div className="flex items-center justify-between">
            <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">All Items</div>
            <div className="text-xs text-[var(--text-secondary)]">
              {counts.decided} / {counts.total} decided
            </div>
          </div>
          <div className="mt-2 space-y-1">
            {items.map((item) => {
              const draft = drafts[item.id]
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setSelectedItemId(item.id)}
                  className={`w-full rounded-lg border px-3 py-2 text-left text-xs transition ${
                    item.id === selectedItemId
                      ? 'border-[var(--accent)] bg-[var(--accent)]/10'
                      : 'border-[var(--border-muted)] bg-[var(--bg-primary)] hover:border-[var(--accent)]/40'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className={`shrink-0 rounded-full border px-1.5 py-0.5 text-[10px] font-bold ${severityClasses(item.severity, item.priority)}`}>
                      {item.severity.toUpperCase().slice(0, 3)}
                    </span>
                    <span className="truncate font-semibold text-[var(--text-primary)]">{formatCategory(item.category)}</span>
                    {draft?.decision ? (
                      <span className={`ml-auto shrink-0 rounded border px-1.5 py-0.5 text-[10px] font-bold ${decisionActiveClasses(draft.decision)}`}>
                        {decisionShortLabel(draft.decision)}
                      </span>
                    ) : null}
                  </div>
                  <div className="mt-0.5 truncate text-[var(--text-secondary)]">{item.message}</div>
                </button>
              )
            })}
          </div>
        </div>

        {/* ── auto-accept hint ──────────────────────────────────────────── */}
        {counts.undecided > 0 ? (
          <div className="mt-4 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-xs text-[var(--text-secondary)]">
            {counts.undecided} item{counts.undecided !== 1 ? 's' : ''} without a decision will be auto-accepted by Stage 9.
          </div>
        ) : null}

        {layout === 'card' ? (
          <div className="mt-5 flex flex-col gap-2">{actionButtons}</div>
        ) : null}
      </aside>
    </>
  )

  // ── layout: workspace ────────────────────────────────────────────────────

  if (layout === 'workspace') {
    return (
      <div className="flex h-full min-h-0 flex-col overflow-hidden bg-[var(--bg-primary)] text-[var(--text-primary)]">
        <div className="shrink-0 flex items-center justify-between border-b border-[var(--border-muted)] bg-[var(--bg-secondary)] px-6 py-4">
          <div>
            <div className="text-lg font-semibold">Pipeline HITL Review</div>
            <div className="text-xs text-[var(--text-secondary)]">Gate 4: Stage 8 graph QA review.</div>
          </div>
          {actionButtons}
        </div>
        <div className="shrink-0 border-b border-[var(--border-muted)] bg-[var(--bg-secondary)] px-6 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-[var(--accent)] bg-[var(--accent)]/10 px-3 py-1 text-xs font-semibold text-[var(--accent)]">
              Stage 8 QA
            </span>
            <span className="text-xs text-[var(--text-secondary)]">{counts.total} item{counts.total !== 1 ? 's' : ''}</span>
            <span className="rounded-full bg-emerald-500/10 px-2 py-1 text-xs font-semibold text-emerald-700">{counts.decided} decided</span>
            {counts.undecided > 0 ? (
              <span className="rounded-full bg-[var(--bg-primary)] border border-[var(--border-muted)] px-2 py-1 text-xs font-semibold text-[var(--text-secondary)]">
                {counts.undecided} undecided
              </span>
            ) : null}
            <span className="ml-auto flex gap-2">
              {counts.high > 0 ? <span className="rounded-full bg-red-500/10 px-2 py-1 text-xs font-semibold text-red-700">{counts.high} high</span> : null}
              {counts.medium > 0 ? <span className="rounded-full bg-amber-500/10 px-2 py-1 text-xs font-semibold text-amber-700">{counts.medium} medium</span> : null}
              {counts.info > 0 ? <span className="rounded-full bg-yellow-500/10 px-2 py-1 text-xs font-semibold text-yellow-700">{counts.info} info</span> : null}
            </span>
          </div>
        </div>
        <div className="relative flex min-h-0 flex-1">
          {reviewBody}
        </div>
      </div>
    )
  }

  // ── layout: card ─────────────────────────────────────────────────────────

  return (
    <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="text-sm font-semibold">Stage 8 Graph QA Review</div>
          <div className="mt-1 text-xs text-[var(--text-secondary)]">
            Click a flagged item to inspect evidence and set a decision. Proceed to Stage 9 when ready.
          </div>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="rounded-full bg-emerald-500/10 px-2 py-1 text-emerald-700">{counts.decided} decided</span>
          {counts.high > 0 ? <span className="rounded-full bg-red-500/10 px-2 py-1 text-red-700">{counts.high} high</span> : null}
          {counts.medium > 0 ? <span className="rounded-full bg-amber-500/10 px-2 py-1 text-amber-700">{counts.medium} medium</span> : null}
          <span className="rounded-full bg-yellow-500/10 px-2 py-1 text-yellow-700">{counts.info} info</span>
        </div>
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        {reviewBody}
      </div>
    </div>
  )
}
