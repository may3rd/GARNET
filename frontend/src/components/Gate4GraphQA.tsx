import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Spinner } from '@heroui/react'
import { Minus, Plus, Play } from 'lucide-react'
import { Card, ResizableSidebar, SectionHeader, Tag, type TagTone } from '@/components/ui/primitives'
import { useResizableSidebar } from '@/hooks/useResizableSidebar'
import { APIError, getPipelineArtifactJson, putPipelineArtifact } from '@/lib/api'
import { clampPan, fitScale as computeFit, wheelIntent, zoomAbout } from '@/lib/viewport'
import { useRunStore, type Sheet } from '@/stores/runStore'

/** The real shape of stage8_review_items.json — verified against a live job. */
type Severity = 'high' | 'review' | 'medium' | 'info' | 'low' | string
type ReviewItem = {
  id: string
  source_stage: string | string[]
  category: string
  severity: Severity
  priority: number
  message: string
  evidence?: Record<string, unknown>
  geometry?: { x: number; y: number }
  /** 'flow_direction' items get direction buttons instead of the generic three. */
  review_item_type?: string
  /** The edge `set_flow_direction` applies to. Only flow_direction items carry it. */
  edge_id?: string
}
type ReviewItemsArtifact = { image_id?: string; review_items: ReviewItem[] }

type Decision = 'accept_as_is' | 'false_positive' | 'defer'
const DECISIONS: { key: Decision; label: string }[] = [
  { key: 'accept_as_is', label: 'Accept as-is' },
  { key: 'false_positive', label: 'False positive' },
  { key: 'defer', label: 'Defer' },
]

/**
 * Flow-direction review. `forward`/`reverse` are relative to the stored
 * polyline order, which is why the selected edge is drawn on the canvas with
 * a start marker — direction is meaningless without seeing which end is which.
 */
type FlowState = 'forward' | 'reverse' | 'bidirectional' | 'unknown'
const FLOW_DECISIONS: { key: FlowState; label: string }[] = [
  { key: 'forward', label: 'Forward' },
  { key: 'reverse', label: 'Reverse' },
  { key: 'bidirectional', label: 'Both' },
  { key: 'unknown', label: 'Unknown' },
]
const FLOW_STATES = new Set<string>(FLOW_DECISIONS.map((d) => d.key))
const isFlowItem = (item: ReviewItem) => item.review_item_type === 'flow_direction' && !!item.edge_id

/** stage8_review_items.json carries the edge polyline under both keys. */
const edgePolyline = (item: ReviewItem): { x: number; y: number }[] => {
  const geom = (item as Record<string, unknown>).edge_geometry ?? (item.evidence ?? {}).edge_geometry
  const pts = (geom as { polyline?: unknown } | undefined)?.polyline
  return Array.isArray(pts)
    ? pts.filter((pt): pt is { x: number; y: number } => typeof pt?.x === 'number' && typeof pt?.y === 'number')
    : []
}

const SEVERITY_TONE: Record<string, TagTone> = {
  high: 'danger',
  medium: 'warning',
  review: 'accent',
  info: 'neutral',
  low: 'neutral',
}

const SEVERITY_DOT: Record<string, string> = {
  high: 'var(--danger)',
  medium: 'var(--warning)',
  review: 'var(--accent)',
  info: 'var(--muted)',
  low: 'var(--muted)',
}

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

/**
 * Gate 4 — graph QA review. Unlike Gates 1-3, most review items (component-
 * level line-number gaps, review-density flags) have no single (x, y) —
 * this is fundamentally a punch list, with the canvas as secondary spatial
 * context for the items that do have geometry. Persisted as a plain PUT to
 * stage8_review_decisions.json (registered in ARTIFACT_INVALIDATION_START_STAGE,
 * same simple pattern as Gate 1 and Gate 3 — no review-workspace step here).
 *
 * Decisions map onto stage9_review_decisions.py: accept_as_is /
 * false_positive / defer for ordinary items, and set_flow_direction for
 * flow_direction items (forward / reverse / bidirectional / unknown).
 * set_line_number is still missing — it edits the graph directly, picking a
 * line number id plus specific edge ids, and is a materially bigger feature.
 *
 * Silence is NOT a safe default. Stage 9 counts an undecided release-blocking
 * item as "unresolved", and stage9_release_gate stays blocked — which redacts
 * the Stage 10 final export. Only accept_as_is, false_positive,
 * set_line_number and set_flow_direction clear an item; defer does not.
 */
export function Gate4GraphQA({ sheet, onBack }: { sheet: Sheet; onBack: () => void }) {
  const resumeGate = useRunStore((s) => s.resumeGate)
  const setScreen = useRunStore((s) => s.setScreen)
  const sidebar = useResizableSidebar(420, { min: 280, max: 640 })

  const [items, setItems] = useState<ReviewItem[] | null>(null)
  const [decisions, setDecisions] = useState<Map<string, Decision | FlowState>>(new Map())
  const [loadError, setLoadError] = useState<string | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [dirty, setDirty] = useState(false)

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [severityFilter, setSeverityFilter] = useState<'all' | Severity>('all')

  const [zoom, setZoom] = useState<number | null>(null)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const viewportRef = useRef<HTMLDivElement>(null)
  const [viewport, setViewport] = useState({ w: 0, h: 0 })

  const jobId = sheet.jobId
  const imgW = sheet.size?.width ?? 0
  const imgH = sheet.size?.height ?? 0

  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    getPipelineArtifactJson<ReviewItemsArtifact>(jobId, 'stage8_review_items.json')
      .then(async (ri) => {
        if (cancelled) return
        setItems(ri.review_items ?? [])
        try {
          const existing = await getPipelineArtifactJson<{ decisions: Array<Record<string, unknown>> }>(
            jobId,
            'stage8_review_decisions.json'
          )
          const seeded = new Map<string, Decision | FlowState>()
          for (const d of existing.decisions ?? []) {
            const id = String(d.review_item_id ?? '')
            const decision = String(d.decision ?? '')
            if (!id) continue
            if (decision === 'accept_as_is' || decision === 'false_positive' || decision === 'defer') {
              seeded.set(id, decision)
            } else if (decision === 'set_flow_direction') {
              const state = String(d.flow_direction_state ?? '')
              if (FLOW_STATES.has(state)) seeded.set(id, state as FlowState)
            }
          }
          if (!cancelled) setDecisions(seeded)
        } catch (err) {
          if (!(err instanceof APIError && err.status === 404)) throw err
        }
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : 'Could not load review items')
      })
    return () => {
      cancelled = true
    }
  }, [jobId])

  const filtered = useMemo(
    () => (items ?? []).filter((it) => severityFilter === 'all' || it.severity === severityFilter),
    [items, severityFilter]
  )
  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const measure = () => setViewport({ w: el.clientWidth, h: el.clientHeight })
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    measure()
    return () => ro.disconnect()
    // items stays null behind a Spinner until load finishes — see Gate1/2/3 for why deps can't be [].
  }, [items !== null])

  const fitScale = computeFit(imgW, imgH, viewport.w, viewport.h)
  const scale = zoom ?? fitScale

  const settle = useCallback(
    (p: { x: number; y: number }, atScale: number) =>
      clampPan(p, atScale, imgW, imgH, viewport.w, viewport.h),
    [imgW, imgH, viewport.w, viewport.h]
  )
  useEffect(() => {
    setPan((p) => settle(p, scale))
  }, [settle, scale])

  const panRef = useRef(pan)
  panRef.current = pan
  const scaleRef = useRef(scale)
  scaleRef.current = scale

  const zoomAt = useCallback(
    (factor: number, cx: number, cy: number) => {
      const current = scaleRef.current
      const result = zoomAbout(panRef.current, current, factor, cx, cy, imgW, imgH, viewport.w, viewport.h, 0.02, 8)
      if (result.scale === current) return
      scaleRef.current = result.scale
      panRef.current = result.pan
      setZoom(result.scale)
      setPan(result.pan)
    },
    [imgW, imgH, viewport.w, viewport.h]
  )

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    let zoomDelta = 0
    let panX = 0
    let panY = 0
    let originX = 0
    let originY = 0
    let frame = 0
    const panBy = (dx: number, dy: number) => {
      const next = settle({ x: panRef.current.x + dx, y: panRef.current.y + dy }, scaleRef.current)
      panRef.current = next
      setPan(next)
    }
    const flush = () => {
      frame = 0
      if (zoomDelta !== 0) {
        const d = zoomDelta
        zoomDelta = 0
        zoomAt(Math.exp(-d * 0.0015), originX, originY)
      }
      if (panX !== 0 || panY !== 0) {
        const dx = panX
        const dy = panY
        panX = 0
        panY = 0
        panBy(-dx, -dy)
      }
    }
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const intent = wheelIntent(e, rect.width, rect.height)
      if (intent.kind === 'zoom') {
        originX = e.clientX - rect.left
        originY = e.clientY - rect.top
        zoomDelta = clamp(zoomDelta + intent.delta, -240, 240)
      } else {
        panX = clamp(panX + intent.dx, -240, 240)
        panY = clamp(panY + intent.dy, -240, 240)
      }
      if (!frame) frame = requestAnimationFrame(flush)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => {
      el.removeEventListener('wheel', onWheel)
      if (frame) cancelAnimationFrame(frame)
    }
  }, [zoomAt, settle])

  const startPan = (e: React.MouseEvent) => {
    const origin = { px: pan.x, py: pan.y, mx: e.clientX, my: e.clientY }
    const move = (ev: MouseEvent) => {
      setPan(settle({ x: origin.px + (ev.clientX - origin.mx), y: origin.py + (ev.clientY - origin.my) }, scale))
    }
    const up = () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  const selectItem = (item: ReviewItem) => {
    setSelectedId(item.id)
    if (item.geometry && viewport.w && viewport.h) {
      const next = settle({ x: viewport.w / 2 - item.geometry.x * scale, y: viewport.h / 2 - item.geometry.y * scale }, scale)
      setPan(next)
    }
  }

  const setDecision = (item: ReviewItem, decision: Decision | FlowState | null) => {
    setDecisions((prev) => {
      const next = new Map(prev)
      if (decision) next.set(item.id, decision)
      else next.delete(item.id)
      return next
    })
    setDirty(true)
  }

  const counts = useMemo(() => {
    const total = items?.length ?? 0
    return { total, decided: decisions.size, undecided: total - decisions.size }
  }, [items, decisions])

  const save = async (): Promise<boolean> => {
    if (!jobId) return false
    setSaving(true)
    setSaveError(null)
    try {
      const edgeById = new Map((items ?? []).map((it) => [it.id, it.edge_id]))
      const payload = {
        decisions: [...decisions.entries()].map(([review_item_id, decision]) =>
          FLOW_STATES.has(decision)
            ? {
                review_item_id,
                decision: 'set_flow_direction',
                flow_direction_state: decision,
                edge_id: edgeById.get(review_item_id),
                reviewer: 'hitl',
              }
            : { review_item_id, decision, reviewer: 'hitl' }
        ),
      }
      await putPipelineArtifact(jobId, 'stage8_review_decisions.json', payload)
      setDirty(false)
      return true
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Could not save review decisions')
      return false
    } finally {
      setSaving(false)
    }
  }

  const saveAndConfirm = async () => {
    setConfirming(true)
    try {
      if (dirty && !(await save())) return
      setScreen('run')
      await resumeGate(sheet.id, 4)
    } finally {
      setConfirming(false)
    }
  }

  if (!jobId) return null

  if (loadError) {
    return (
      <div className="p-6">
        <Card padding={20}>
          <SectionHeader title="Could not load graph QA review" description={loadError} />
        </Card>
      </div>
    )
  }

  if (!items) {
    return (
      <div className="flex h-full items-center justify-center gap-2">
        <Spinner size="sm" />
        <span style={{ fontSize: 13, color: 'var(--muted)' }}>Loading review items…</span>
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex min-h-0 flex-1 gap-4" style={{ padding: '16px 24px' }}>
        <div className="relative flex min-w-0 flex-[1.3] flex-col overflow-hidden">
          <div
            ref={viewportRef}
            className="relative min-h-0 flex-1 overflow-hidden"
            style={{
              background: 'var(--surface-tertiary)',
              borderRadius: 'var(--r-table)',
              cursor: 'grab',
              userSelect: 'none',
              WebkitUserSelect: 'none',
            }}
            onMouseDown={startPan}
          >
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`,
                transformOrigin: '0 0',
              }}
            >
              <img
                src={sheet.previewUrl}
                alt={sheet.label}
                draggable={false}
                style={{ display: 'block', background: '#ffffff', width: imgW, height: imgH, maxWidth: 'none' }}
              />
              {imgW > 0 && (
                <svg
                  width={imgW}
                  height={imgH}
                  viewBox={`0 0 ${imgW} ${imgH}`}
                  style={{ position: 'absolute', left: 0, top: 0, overflow: 'visible', pointerEvents: 'none' }}
                >
                  {(() => {
                    // Forward/reverse are relative to polyline order, so the
                    // selected edge is drawn with its start end marked.
                    const sel = filtered.find((it) => it.id === selectedId)
                    const pts = sel && isFlowItem(sel) ? edgePolyline(sel) : []
                    if (pts.length < 2) return null
                    return (
                      <g>
                        <polyline
                          points={pts.map((pt) => `${pt.x},${pt.y}`).join(' ')}
                          fill="none"
                          stroke="var(--accent)"
                          strokeWidth={4 / scale}
                        />
                        <circle cx={pts[0].x} cy={pts[0].y} r={7 / scale} fill="var(--accent)" />
                        <text
                          x={pts[0].x + 10 / scale}
                          y={pts[0].y - 10 / scale}
                          fill="var(--accent)"
                          style={{ fontSize: 14 / scale }}
                        >
                          start
                        </text>
                      </g>
                    )
                  })()}
                  {filtered.map((item) => {
                    if (!item.geometry) return null
                    const isSel = item.id === selectedId
                    const color = SEVERITY_DOT[item.severity] ?? 'var(--muted)'
                    const decided = decisions.has(item.id)
                    return (
                      <g
                        key={item.id}
                        style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                        opacity={decided ? 0.45 : 1}
                        onClick={(e) => {
                          e.stopPropagation()
                          selectItem(item)
                        }}
                      >
                        <circle cx={item.geometry.x} cy={item.geometry.y} r={(isSel ? 11 : 8) / scale} fill="none" stroke={color} strokeWidth={(isSel ? 3 : 2) / scale} />
                        <circle cx={item.geometry.x} cy={item.geometry.y} r={3 / scale} fill={color} />
                      </g>
                    )
                  })}
                </svg>
              )}
            </div>

            <div
              className="absolute flex items-center gap-1"
              style={{
                top: 12,
                left: 12,
                padding: 4,
                background: 'var(--overlay)',
                borderRadius: 'var(--r-btn)',
                boxShadow: 'inset 0 0 0 1px var(--border), 0 8px 24px rgba(0,0,0,.14)',
              }}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                aria-label="Zoom in"
                title="Zoom in"
                onClick={() => zoomAt(1.25, viewport.w / 2, viewport.h / 2)}
                className="flex items-center justify-center"
                style={{ width: 26, height: 26, border: 0, background: 'transparent', borderRadius: 'var(--r-btn)', color: 'var(--foreground)', cursor: 'pointer' }}
              >
                <Plus size={15} strokeWidth={1.8} />
              </button>
              <button
                type="button"
                aria-label="Zoom out"
                title="Zoom out"
                onClick={() => zoomAt(1 / 1.25, viewport.w / 2, viewport.h / 2)}
                className="flex items-center justify-center"
                style={{ width: 26, height: 26, border: 0, background: 'transparent', borderRadius: 'var(--r-btn)', color: 'var(--foreground)', cursor: 'pointer' }}
              >
                <Minus size={15} strokeWidth={1.8} />
              </button>
              <span style={{ width: 1, height: 20, background: 'var(--separator)', margin: '0 2px' }} />
              <span className="mono" style={{ padding: '0 8px', fontSize: 12, color: 'var(--muted)' }}>
                {Math.round(scale * 100)}%
              </span>
              <button
                type="button"
                onClick={() => {
                  setZoom(null)
                  setPan(settle({ x: 0, y: 0 }, fitScale))
                }}
                className="flex items-center justify-center"
                style={{ height: 26, padding: '0 10px', border: 0, background: 'transparent', borderRadius: 'var(--r-btn)', color: 'var(--foreground)', cursor: 'pointer', fontSize: 12 }}
              >
                Fit
              </button>
            </div>
          </div>
        </div>

        {/* Punch list — most review items have no single point, so this is the primary surface. */}
        <ResizableSidebar
          width={sidebar.width}
          collapsed={sidebar.collapsed}
          onToggleCollapsed={sidebar.toggleCollapsed}
          onStartResize={sidebar.startResize}
        >
          <div className="flex min-h-0 flex-1 flex-col gap-2.5">
          <div className="flex shrink-0 items-center gap-1.5">
            {(['all', 'high', 'medium', 'review', 'info'] as const).map((sev) => (
              <button
                key={sev}
                type="button"
                onClick={() => setSeverityFilter(sev)}
                className="flex items-center justify-center"
                style={{
                  height: 26,
                  padding: '0 10px',
                  border: 0,
                  background: severityFilter === sev ? 'var(--accent-soft)' : 'var(--surface-secondary)',
                  color: severityFilter === sev ? 'var(--accent-soft-fg)' : 'var(--muted)',
                  borderRadius: 'var(--r-btn)',
                  cursor: 'pointer',
                  fontSize: 12,
                  textTransform: 'capitalize',
                }}
              >
                {sev}
              </button>
            ))}
          </div>
          <Card className="flex min-h-0 flex-1 flex-col overflow-hidden" padding={0}>
            <div className="min-h-0 flex-1 overflow-y-auto">
              {filtered.length === 0 ? (
                <div style={{ padding: 20 }}>
                  <SectionHeader title="Nothing here" description="No review items match this filter." />
                </div>
              ) : (
                filtered.map((item) => {
                  const decision = decisions.get(item.id)
                  const isSel = item.id === selectedId
                  return (
                    <div
                      key={item.id}
                      onClick={() => selectItem(item)}
                      style={{
                        padding: '10px 14px',
                        borderBottom: '1px solid var(--separator)',
                        background: isSel ? 'var(--accent-soft)' : undefined,
                        cursor: 'pointer',
                      }}
                    >
                      <div className="flex items-center gap-2">
                        <Tag tone={SEVERITY_TONE[item.severity] ?? 'neutral'}>{item.severity}</Tag>
                        <span className="mono" style={{ fontSize: 11.5, color: 'var(--muted)' }}>{item.category}</span>
                        <div className="flex-1" />
                        {!item.geometry && <span style={{ fontSize: 10.5, color: 'var(--muted)' }}>no location</span>}
                      </div>
                      <div style={{ fontSize: 13, marginTop: 4 }}>{item.message}</div>
                      <div className="mt-2 flex items-center gap-1.5">
                        {(isFlowItem(item) ? FLOW_DECISIONS : DECISIONS).map((d) => (
                          <button
                            key={d.key}
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              setDecision(item, decision === d.key ? null : d.key)
                            }}
                            style={{
                              height: 24,
                              padding: '0 8px',
                              border: 0,
                              borderRadius: 'var(--r-btn)',
                              background: decision === d.key ? 'var(--accent)' : 'var(--surface-secondary)',
                              color: decision === d.key ? 'var(--white)' : 'var(--muted)',
                              cursor: 'pointer',
                              fontSize: 11.5,
                            }}
                          >
                            {d.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  )
                })
              )}
            </div>
          </Card>
          </div>
        </ResizableSidebar>
      </div>

      {/* Footer */}
      <div className="flex shrink-0 items-center gap-2.5" style={{ height: 56, padding: '0 24px', borderTop: '1px solid var(--separator)' }}>
        <span className="mono" style={{ fontSize: 13, color: 'var(--muted)' }}>
          {counts.total} review items · {counts.decided} decided · {counts.undecided} undecided
          {dirty ? ' · unsaved' : ''}
        </span>
        {saveError && <span style={{ fontSize: 12, color: 'var(--danger)' }}>{saveError}</span>}
        <div className="flex-1" />
        <Button variant="ghost" style={{ height: 32, borderRadius: 'var(--r-btn)' }} onPress={onBack}>
          Back to queue
        </Button>
        <Button variant="secondary" isDisabled={!dirty || saving} style={{ height: 32, borderRadius: 'var(--r-btn)' }} onPress={() => void save()}>
          {saving ? <Spinner size="sm" /> : 'Save'}
        </Button>
        <Button variant="primary" isDisabled={confirming} style={{ height: 32, borderRadius: 'var(--r-btn)' }} onPress={() => void saveAndConfirm()}>
          {confirming ? <Spinner size="sm" /> : <Play size={14} strokeWidth={1.6} />}
          Save &amp; continue
        </Button>
      </div>
    </div>
  )
}
