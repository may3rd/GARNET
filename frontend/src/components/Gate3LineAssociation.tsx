import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Spinner } from '@heroui/react'
import { Check, Minus, Pencil, Plus, Play, X } from 'lucide-react'
import { Card, SectionHeader, Tag } from '@/components/ui/primitives'
import { getPipelineArtifactJson, putPipelineArtifact } from '@/lib/api'
import { clampPan, fitScale as computeFit, wheelIntent, zoomAbout, type Bbox } from '@/lib/viewport'
import { useRunStore, type Sheet } from '@/stores/runStore'

/** The real shape of stage6_line_number_review.json / stage6_trace_associations.json — verified against source. */
type TraceSegment = { x1: number; y1: number; x2: number; y2: number }
type TraceEdge = { trace_id: string; segments: TraceSegment[] }
type TraceAssociationsArtifact = { trace_edges: TraceEdge[] }

type LineAssoc = {
  id: string
  bbox: Bbox
  text?: string
  normalized_text?: string
  confidence?: number | null
  trace_id: string
  // Absent on some accepted items (e.g. simulated/seeded review data) — no
  // trace-projection point was computed for them, so there's nothing to draw
  // a connector line to.
  projected_xy?: [number, number]
  distance_px?: number
  review_state?: string
  reason?: string
}
type LineReviewArtifact = {
  image_id?: string
  review_assumption?: string
  accepted: LineAssoc[]
  needs_review: LineAssoc[]
  traces_without_line_number: string[]
}

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

const pathFor = (segments: TraceSegment[] | undefined): string => {
  if (!segments?.length) return ''
  const pts = [`${segments[0].x1},${segments[0].y1}`, ...segments.map((s) => `${s.x2},${s.y2}`)]
  return `M${pts.join('L')}`
}

const bboxCenter = (b: Bbox) => ({ x: (b.x_min + b.x_max) / 2, y: (b.y_min + b.y_max) / 2 })

/** Closest point on any segment of a trace's polyline to (px, py), with its distance — used to re-project a line-number onto a newly-picked trace. */
const nearestPointOnSegments = (
  px: number,
  py: number,
  segments: TraceSegment[] | undefined
): { x: number; y: number; dist: number } | null => {
  if (!segments?.length) return null
  let best: { x: number; y: number; dist: number } | null = null
  for (const s of segments) {
    const dx = s.x2 - s.x1
    const dy = s.y2 - s.y1
    const lenSq = dx * dx + dy * dy
    const t = lenSq > 0 ? clamp(((px - s.x1) * dx + (py - s.y1) * dy) / lenSq, 0, 1) : 0
    const x = s.x1 + t * dx
    const y = s.y1 + t * dy
    const dist = Math.hypot(px - x, py - y)
    if (!best || dist < best.dist) best = { x, y, dist }
  }
  return best
}

/**
 * The same line-number box is often proposed against several candidate
 * traces/branches — `item.id` alone (the line-number's own id) collides
 * across those rows, so it can't be used as a React key or a decision key:
 * accepting one candidate would silently accept every row sharing that id.
 */
const rowKey = (item: LineAssoc) => `${item.id}::${item.trace_id}`

/**
 * Gate 3 — line-number-to-trace association review. Each item is a
 * line-number text box the system proposed attaching to a traced pipe path;
 * the reviewer accepts or rejects the pairing. Persisted as a plain PUT to
 * stage6_line_number_review.json (registered in ARTIFACT_INVALIDATION_START_STAGE,
 * unlike Gate 2's trace_overrides — there is no review-workspace apply step
 * for this artifact).
 */
export function Gate3LineAssociation({ sheet, onBack }: { sheet: Sheet; onBack: () => void }) {
  const resumeGate = useRunStore((s) => s.resumeGate)
  const setScreen = useRunStore((s) => s.setScreen)

  const [edges, setEdges] = useState<TraceEdge[] | null>(null)
  const [review, setReview] = useState<LineReviewArtifact | null>(null)
  const [items, setItems] = useState<LineAssoc[] | null>(null)
  const [accepted, setAccepted] = useState<Set<string>>(new Set())
  const [loadError, setLoadError] = useState<string | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [dirty, setDirty] = useState(false)

  const [selectedId, setSelectedId] = useState<string | null>(null)
  /** rowKey of the item whose trace association is being re-picked — while set, traces become clickable on the canvas. */
  const [reassigningKey, setReassigningKey] = useState<string | null>(null)
  const [hoverTraceId, setHoverTraceId] = useState<string | null>(null)

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
    Promise.all([
      getPipelineArtifactJson<TraceAssociationsArtifact>(jobId, 'stage6_trace_associations.json'),
      getPipelineArtifactJson<LineReviewArtifact>(jobId, 'stage6_line_number_review.json'),
    ])
      .then(([ta, lnr]) => {
        if (cancelled) return
        setEdges(ta.trace_edges ?? [])
        setReview(lnr)
        setItems([...(lnr.accepted ?? []), ...(lnr.needs_review ?? [])])
        setAccepted(new Set((lnr.accepted ?? []).map(rowKey)))
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : 'Could not load line associations')
      })
    return () => {
      cancelled = true
    }
  }, [jobId])

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const measure = () => setViewport({ w: el.clientWidth, h: el.clientHeight })
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    measure()
    return () => ro.disconnect()
    // items stays null behind a Spinner until load finishes — see Gate1/Gate2 for why deps can't be [].
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
    let dragged = false
    const move = (ev: MouseEvent) => {
      if (Math.abs(ev.clientX - origin.mx) > 3 || Math.abs(ev.clientY - origin.my) > 3) dragged = true
      setPan(settle({ x: origin.px + (ev.clientX - origin.mx), y: origin.py + (ev.clientY - origin.my) }, scale))
    }
    const up = () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
      if (!dragged) setSelectedId(null)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  const setDecision = (item: LineAssoc, isAccepted: boolean) => {
    setAccepted((prev) => {
      const next = new Set(prev)
      if (isAccepted) next.add(rowKey(item))
      else next.delete(rowKey(item))
      return next
    })
    setDirty(true)
  }

  const startReassign = (item: LineAssoc) => {
    setSelectedId(rowKey(item))
    setReassigningKey(rowKey(item))
  }

  const cancelReassign = () => setReassigningKey(null)

  useEffect(() => {
    if (!reassigningKey) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') cancelReassign()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [reassigningKey])

  /** User picked a new trace on the canvas while reassigning — re-point the selected item at it and re-project its connector. */
  const pickTraceForReassign = (edge: TraceEdge) => {
    const item = items?.find((it) => rowKey(it) === reassigningKey)
    if (!item) return
    const center = bboxCenter(item.bbox)
    const nearest = nearestPointOnSegments(center.x, center.y, edge.segments)
    const oldKey = rowKey(item)
    const updated: LineAssoc = {
      ...item,
      trace_id: edge.trace_id,
      projected_xy: nearest ? [nearest.x, nearest.y] : undefined,
      distance_px: nearest?.dist,
    }
    const newKey = rowKey(updated)
    setItems((prev) => prev!.map((it) => (rowKey(it) === oldKey ? updated : it)))
    setAccepted((prev) => {
      if (!prev.has(oldKey)) return prev
      const next = new Set(prev)
      next.delete(oldKey)
      next.add(newKey)
      return next
    })
    setSelectedId(newKey)
    setReassigningKey(null)
    setDirty(true)
  }

  const counts = useMemo(() => {
    const total = items?.length ?? 0
    return { total, accepted: accepted.size, needsReview: total - accepted.size }
  }, [items, accepted])

  const save = async (): Promise<boolean> => {
    if (!jobId || !review || !items) return false
    setSaving(true)
    setSaveError(null)
    try {
      const acceptedItems = items
        .filter((it) => accepted.has(rowKey(it)))
        .map((it) => ({ ...it, review_state: 'accepted', review_source: 'human', review_required: false }))
      const needsReviewItems = items
        .filter((it) => !accepted.has(rowKey(it)))
        .map((it) => ({ ...it, review_state: 'needs_review', review_source: 'human', review_required: false }))
      const payload: LineReviewArtifact = {
        ...review,
        accepted: acceptedItems,
        needs_review: needsReviewItems,
      }
      await putPipelineArtifact(jobId, 'stage6_line_number_review.json', payload)
      setReview(payload)
      setDirty(false)
      return true
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Could not save line association review')
      return false
    } finally {
      setSaving(false)
    }
  }

  const saveAndConfirm = async () => {
    setConfirming(true)
    try {
      if (dirty && !(await save())) return
      await resumeGate(sheet.id, 3)
      setScreen('run')
    } finally {
      setConfirming(false)
    }
  }

  if (!jobId) return null

  if (loadError) {
    return (
      <div className="p-6">
        <Card padding={20}>
          <SectionHeader title="Could not load line associations" description={loadError} />
        </Card>
      </div>
    )
  }

  if (!items || !edges) {
    return (
      <div className="flex h-full items-center justify-center gap-2">
        <Spinner size="sm" />
        <span style={{ fontSize: 13, color: 'var(--muted)' }}>Loading line associations…</span>
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex min-h-0 flex-1 gap-4" style={{ padding: '16px 24px' }}>
        <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
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
                  {/* Trace edges — context only, unless reassigning a line number: then every trace becomes a clickable pick target (wide invisible stroke for hit-testing, thin one for looks). */}
                  {edges.map((edge) => {
                    const d = pathFor(edge.segments)
                    if (!d) return null
                    const isHover = reassigningKey !== null && edge.trace_id === hoverTraceId
                    return (
                      <g key={edge.trace_id}>
                        {reassigningKey && (
                          <path
                            d={d}
                            fill="none"
                            stroke="transparent"
                            strokeWidth={16 / scale}
                            style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                            onMouseEnter={() => setHoverTraceId(edge.trace_id)}
                            onMouseLeave={() => setHoverTraceId((id) => (id === edge.trace_id ? null : id))}
                            onClick={(e) => {
                              e.stopPropagation()
                              pickTraceForReassign(edge)
                            }}
                          />
                        )}
                        <path
                          d={d}
                          fill="none"
                          stroke={isHover ? 'var(--accent)' : 'var(--muted)'}
                          strokeWidth={(isHover ? 4 : reassigningKey ? 2.4 : 1.4) / scale}
                          opacity={reassigningKey ? 1 : 0.5}
                          pointerEvents="none"
                        />
                      </g>
                    )
                  })}
                  {items.map((item) => {
                    const key = rowKey(item)
                    const isSel = key === selectedId
                    const isReassignTarget = key === reassigningKey
                    const isAccepted = accepted.has(key)
                    const color = isReassignTarget ? 'var(--accent)' : isAccepted ? 'var(--success)' : 'var(--warning)'
                    const muted = reassigningKey !== null && !isReassignTarget
                    const center = bboxCenter(item.bbox)
                    const projection = item.projected_xy
                    return (
                      <g
                        key={key}
                        opacity={muted ? 0.35 : 1}
                        style={{ pointerEvents: reassigningKey ? 'none' : 'auto', cursor: 'pointer' }}
                        onClick={(e) => {
                          e.stopPropagation()
                          setSelectedId(key)
                        }}
                      >
                        {projection && (
                          <line
                            x1={center.x}
                            y1={center.y}
                            x2={projection[0]}
                            y2={projection[1]}
                            stroke={color}
                            strokeWidth={(isSel ? 2 : 1) / scale}
                            strokeDasharray={`${5 / scale} ${4 / scale}`}
                          />
                        )}
                        <rect
                          x={item.bbox.x_min}
                          y={item.bbox.y_min}
                          width={item.bbox.x_max - item.bbox.x_min}
                          height={item.bbox.y_max - item.bbox.y_min}
                          fill={color}
                          fillOpacity={0.13}
                          stroke={color}
                          strokeWidth={(isSel || isReassignTarget ? 3 : 1.6) / scale}
                          strokeDasharray={isReassignTarget ? `${6 / scale} ${4 / scale}` : undefined}
                        />
                        {projection && <circle cx={projection[0]} cy={projection[1]} r={4 / scale} fill={color} />}
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

            {/* Reassign-mode banner — appears while picking a new trace for the selected line number. */}
            {reassigningKey && (
              <div
                className="absolute flex items-center gap-2.5"
                style={{
                  top: 12,
                  right: 12,
                  padding: '8px 10px',
                  background: 'var(--overlay)',
                  borderRadius: 'var(--r-btn)',
                  boxShadow: 'inset 0 0 0 1px var(--accent), 0 8px 24px rgba(0,0,0,.14)',
                }}
                onMouseDown={(e) => e.stopPropagation()}
              >
                <span style={{ fontSize: 12.5, color: 'var(--foreground)' }}>
                  Click the traced line to associate with{' '}
                  <strong>
                    {(() => {
                      const target = items.find((it) => rowKey(it) === reassigningKey)
                      return target ? target.text || target.normalized_text || target.id : ''
                    })()}
                  </strong>
                </span>
                <Button variant="ghost" style={{ height: 26, padding: '0 10px', borderRadius: 'var(--r-btn)' }} onPress={cancelReassign}>
                  Cancel
                </Button>
              </div>
            )}
          </div>

        </div>

        {/* Line association list — the actual review surface. Canvas markers exist
            only for items with a proposed attachment point; traces with no proposal
            at all (the common case — see traces_without_line_number) have nothing to
            draw, so they must show up here or the reviewer never sees them. */}
        <div className="flex min-h-0 flex-col gap-2.5 shrink-0" style={{ width: 280 }}>
          <Card className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-hidden" padding={16}>
            <SectionHeader
              title="Line association"
              description={`${counts.total} proposed · ${review?.traces_without_line_number.length ?? 0} unlabeled`}
              actions={<Tag tone="neutral">{counts.total + (review?.traces_without_line_number.length ?? 0)}</Tag>}
            />
            <div className="min-h-0 flex-1 overflow-y-auto">
              {items.map((item) => {
                const isAccepted = accepted.has(rowKey(item))
                const isSel = rowKey(item) === selectedId
                const isReassignTarget = rowKey(item) === reassigningKey
                return (
                  <div
                    key={rowKey(item)}
                    onClick={() => {
                      if (reassigningKey && reassigningKey !== rowKey(item)) setReassigningKey(null)
                      setSelectedId(rowKey(item))
                    }}
                    style={{
                      padding: '8px 0',
                      borderBottom: '1px solid color-mix(in oklab, var(--separator) 50%, transparent)',
                      background: isSel ? 'var(--accent-soft)' : undefined,
                      cursor: 'pointer',
                    }}
                  >
                    <div className="flex items-center gap-2">
                      {isAccepted ? (
                        <Check size={13} strokeWidth={2.2} style={{ color: 'var(--success)', flexShrink: 0 }} />
                      ) : (
                        <X size={13} strokeWidth={2.2} style={{ color: 'var(--warning)', flexShrink: 0 }} />
                      )}
                      <span className="mono truncate" style={{ flex: 1, fontSize: 12.5 }}>
                        {item.text || item.normalized_text || item.trace_id}
                      </span>
                      {isReassignTarget ? (
                        <Tag tone="accent">Picking…</Tag>
                      ) : (
                        <span className="mono" style={{ fontSize: 11, color: 'var(--muted)', flexShrink: 0 }}>
                          {item.distance_px != null ? `${Math.round(item.distance_px)}px` : ''}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 1, paddingLeft: 21 }}>
                      trace {item.trace_id}
                    </div>
                    <div className="mt-1.5 flex items-center gap-1.5" style={{ paddingLeft: 21 }}>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          setDecision(item, true)
                        }}
                        style={{
                          height: 22,
                          padding: '0 8px',
                          border: 0,
                          borderRadius: 'var(--r-btn)',
                          background: isAccepted ? 'var(--accent)' : 'var(--surface-secondary)',
                          color: isAccepted ? 'var(--white)' : 'var(--muted)',
                          cursor: 'pointer',
                          fontSize: 11,
                        }}
                      >
                        Accept
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          setDecision(item, false)
                        }}
                        style={{
                          height: 22,
                          padding: '0 8px',
                          border: 0,
                          borderRadius: 'var(--r-btn)',
                          background: !isAccepted ? 'var(--accent)' : 'var(--surface-secondary)',
                          color: !isAccepted ? 'var(--white)' : 'var(--muted)',
                          cursor: 'pointer',
                          fontSize: 11,
                        }}
                      >
                        Needs review
                      </button>
                      <span style={{ flex: 1 }} />
                      <button
                        type="button"
                        title="Pick a different traced line for this line number"
                        onClick={(e) => {
                          e.stopPropagation()
                          startReassign(item)
                        }}
                        className="flex items-center gap-1"
                        style={{
                          height: 22,
                          padding: '0 8px',
                          border: 0,
                          borderRadius: 'var(--r-btn)',
                          background: 'var(--surface-secondary)',
                          color: 'var(--foreground)',
                          cursor: 'pointer',
                          fontSize: 11,
                        }}
                      >
                        <Pencil size={11} strokeWidth={2} />
                        Change trace
                      </button>
                    </div>
                  </div>
                )
              })}
              {(review?.traces_without_line_number ?? []).map((traceId) => (
                <div
                  key={traceId}
                  className="flex items-center gap-2"
                  style={{
                    padding: '7px 0',
                    borderBottom: '1px solid color-mix(in oklab, var(--separator) 50%, transparent)',
                    opacity: 0.7,
                  }}
                >
                  <Minus size={13} strokeWidth={2} style={{ color: 'var(--muted)', flexShrink: 0 }} />
                  <span className="mono truncate" style={{ flex: 1, fontSize: 12.5, color: 'var(--muted)' }}>
                    {traceId}
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--muted)', flexShrink: 0 }}>no line number</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      {/* Footer */}
      <div className="flex shrink-0 items-center gap-2.5" style={{ height: 56, padding: '0 24px', borderTop: '1px solid var(--separator)' }}>
        <span className="mono" style={{ fontSize: 13, color: 'var(--muted)' }}>
          {counts.total} line numbers · {counts.accepted} accepted · {counts.needsReview} needs review
          {review && review.traces_without_line_number.length > 0 ? ` · ${review.traces_without_line_number.length} traces unlabeled` : ''}
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
