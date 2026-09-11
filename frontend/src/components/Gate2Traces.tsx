import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Spinner } from '@heroui/react'
import { Minus, Pencil, Plus, Play, RefreshCw, RotateCcw, Trash2, X } from 'lucide-react'
import { Card, ResizableSidebar, SectionHeader, Tag } from '@/components/ui/primitives'
import { useResizableSidebar } from '@/hooks/useResizableSidebar'
import { classColor, normalizeClass } from '@/lib/detectionClasses'
import {
  APIError,
  getPipelineArtifactJson,
  getPipelineReviewWorkspace,
  commitPipelineReviewWorkspace,
  putPipelineArtifact,
} from '@/lib/api'
import { clampPan, fitScale as computeFit, wheelIntent, zoomAbout } from '@/lib/viewport'
import { useRunStore, type Sheet } from '@/stores/runStore'
import type { PipelineReviewWorkspaceState } from '@/types'

/** The real shape of stage5b_trace_results.json / .../branch_trace_results.json — verified against a live job. */
type TraceSegment = { x1: number; y1: number; x2: number; y2: number }
type TraceRecord = {
  terminal_type?: string
  terminal_obj_id?: string | null
  // Absent (not just empty) on branches whose trace was skipped.
  segments?: TraceSegment[]
  trace_length_px?: number
  status?: string
}
type TraceResultsArtifact = Record<string, TraceRecord>
type BranchResultsArtifact = { branches: Record<string, TraceRecord> }
/** stage4_objects.json row — every detected + manually-added object/equipment box, shown here as background context (frontend_old's PipelineReviewWorkspaceView shows all of these alongside the trace/port overlay). */
type Bbox = { x_min: number; y_min: number; x_max: number; y_max: number }
type Stage4Object = { id: string; class_name?: string; bbox: Bbox }
/** stage3_equipment_bboxes.json — the id equipment ports are actually keyed under (see Direction below). */
type EquipmentArtifact = { equipment?: { id: string; bbox: Bbox }[] }
/** stage5_connection_ports.json — one port per (x, y, direction) triple, keyed by object id. */
type Direction = 'UP' | 'DOWN' | 'LEFT' | 'RIGHT'
type Port = [number, number, Direction]
type PortsArtifact = Record<string, Port[]>

type Kind = 'trace' | 'branch'
type Entry = { key: string; kind: Kind; id: string; record: TraceRecord }

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

const pathFor = (segments: TraceSegment[] | undefined): string => {
  if (!segments?.length) return ''
  const pts = [`${segments[0].x1},${segments[0].y1}`, ...segments.map((s) => `${s.x2},${s.y2}`)]
  return `M${pts.join('L')}`
}

/**
 * Objects carry ports under two different keys depending on class — mirrors
 * backend/garnet/path_tracer/stage5b_pipeline.py's _compute_connection_ports:
 * page-connection symbols are keyed by their own stage4_objects.json id, but
 * equipment is keyed by its id in stage3_equipment_bboxes.json (a tag number,
 * or an "equip_NNN" fallback) — a *different* id than the same box's
 * stage4_objects.json entry, joined here by matching bbox.
 */
const PAGE_CONNECTION_CLASSES = new Set(['page connection', 'connection', 'utility connection', 'page connection symbol'])
const bboxKey = (b: Bbox) => `${b.x_min},${b.y_min},${b.x_max},${b.y_max}`

/** Snap a click point onto the nearest edge of an object's bbox, in the same TOP/BOTTOM/LEFT/RIGHT -> UP/DOWN/LEFT/RIGHT convention _detect_equipment_ports_cv uses. */
const snapToEdge = (px: number, py: number, b: Bbox): { x: number; y: number; direction: Direction } => {
  const cx = clamp(px, b.x_min, b.x_max)
  const cy = clamp(py, b.y_min, b.y_max)
  const dTop = cy - b.y_min
  const dBottom = b.y_max - cy
  const dLeft = cx - b.x_min
  const dRight = b.x_max - cx
  const min = Math.min(dTop, dBottom, dLeft, dRight)
  if (min === dTop) return { x: cx, y: b.y_min, direction: 'UP' }
  if (min === dBottom) return { x: cx, y: b.y_max, direction: 'DOWN' }
  if (min === dLeft) return { x: b.x_min, y: cy, direction: 'LEFT' }
  return { x: b.x_max, y: cy, direction: 'RIGHT' }
}

/**
 * Gate 2 — traced-path review. Traces/branches are read-only polylines: the
 * only edit is reject, persisted as a `trace_overrides` entry through the
 * review-workspace commit endpoint (never a raw artifact PUT — the backend
 * deletes the rejected entries from stage5b_*.json itself).
 */
export function Gate2Traces({ sheet, onBack }: { sheet: Sheet; onBack: () => void }) {
  const resumeGate = useRunStore((s) => s.resumeGate)
  const retraceStage5b = useRunStore((s) => s.retraceStage5b)
  const setScreen = useRunStore((s) => s.setScreen)
  const sidebar = useResizableSidebar(240)

  const [entries, setEntries] = useState<Entry[] | null>(null)
  const [objects, setObjects] = useState<Stage4Object[]>([])
  const [showObjects, setShowObjects] = useState(true)
  const [workspace, setWorkspace] = useState<PipelineReviewWorkspaceState | null>(null)
  const [rejected, setRejected] = useState<Set<string>>(new Set())
  const [loadError, setLoadError] = useState<string | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [retracing, setRetracing] = useState(false)
  const [dirty, setDirty] = useState(false)

  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [showTraces, setShowTraces] = useState(true)
  const [showBranches, setShowBranches] = useState(true)

  // Port editing (Gate 2's "add/remove port around equipment" tool).
  const [ports, setPorts] = useState<PortsArtifact>({})
  const [portKeyByBbox, setPortKeyByBbox] = useState<Map<string, string>>(new Map())
  const [showPorts, setShowPorts] = useState(true)
  const [editingPorts, setEditingPorts] = useState(false)
  const [portsDirty, setPortsDirty] = useState(false)

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
    const orMissing = <T,>(fallback: T) => (err: unknown): T => {
      if (err instanceof APIError && err.status === 404) return fallback
      throw err
    }
    Promise.all([
      getPipelineArtifactJson<TraceResultsArtifact>(jobId, 'stage5b_trace_results.json'),
      getPipelineArtifactJson<BranchResultsArtifact>(jobId, 'stage5b_branch_trace_results.json'),
      getPipelineReviewWorkspace(jobId),
      getPipelineArtifactJson<{ objects?: Stage4Object[] }>(jobId, 'stage4_objects.json').catch(() => ({ objects: [] })),
      getPipelineArtifactJson<PortsArtifact>(jobId, 'stage5_connection_ports.json').catch(orMissing<PortsArtifact>({})),
      getPipelineArtifactJson<EquipmentArtifact>(jobId, 'stage3_equipment_bboxes.json').catch(orMissing<EquipmentArtifact>({})),
    ])
      .then(([traces, branchPayload, ws, objectsPayload, portsPayload, equipmentPayload]) => {
        if (cancelled) return
        const list: Entry[] = [
          ...Object.entries(traces ?? {}).map(([id, record]) => ({ key: `trace:${id}`, kind: 'trace' as const, id, record })),
          ...Object.entries(branchPayload?.branches ?? {}).map(([id, record]) => ({
            key: `branch:${id}`,
            kind: 'branch' as const,
            id,
            record,
          })),
        ]
        setEntries(list)
        setObjects(objectsPayload?.objects ?? [])
        setPorts(portsPayload ?? {})
        setPortKeyByBbox(new Map((equipmentPayload?.equipment ?? []).map((eq) => [bboxKey(eq.bbox), eq.id])))
        setWorkspace(ws.workspace)
        setRejected(
          new Set(
            (ws.workspace.trace_overrides ?? [])
              .filter((o) => String(o.review_state ?? o.decision ?? '') === 'rejected')
              .map((o) => `${String(o.target_type ?? o.kind ?? 'trace')}:${String(o.target_id ?? o.id ?? '')}`)
          )
        )
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : 'Could not load traces')
      })
    return () => {
      cancelled = true
    }
  }, [jobId])

  const selected = entries?.find((e) => e.key === selectedKey) ?? null

  /** The key an object's ports live under in `ports`, or null if this object isn't a port-bearing class (or its equipment bridge hasn't been saved from Gate 1 yet). */
  const portKeyFor = (o: Stage4Object): string | null => {
    if (PAGE_CONNECTION_CLASSES.has(normalizeClass(o.class_name ?? ''))) return o.id
    return portKeyByBbox.get(bboxKey(o.bbox)) ?? null
  }

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const measure = () => setViewport({ w: el.clientWidth, h: el.clientHeight })
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    measure()
    return () => ro.disconnect()
    // entries stays null behind a Spinner until load finishes, same as Gate1 — see there for why deps can't be [].
  }, [entries !== null])

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

  /** Pan so an entry's polyline is centred in the view, without changing zoom — same behaviour as Gate 1's sidebar select. */
  const focusOnEntry = useCallback(
    (entry: Entry) => {
      const segments = entry.record.segments
      if (!segments?.length) return
      let minX = Infinity
      let minY = Infinity
      let maxX = -Infinity
      let maxY = -Infinity
      for (const s of segments) {
        minX = Math.min(minX, s.x1, s.x2)
        maxX = Math.max(maxX, s.x1, s.x2)
        minY = Math.min(minY, s.y1, s.y2)
        maxY = Math.max(maxY, s.y1, s.y2)
      }
      const current = scaleRef.current
      const next = settle(
        { x: viewport.w / 2 - ((minX + maxX) / 2) * current, y: viewport.h / 2 - ((minY + maxY) / 2) * current },
        current
      )
      panRef.current = next
      setPan(next)
    },
    [settle, viewport.w, viewport.h]
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

  /** Screen point -> image-space point, accounting for the current pan/scale — same formula Gate 1 uses for drawing boxes. */
  const toImage = (e: React.MouseEvent) => {
    const rect = viewportRef.current!.getBoundingClientRect()
    return {
      x: clamp((e.clientX - rect.left - pan.x) / scale, 0, imgW),
      y: clamp((e.clientY - rect.top - pan.y) / scale, 0, imgH),
    }
  }

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
      if (!dragged) setSelectedKey(null)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  const addPort = (portKey: string, x: number, y: number, direction: Direction) => {
    setPorts((prev) => ({ ...prev, [portKey]: [...(prev[portKey] ?? []), [x, y, direction]] }))
    setPortsDirty(true)
    setDirty(true)
  }

  const removePort = (portKey: string, index: number) => {
    setPorts((prev) => ({ ...prev, [portKey]: (prev[portKey] ?? []).filter((_, i) => i !== index) }))
    setPortsDirty(true)
    setDirty(true)
  }

  const toggleReject = (entry: Entry) => {
    setRejected((prev) => {
      const next = new Set(prev)
      if (next.has(entry.key)) next.delete(entry.key)
      else next.add(entry.key)
      return next
    })
    setDirty(true)
  }

  const counts = useMemo(() => {
    const traces = entries?.filter((e) => e.kind === 'trace') ?? []
    const branches = entries?.filter((e) => e.kind === 'branch') ?? []
    return {
      traces: traces.length,
      branches: branches.length,
      rejected: rejected.size,
    }
  }, [entries, rejected])

  const save = async (): Promise<boolean> => {
    if (!jobId || !workspace) return false
    setSaving(true)
    setSaveError(null)
    try {
      const trace_overrides = [...rejected].map((key) => {
        const [target_type, target_id] = key.split(/:(.*)/s) as [Kind, string]
        return { target_id, target_type, review_state: 'rejected' }
      })
      await commitPipelineReviewWorkspace(jobId, { ...workspace, trace_overrides })
      if (portsDirty) {
        await putPipelineArtifact(jobId, 'stage5_connection_ports.json', ports)
        setPortsDirty(false)
      }
      setDirty(false)
      return true
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Could not save trace review')
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
      await resumeGate(sheet.id, 2)
    } finally {
      setConfirming(false)
    }
  }

  /** Persists any pending port edits, then re-runs stage5b_pipe_trace against them and comes back to rest at Gate 2 with fresh trace results. */
  const retrace = async () => {
    if (dirty && !(await save())) return
    setRetracing(true)
    try {
      await retraceStage5b(sheet.id)
    } finally {
      setRetracing(false)
    }
  }

  if (!jobId) return null

  if (loadError) {
    return (
      <div className="p-6">
        <Card padding={20}>
          <SectionHeader title="Could not load traced paths" description={loadError} />
        </Card>
      </div>
    )
  }

  if (!entries) {
    return (
      <div className="flex h-full items-center justify-center gap-2">
        <Spinner size="sm" />
        <span style={{ fontSize: 13, color: 'var(--muted)' }}>Loading traces…</span>
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
              {imgW > 0 && showObjects && (
                <svg
                  width={imgW}
                  height={imgH}
                  viewBox={`0 0 ${imgW} ${imgH}`}
                  style={{ position: 'absolute', left: 0, top: 0, overflow: 'visible', pointerEvents: 'none' }}
                >
                  {objects.map((o) => {
                    const portKey = editingPorts ? portKeyFor(o) : null
                    return (
                      <rect
                        key={o.id}
                        x={o.bbox.x_min}
                        y={o.bbox.y_min}
                        width={o.bbox.x_max - o.bbox.x_min}
                        height={o.bbox.y_max - o.bbox.y_min}
                        fill={portKey ? 'var(--accent)' : 'none'}
                        fillOpacity={portKey ? 0.05 : undefined}
                        stroke={classColor(o.class_name ?? '')}
                        strokeWidth={(portKey ? 2 : 1.5) / scale}
                        opacity={editingPorts && !portKey ? 0.25 : 0.55}
                        style={portKey ? { pointerEvents: 'auto', cursor: 'copy' } : undefined}
                        onClick={
                          portKey
                            ? (e) => {
                                e.stopPropagation()
                                const p = toImage(e)
                                const snapped = snapToEdge(p.x, p.y, o.bbox)
                                addPort(portKey, snapped.x, snapped.y, snapped.direction)
                              }
                            : undefined
                        }
                      />
                    )
                  })}
                </svg>
              )}
              {imgW > 0 && (
                <svg
                  width={imgW}
                  height={imgH}
                  viewBox={`0 0 ${imgW} ${imgH}`}
                  style={{ position: 'absolute', left: 0, top: 0, overflow: 'visible', pointerEvents: 'none' }}
                >
                  {entries.map((entry) => {
                    if (entry.kind === 'trace' && !showTraces) return null
                    if (entry.kind === 'branch' && !showBranches) return null
                    const isSel = entry.key === selectedKey
                    const isRejected = rejected.has(entry.key)
                    const color = isRejected ? 'var(--danger)' : entry.kind === 'trace' ? 'var(--accent)' : 'var(--warning, orange)'
                    const d = pathFor(entry.record.segments)
                    if (!d) return null
                    return (
                      <g key={entry.key} opacity={isRejected ? 0.4 : 1}>
                        <path
                          d={d}
                          fill="none"
                          stroke="transparent"
                          strokeWidth={12 / scale}
                          style={{ cursor: 'pointer', pointerEvents: 'stroke' }}
                          onClick={(e) => {
                            e.stopPropagation()
                            setSelectedKey(entry.key)
                          }}
                        />
                        <path
                          d={d}
                          fill="none"
                          stroke={color}
                          strokeWidth={(isSel ? 3.5 : 2) / scale}
                          strokeDasharray={isRejected ? `${6 / scale} ${4 / scale}` : undefined}
                          pointerEvents="none"
                        />
                      </g>
                    )
                  })}
                </svg>
              )}
              {imgW > 0 && (showPorts || editingPorts) && (
                <svg
                  width={imgW}
                  height={imgH}
                  viewBox={`0 0 ${imgW} ${imgH}`}
                  style={{ position: 'absolute', left: 0, top: 0, overflow: 'visible', pointerEvents: 'none' }}
                >
                  {Object.entries(ports).flatMap(([portKey, list]) =>
                    list.map((port, index) => {
                      const [x, y, direction] = port
                      const [dx, dy] =
                        direction === 'UP' ? [0, -1] : direction === 'DOWN' ? [0, 1] : direction === 'LEFT' ? [-1, 0] : [1, 0]
                      const stub = 14 / scale
                      return (
                        <g
                          key={`${portKey}:${index}`}
                          style={editingPorts ? { pointerEvents: 'auto', cursor: 'pointer' } : undefined}
                          onClick={
                            editingPorts
                              ? (e) => {
                                  e.stopPropagation()
                                  removePort(portKey, index)
                                }
                              : undefined
                          }
                        >
                          <line x1={x} y1={y} x2={x + dx * stub} y2={y + dy * stub} stroke="var(--accent)" strokeWidth={2 / scale} />
                          <circle cx={x} cy={y} r={editingPorts ? 6 / scale : 4 / scale} fill="var(--accent)" stroke="var(--overlay)" strokeWidth={1.5 / scale} />
                        </g>
                      )
                    })
                  )}
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
              <span style={{ width: 1, height: 20, background: 'var(--separator)', margin: '0 2px' }} />
              <button
                type="button"
                onClick={() => setShowTraces((v) => !v)}
                className="flex items-center justify-center"
                style={{
                  height: 26,
                  padding: '0 10px',
                  border: 0,
                  background: showTraces ? 'var(--accent-soft)' : 'transparent',
                  color: showTraces ? 'var(--accent-soft-fg)' : 'var(--muted)',
                  borderRadius: 'var(--r-btn)',
                  cursor: 'pointer',
                  fontSize: 12,
                }}
              >
                Traces
              </button>
              <button
                type="button"
                onClick={() => setShowBranches((v) => !v)}
                className="flex items-center justify-center"
                style={{
                  height: 26,
                  padding: '0 10px',
                  border: 0,
                  background: showBranches ? 'var(--accent-soft)' : 'transparent',
                  color: showBranches ? 'var(--accent-soft-fg)' : 'var(--muted)',
                  borderRadius: 'var(--r-btn)',
                  cursor: 'pointer',
                  fontSize: 12,
                }}
              >
                Branches
              </button>
              <button
                type="button"
                onClick={() => setShowObjects((v) => !v)}
                className="flex items-center justify-center"
                style={{
                  height: 26,
                  padding: '0 10px',
                  border: 0,
                  background: showObjects ? 'var(--accent-soft)' : 'transparent',
                  color: showObjects ? 'var(--accent-soft-fg)' : 'var(--muted)',
                  borderRadius: 'var(--r-btn)',
                  cursor: 'pointer',
                  fontSize: 12,
                }}
              >
                Objects
              </button>
              <button
                type="button"
                onClick={() => setShowPorts((v) => !v)}
                className="flex items-center justify-center"
                style={{
                  height: 26,
                  padding: '0 10px',
                  border: 0,
                  background: showPorts ? 'var(--accent-soft)' : 'transparent',
                  color: showPorts ? 'var(--accent-soft-fg)' : 'var(--muted)',
                  borderRadius: 'var(--r-btn)',
                  cursor: 'pointer',
                  fontSize: 12,
                }}
              >
                Ports
              </button>
              <span style={{ width: 1, height: 20, background: 'var(--separator)', margin: '0 2px' }} />
              <button
                type="button"
                title="Add or remove ports on equipment and page-connection boxes"
                onClick={() => {
                  setEditingPorts((v) => !v)
                  setShowObjects(true)
                  setShowPorts(true)
                }}
                className="flex items-center gap-1.5 justify-center"
                style={{
                  height: 26,
                  padding: '0 10px',
                  border: 0,
                  background: editingPorts ? 'var(--accent)' : 'transparent',
                  color: editingPorts ? 'var(--white)' : 'var(--muted)',
                  borderRadius: 'var(--r-btn)',
                  cursor: 'pointer',
                  fontSize: 12,
                }}
              >
                <Pencil size={12} strokeWidth={2} />
                Edit ports
              </button>
            </div>

            {/* Edit-ports mode banner */}
            {editingPorts && (
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
                  Click an equipment or connection box to add a port there. Click a port to remove it.
                </span>
                <Button variant="ghost" style={{ height: 26, padding: '0 10px', borderRadius: 'var(--r-btn)' }} onPress={() => setEditingPorts(false)}>
                  Done
                </Button>
              </div>
            )}
          </div>

          {/* Selected trace/branch — floating dialog */}
          <div
            aria-hidden={!selected}
            style={{
              position: 'absolute',
              top: 12,
              right: 12,
              width: 300,
              maxWidth: 'calc(100% - 24px)',
              opacity: selected ? 1 : 0,
              transform: selected ? 'scale(1)' : 'scale(0.96)',
              transition: 'opacity .16s ease, transform .16s ease',
              visibility: selected ? 'visible' : 'hidden',
              pointerEvents: selected ? 'auto' : 'none',
            }}
          >
            <div
              style={{
                background: 'var(--overlay)',
                borderRadius: 'var(--r-card)',
                padding: 16,
                boxShadow: 'inset 0 0 0 1px var(--border), 0 1px 2px rgba(0,0,0,.24), 0 14px 32px rgba(0,0,0,.38)',
              }}
            >
              {selected && (
                <>
                  <div className="mb-3 flex items-center gap-2.5">
                    <span style={{ fontSize: 14, fontWeight: 500, textTransform: 'capitalize' }}>{selected.kind}</span>
                    <span className="mono" style={{ fontSize: 12, color: 'var(--muted)' }}>{selected.id}</span>
                    <div className="flex-1" />
                    <button
                      type="button"
                      aria-label="Close"
                      onClick={() => setSelectedKey(null)}
                      className="flex items-center justify-center"
                      style={{ width: 28, height: 28, border: 0, background: 'transparent', borderRadius: 'var(--r-btn)', color: 'var(--muted)', cursor: 'pointer' }}
                    >
                      <X size={16} strokeWidth={1.8} />
                    </button>
                  </div>
                  <div className="flex flex-col gap-2.5">
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '10px 12px' }}>
                      <div className="flex flex-col gap-0.5">
                        <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--muted)' }}>Terminal</span>
                        <span style={{ fontSize: 14, fontWeight: 500 }}>{selected.record.terminal_type ?? '—'}</span>
                      </div>
                      <div className="flex flex-col gap-0.5">
                        <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--muted)' }}>Length</span>
                        <span className="mono" style={{ fontSize: 14, fontWeight: 500 }}>
                          {selected.record.trace_length_px ? `${Math.round(selected.record.trace_length_px)} px` : '—'}
                        </span>
                      </div>
                      <div className="flex flex-col gap-0.5" style={{ gridColumn: '1 / -1' }}>
                        <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--muted)' }}>Status</span>
                        <Tag tone={selected.record.status === 'ok' ? 'success' : 'warning'}>{selected.record.status ?? '—'}</Tag>
                      </div>
                    </div>
                    <div className="flex items-center justify-end gap-2">
                      {rejected.has(selected.key) ? (
                        <Button variant="primary" style={{ height: 32, borderRadius: 'var(--r-btn)' }} onPress={() => toggleReject(selected)}>
                          <RotateCcw size={15} strokeWidth={1.8} />
                          Restore
                        </Button>
                      ) : (
                        <Button
                          variant="ghost"
                          style={{ height: 32, borderRadius: 'var(--r-btn)', background: 'var(--danger-soft)', color: 'var(--danger-soft-fg)' }}
                          onPress={() => toggleReject(selected)}
                        >
                          <Trash2 size={15} strokeWidth={1.8} />
                          Reject
                        </Button>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Traces & branches list — grouped, selectable, reject inline; mirrors
            frontend_old's per-item accept/reject list rows (ObjectSidebar),
            reskinned to match Gate 1/3's list-card style. */}
        <ResizableSidebar
          width={sidebar.width}
          collapsed={sidebar.collapsed}
          onToggleCollapsed={sidebar.toggleCollapsed}
          onStartResize={sidebar.startResize}
        >
          <Card className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-hidden" padding={16}>
            <SectionHeader
              title="Traces & branches"
              description={`${counts.traces} traces · ${counts.branches} branches`}
              actions={counts.rejected > 0 ? <Tag tone="warning">{counts.rejected} rejected</Tag> : undefined}
            />
            <div className="min-h-0 flex-1 overflow-y-auto">
              {(['trace', 'branch'] as Kind[]).map((kind) => {
                const groupEntries = entries.filter((e) => e.kind === kind)
                if (groupEntries.length === 0) return null
                const groupColor = kind === 'trace' ? 'var(--accent)' : 'var(--warning, orange)'
                return (
                  <div key={kind} className="shrink-0" style={{ marginBottom: 10 }}>
                    <div className="flex items-center gap-1.5" style={{ padding: '4px 2px' }}>
                      <span className="shrink-0" style={{ width: 10, height: 10, borderRadius: 3, background: groupColor }} />
                      <span className="min-w-0 flex-1 truncate" style={{ fontSize: 12.5, fontWeight: 500 }}>
                        {kind === 'trace' ? 'Traced paths' : 'Branches'}
                      </span>
                      <span className="mono shrink-0" style={{ fontSize: 11, color: 'var(--muted)' }}>
                        {groupEntries.length}
                      </span>
                    </div>
                    {groupEntries.map((entry) => {
                      const isSel = entry.key === selectedKey
                      const isRejected = rejected.has(entry.key)
                      return (
                        <div
                          key={entry.key}
                          onClick={() => {
                            setSelectedKey(entry.key)
                            focusOnEntry(entry)
                          }}
                          style={{
                            padding: '6px 8px',
                            marginLeft: 16,
                            borderRadius: 'var(--r-chip)',
                            cursor: 'pointer',
                            background: isSel ? 'var(--accent-soft)' : 'transparent',
                            color: isSel ? 'var(--accent-soft-fg)' : 'var(--foreground)',
                            opacity: isRejected ? 0.5 : 1,
                          }}
                        >
                          <div className="flex items-center gap-2">
                            <span className="mono min-w-0 flex-1 truncate" style={{ fontSize: 12 }}>
                              {entry.id}
                            </span>
                            <span className="mono shrink-0" style={{ fontSize: 11, color: 'var(--muted)' }}>
                              {entry.record.trace_length_px ? `${Math.round(entry.record.trace_length_px)}px` : ''}
                            </span>
                          </div>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              toggleReject(entry)
                            }}
                            style={{
                              marginTop: 4,
                              height: 20,
                              padding: '0 8px',
                              border: 0,
                              borderRadius: 'var(--r-btn)',
                              background: isRejected ? 'var(--success-soft)' : 'var(--danger-soft)',
                              color: isRejected ? 'var(--success-soft-fg)' : 'var(--danger-soft-fg)',
                              cursor: 'pointer',
                              fontSize: 11,
                            }}
                          >
                            {isRejected ? 'Restore' : 'Reject'}
                          </button>
                        </div>
                      )
                    })}
                  </div>
                )
              })}
            </div>
          </Card>
        </ResizableSidebar>
      </div>

      {/* Footer */}
      <div className="flex shrink-0 items-center gap-2.5" style={{ height: 56, padding: '0 24px', borderTop: '1px solid var(--separator)' }}>
        <span className="mono" style={{ fontSize: 13, color: 'var(--muted)' }}>
          {counts.traces} traces · {counts.branches} branches
          {counts.rejected ? ` · ${counts.rejected} rejected` : ''}
          {dirty ? ' · unsaved' : ''}
        </span>
        {saveError && <span style={{ fontSize: 12, color: 'var(--danger)' }}>{saveError}</span>}
        <div className="flex-1" />
        <Button variant="ghost" style={{ height: 32, borderRadius: 'var(--r-btn)' }} onPress={onBack}>
          Back to queue
        </Button>
        <Button
          variant="ghost"
          isDisabled={retracing || saving || confirming}
          style={{ height: 32, borderRadius: 'var(--r-btn)' }}
          onPress={() => void retrace()}
        >
          {retracing ? <Spinner size="sm" /> : <RefreshCw size={14} strokeWidth={1.8} />}
          Re-trace
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
