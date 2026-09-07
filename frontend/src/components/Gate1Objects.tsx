import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Spinner } from '@heroui/react'
import { Check, Pencil, Play, Plus, Trash2, X } from 'lucide-react'
import { Card, SectionHeader, Tag } from '@/components/ui/primitives'
import { classColor, normalizeClass } from '@/lib/detectionClasses'
import { getPipelineArtifactJson, putPipelineArtifact } from '@/lib/api'
import {
  bboxFromBox,
  boxFromBbox,
  clampPan,
  fitScale as computeFit,
  HANDLE_CURSOR,
  HANDLES,
  handlePoint,
  moveBox,
  resizeBox,
  wheelIntent,
  zoomAbout,
  type Bbox,
  type Box,
} from '@/lib/viewport'
import { useRunStore, type Sheet } from '@/stores/runStore'

/** The real shape of stage4_objects.json — verified against a live job. */
type Stage4Object = {
  id: string
  class_name: string
  confidence: number
  bbox: Bbox
  source_model?: string
  source_weight?: string
}
type Stage4Artifact = { image_id?: string; pass_type?: string; objects: Stage4Object[] }

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

const FIELD: React.CSSProperties = {
  height: 34,
  padding: '0 10px',
  borderRadius: 'var(--r-field)',
  background: 'var(--field-background)',
  boxShadow: 'inset 0 0 0 1px var(--border)',
  color: 'var(--foreground)',
  fontSize: 13,
  fontFamily: 'inherit',
  border: 0,
  outline: 'none',
  width: '100%',
  boxSizing: 'border-box',
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1" style={{ minWidth: 0 }}>
      <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--muted)' }}>{label}</span>
      {children}
    </div>
  )
}

/**
 * Gate 1 — the real object/equipment review. Reads and writes
 * stage4_objects.json directly (the same plain artifact PUT the old
 * frontend used), rather than the heavier ports/layers review-workspace
 * system, which is a different, broader tool.
 */
export function Gate1Objects({ sheet, onBack }: { sheet: Sheet; onBack: () => void }) {
  const resumeGate = useRunStore((s) => s.resumeGate)

  const [objects, setObjects] = useState<Stage4Object[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [dirty, setDirty] = useState(false)

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [draft, setDraft] = useState<Stage4Object | null>(null)
  const [editing, setEditing] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [drawing, setDrawing] = useState(false)
  const [drawRect, setDrawRect] = useState<Box | null>(null)

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
    getPipelineArtifactJson<Stage4Artifact>(jobId, 'stage4_objects.json')
      .then((data) => {
        if (!cancelled) setObjects(data.objects)
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : 'Could not load objects')
      })
    return () => {
      cancelled = true
    }
  }, [jobId])

  const selected = objects?.find((o) => o.id === selectedId) ?? null
  // A plain click always lands on the read-only view; a freshly-drawn box
  // is the one exception, since it still needs classifying.
  const openInEditMode = useRef(false)
  useEffect(() => {
    setDraft(selected ? { ...selected } : null)
    setEditing(openInEditMode.current)
    openInEditMode.current = false
    setConfirmDelete(false)
  }, [selectedId, selected?.id])
  const [lastDraft, setLastDraft] = useState<Stage4Object | null>(null)
  useEffect(() => {
    if (draft) setLastDraft(draft)
  }, [draft])
  const shown = draft ?? lastDraft

  const classes = useMemo(() => {
    const names = new Set((objects ?? []).map((o) => normalizeClass(o.class_name)))
    return [...names].sort()
  }, [objects])

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const measure = () => setViewport({ w: el.clientWidth, h: el.clientHeight })
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    measure()
    return () => ro.disconnect()
  }, [])

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
      const result = zoomAbout(
        panRef.current,
        current,
        factor,
        cx,
        cy,
        imgW,
        imgH,
        viewport.w,
        viewport.h,
        0.02,
        8
      )
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
      setPan(
        settle(
          { x: origin.px + (ev.clientX - origin.mx), y: origin.py + (ev.clientY - origin.my) },
          scale
        )
      )
    }
    const up = () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
      if (!dragged && !editing) setSelectedId(null)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  const startBoxDrag = (mode: 'move' | (typeof HANDLES)[number], e: React.MouseEvent) => {
    e.stopPropagation()
    if (!draft) return
    const el = viewportRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const start: Box = boxFromBbox(draft.bbox)
    const toImage = (ev: MouseEvent | React.MouseEvent) => ({
      x: (ev.clientX - rect.left - pan.x) / scale,
      y: (ev.clientY - rect.top - pan.y) / scale,
    })
    const origin = toImage(e)
    const move = (ev: MouseEvent) => {
      const p = toImage(ev)
      const next =
        mode === 'move'
          ? moveBox(start, p.x - origin.x, p.y - origin.y, imgW, imgH)
          : resizeBox(start, mode, p.x, p.y, imgW, imgH)
      setDraft((d) => (d === null ? d : { ...d, bbox: bboxFromBox(next) }))
    }
    const up = () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  const startDrawBox = (e: React.MouseEvent) => {
    setDrawing(false)
    const el = viewportRef.current
    if (!el || !imgW || !imgH) return
    const rect = el.getBoundingClientRect()
    const toImage = (ev: MouseEvent | React.MouseEvent) => ({
      x: clamp((ev.clientX - rect.left - pan.x) / scale, 0, imgW),
      y: clamp((ev.clientY - rect.top - pan.y) / scale, 0, imgH),
    })
    const origin = toImage(e)
    const asBox = (p: { x: number; y: number }): Box => ({
      Left: Math.min(origin.x, p.x),
      Top: Math.min(origin.y, p.y),
      Width: Math.abs(p.x - origin.x),
      Height: Math.abs(p.y - origin.y),
    })
    setDrawRect(asBox(origin))
    const move = (ev: MouseEvent) => setDrawRect(asBox(toImage(ev)))
    const up = (ev: MouseEvent) => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
      const box = asBox(toImage(ev))
      setDrawRect(null)
      if (box.Width >= 6 && box.Height >= 6) {
        const id = `manual_${Date.now().toString(36)}`
        const created: Stage4Object = {
          id,
          class_name: classes[0] ?? 'object',
          confidence: 1,
          bbox: bboxFromBox(box),
          source_model: 'hitl',
        }
        setObjects((prev) => [...(prev ?? []), created])
        setDirty(true)
        openInEditMode.current = true
        setSelectedId(id)
      }
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  useEffect(() => {
    if (!drawing) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setDrawing(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [drawing])

  const applyEdit = () => {
    if (!draft) return
    setObjects((prev) => (prev ?? []).map((o) => (o.id === draft.id ? draft : o)))
    setDirty(true)
    setEditing(false)
  }

  const doDelete = () => {
    if (!shown) return
    setObjects((prev) => (prev ?? []).filter((o) => o.id !== shown.id))
    setDirty(true)
    setConfirmDelete(false)
    setSelectedId(null)
  }

  const save = async (): Promise<boolean> => {
    if (!jobId || !objects) return false
    setSaving(true)
    setSaveError(null)
    try {
      await putPipelineArtifact(jobId, 'stage4_objects.json', { objects })
      setDirty(false)
      return true
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Could not save objects')
      return false
    } finally {
      setSaving(false)
    }
  }

  const saveAndConfirm = async () => {
    setConfirming(true)
    try {
      if (dirty && !(await save())) return
      await resumeGate(sheet.id, 1)
      onBack()
    } finally {
      setConfirming(false)
    }
  }

  if (!jobId) return null

  if (loadError) {
    return (
      <div className="p-6">
        <Card padding={20}>
          <SectionHeader title="Could not load stage 4 objects" description={loadError} />
        </Card>
      </div>
    )
  }

  if (!objects) {
    return (
      <div className="flex h-full items-center justify-center gap-2">
        <Spinner size="sm" />
        <span style={{ fontSize: 13, color: 'var(--muted)' }}>Loading objects…</span>
      </div>
    )
  }

  const visible = objects

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
              cursor: drawing ? 'crosshair' : 'grab',
              userSelect: 'none',
              WebkitUserSelect: 'none',
            }}
            onMouseDown={drawing ? startDrawBox : startPan}
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
                  style={{ position: 'absolute', left: 0, top: 0, overflow: 'visible' }}
                >
                  {visible.map((o) => {
                    const isSel = o.id === selectedId
                    const box = isSel && editing && draft ? boxFromBbox(draft.bbox) : boxFromBbox(o.bbox)
                    const muted = editing && !isSel
                    const color = muted ? 'var(--muted)' : classColor(o.class_name)
                    return (
                      <g key={o.id} onMouseDown={(e) => e.stopPropagation()} opacity={muted ? 0.3 : 1}>
                        <rect
                          x={box.Left}
                          y={box.Top}
                          width={box.Width}
                          height={box.Height}
                          fill="transparent"
                          stroke="transparent"
                          strokeWidth={12 / scale}
                          style={{
                            cursor: muted ? 'default' : isSel && editing ? 'move' : 'pointer',
                            pointerEvents: muted ? 'none' : undefined,
                          }}
                          onClick={() => !editing && setSelectedId(o.id)}
                          onMouseDown={(e) => {
                            if (isSel && editing) startBoxDrag('move', e)
                            else e.stopPropagation()
                          }}
                        />
                        <rect
                          x={box.Left}
                          y={box.Top}
                          width={box.Width}
                          height={box.Height}
                          fill={isSel ? `${muted ? 'transparent' : classColor(o.class_name)}22` : 'transparent'}
                          stroke={color}
                          strokeWidth={(isSel ? 3 : 1.6) / scale}
                          strokeDasharray={isSel && editing ? `${6 / scale} ${4 / scale}` : undefined}
                          pointerEvents="none"
                        />
                        {isSel && editing && (
                          <g>
                            {HANDLES.map((h) => {
                              const pt = handlePoint(box, h)
                              const size = 9 / scale
                              return (
                                <rect
                                  key={h}
                                  x={pt.x - size / 2}
                                  y={pt.y - size / 2}
                                  width={size}
                                  height={size}
                                  fill="var(--white)"
                                  stroke="var(--accent)"
                                  strokeWidth={2 / scale}
                                  style={{ cursor: HANDLE_CURSOR[h] }}
                                  onMouseDown={(e) => startBoxDrag(h, e)}
                                />
                              )
                            })}
                          </g>
                        )}
                      </g>
                    )
                  })}
                  {drawRect && (
                    <rect
                      x={drawRect.Left}
                      y={drawRect.Top}
                      width={drawRect.Width}
                      height={drawRect.Height}
                      fill="var(--accent)"
                      fillOpacity={0.12}
                      stroke="var(--accent)"
                      strokeWidth={1.6 / scale}
                      strokeDasharray={`${6 / scale} ${4 / scale}`}
                      pointerEvents="none"
                    />
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
                style={{
                  height: 26,
                  padding: '0 10px',
                  border: 0,
                  background: 'transparent',
                  borderRadius: 'var(--r-btn)',
                  color: 'var(--foreground)',
                  cursor: 'pointer',
                  fontSize: 12,
                }}
              >
                Fit
              </button>
            </div>
          </div>

          {/* Selected object — floating dialog, matching the Detection pattern */}
          <div
            aria-hidden={!draft}
            style={{
              position: 'absolute',
              top: 12,
              right: 12,
              width: 340,
              maxWidth: 'calc(100% - 24px)',
              maxHeight: 'calc(100% - 24px)',
              overflowY: 'auto',
              opacity: draft ? 1 : 0,
              transform: draft ? 'scale(1)' : 'scale(0.96)',
              transition: 'opacity .16s ease, transform .16s ease',
              visibility: shown ? 'visible' : 'hidden',
              pointerEvents: draft ? 'auto' : 'none',
            }}
          >
            <div
              style={{
                background: 'var(--overlay)',
                borderRadius: 'var(--r-card)',
                padding: 16,
                boxShadow:
                  'inset 0 0 0 1px var(--border), 0 1px 2px rgba(0,0,0,.24), 0 14px 32px rgba(0,0,0,.38)',
              }}
            >
              {shown && (
                <>
                  <div className="mb-3 flex items-center gap-2.5">
                    <span style={{ fontSize: 14, fontWeight: 500 }}>Selected object</span>
                    <span
                      className="shrink-0"
                      style={{ width: 12, height: 12, borderRadius: 4, background: classColor(shown.class_name) }}
                    />
                    <Tag tone={shown.confidence >= 0.8 ? 'success' : 'warning'}>
                      {shown.confidence.toFixed(2)}
                    </Tag>
                    <div className="flex-1" />
                    <button
                      type="button"
                      aria-label="Close selected object"
                      onClick={() => setSelectedId(null)}
                      className="flex items-center justify-center"
                      style={{
                        width: 28,
                        height: 28,
                        border: 0,
                        background: 'transparent',
                        borderRadius: 'var(--r-btn)',
                        color: 'var(--muted)',
                        cursor: 'pointer',
                      }}
                    >
                      <X size={16} strokeWidth={1.8} />
                    </button>
                  </div>

                  {confirmDelete ? (
                    <div className="flex flex-col gap-3">
                      <span style={{ fontSize: 13, lineHeight: '19px' }}>
                        Delete this <strong>{shown.class_name}</strong>? This is undone by re-opening the
                        gate without saving.
                      </span>
                      <div className="flex items-center justify-end gap-2">
                        <Button variant="ghost" style={{ height: 32, borderRadius: 'var(--r-btn)' }} onPress={() => setConfirmDelete(false)}>
                          Cancel
                        </Button>
                        <Button
                          variant="ghost"
                          style={{ height: 32, borderRadius: 'var(--r-btn)', background: 'var(--danger)', color: 'var(--white)' }}
                          onPress={doDelete}
                        >
                          <Trash2 size={15} strokeWidth={1.8} />
                          Delete
                        </Button>
                      </div>
                    </div>
                  ) : editing ? (
                    <div className="flex flex-col gap-2.5">
                      <Field label="Class">
                        <input
                          list="gate1-classes"
                          value={shown.class_name}
                          onChange={(e) => setDraft({ ...shown, class_name: e.target.value })}
                          style={FIELD}
                          aria-label="Class"
                        />
                        <datalist id="gate1-classes">
                          {classes.map((c) => (
                            <option key={c} value={c} />
                          ))}
                        </datalist>
                      </Field>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 8 }}>
                        {(['x_min', 'y_min', 'x_max', 'y_max'] as const).map((k) => (
                          <Field key={k} label={{ x_min: 'x', y_min: 'y', x_max: 'x2', y_max: 'y2' }[k]}>
                            <input
                              type="number"
                              aria-label={k}
                              value={shown.bbox[k]}
                              onChange={(e) =>
                                setDraft({ ...shown, bbox: { ...shown.bbox, [k]: Number(e.target.value) } })
                              }
                              style={FIELD}
                            />
                          </Field>
                        ))}
                      </div>
                      <div className="flex items-center justify-end gap-2">
                        <Button
                          variant="ghost"
                          style={{ height: 32, borderRadius: 'var(--r-btn)' }}
                          onPress={() => {
                            setDraft(selected ? { ...selected } : null)
                            setEditing(false)
                          }}
                        >
                          Cancel
                        </Button>
                        <Button variant="primary" style={{ height: 32, borderRadius: 'var(--r-btn)' }} onPress={applyEdit}>
                          <Check size={15} strokeWidth={2.2} />
                          OK
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-2.5">
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '10px 12px' }}>
                        <div className="flex flex-col gap-0.5">
                          <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--muted)' }}>Class</span>
                          <span style={{ fontSize: 14, fontWeight: 500 }}>{shown.class_name}</span>
                        </div>
                        <div className="flex flex-col gap-0.5">
                          <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--muted)' }}>Bounding box</span>
                          <span className="mono" style={{ fontSize: 14, fontWeight: 500 }}>
                            {shown.bbox.x_min}, {shown.bbox.y_min}, {shown.bbox.x_max - shown.bbox.x_min} ×{' '}
                            {shown.bbox.y_max - shown.bbox.y_min}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center justify-end gap-2">
                        <Button
                          variant="ghost"
                          style={{ height: 32, borderRadius: 'var(--r-btn)', background: 'var(--danger-soft)', color: 'var(--danger-soft-fg)' }}
                          onPress={() => setConfirmDelete(true)}
                        >
                          <Trash2 size={15} strokeWidth={1.8} />
                          Delete
                        </Button>
                        <Button variant="primary" style={{ height: 32, borderRadius: 'var(--r-btn)' }} onPress={() => setEditing(true)}>
                          <Pencil size={15} strokeWidth={1.8} />
                          Edit
                        </Button>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div
        className="flex shrink-0 items-center gap-2.5"
        style={{ height: 56, padding: '0 24px', borderTop: '1px solid var(--separator)' }}
      >
        <span className="mono" style={{ fontSize: 13, color: 'var(--muted)' }}>
          {objects.length} objects{dirty ? ' · unsaved' : ''}
        </span>
        <Button
          variant="ghost"
          isIconOnly
          aria-pressed={drawing}
          aria-label={drawing ? 'Cancel drawing (Esc)' : 'Add an object the detector missed'}
          isDisabled={editing}
          style={{
            width: 28,
            height: 28,
            borderRadius: 'var(--r-btn)',
            background: drawing ? 'var(--accent-soft)' : undefined,
            color: drawing ? 'var(--accent-soft-fg)' : undefined,
          }}
          onPress={() => setDrawing((d) => !d)}
        >
          <Plus size={15} strokeWidth={2} />
        </Button>
        {saveError && <span style={{ fontSize: 12, color: 'var(--danger)' }}>{saveError}</span>}
        <div className="flex-1" />
        <Button variant="ghost" style={{ height: 32, borderRadius: 'var(--r-btn)' }} onPress={onBack}>
          Back to queue
        </Button>
        <Button
          variant="secondary"
          isDisabled={!dirty || saving}
          style={{ height: 32, borderRadius: 'var(--r-btn)' }}
          onPress={() => void save()}
        >
          {saving ? <Spinner size="sm" /> : 'Save'}
        </Button>
        <Button
          variant="primary"
          isDisabled={confirming}
          style={{ height: 32, borderRadius: 'var(--r-btn)' }}
          onPress={() => void saveAndConfirm()}
        >
          {confirming ? <Spinner size="sm" /> : <Play size={14} strokeWidth={1.6} />}
          Save &amp; continue
        </Button>
      </div>
    </div>
  )
}
