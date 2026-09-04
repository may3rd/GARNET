import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Spinner } from '@heroui/react'
import { Check, ChevronDown, ChevronRight, Maximize2, Minus, Monitor, Pencil, Plus, RotateCw, Scan, Tag as TagIcon, Trash2, X } from 'lucide-react'
import { Card, SectionHeader, Tag, Toggle } from '@/components/ui/primitives'
import { classColor, summarizeClasses, normalizeClass } from '@/lib/detectionClasses'
import { exportCoco } from '@/lib/exportFormats'
import { exportResultsToExcel } from '@/lib/api'
import { availabilityOf, controlHeight, GIVES_WAY, useWidth } from '@/lib/responsive'
import {
  clampPan,
  fitScale as computeFit,
  HANDLE_CURSOR,
  HANDLES,
  handlePoint,
  moveBox,
  resizeBox,
  wheelPixels,
  zoomAbout,
  type Box,
  type Handle,
} from '@/lib/viewport'
import { useRunStore } from '@/stores/runStore'
import type { DetectedObject } from '@/types'

const MINIMAP = { w: 150, h: 100 }
const ZOOM_RANGE = { min: 0.02, max: 8 }
/** The class legend may shrink to fit the window, but not below this. */
const CLASSES_MIN_HEIGHT = 220
/** How far the object sheet is pushed down when parked, in px. */
const PARKED_OFFSET = 360

const FIELD: React.CSSProperties = {
  height: 36,
  padding: '0 12px',
  borderRadius: 'var(--r-field)',
  background: 'var(--field-background)',
  boxShadow: 'inset 0 0 0 1px var(--border)',
  color: 'var(--foreground)',
  fontSize: 14,
  fontFamily: 'inherit',
  border: 0,
  outline: 'none',
  width: '100%',
  boxSizing: 'border-box',
}

type LabelMode = 'off' | 'class' | 'tag'

const LABEL_MODES: { key: LabelMode; label: string }[] = [
  { key: 'off', label: 'Hidden' },
  { key: 'class', label: 'Class name' },
  { key: 'tag', label: 'Tag / text' },
]

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

/** Where an object-fit:contain image actually sits inside its box. */
function containRect(boxW: number, boxH: number, imgW: number, imgH: number) {
  const s = Math.min(boxW / imgW, boxH / imgH)
  const w = imgW * s
  const h = imgH * s
  return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h }
}

function LabelledField({
  label,
  children,
  flex,
  width,
}: {
  label: string
  children: React.ReactNode
  flex?: number
  width?: number
}) {
  return (
    <div
      className="flex flex-col gap-1.5"
      style={{ flex: width ? undefined : (flex ?? 1), width, minWidth: 0 }}
    >
      <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--muted)' }}>{label}</div>
      {children}
    </div>
  )
}

/** A label over its value as text, for the panel's read-only state. */
function ReadOnly({
  label,
  value,
  mono,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--muted)' }}>{label}</span>
      <span
        className={mono ? 'mono truncate' : 'truncate'}
        style={{ fontSize: 14, fontWeight: 500 }}
      >
        {value}
      </span>
    </div>
  )
}

function download(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  URL.revokeObjectURL(url)
}

export function DetectionResults() {
  const sheets = useRunStore((s) => s.sheets)
  const selectedSheetId = useRunStore((s) => s.selectedSheetId)
  const selectSheet = useRunStore((s) => s.selectSheet)
  const config = useRunStore((s) => s.config)
  const setConfig = useRunStore((s) => s.setConfig)
  const runDetectionFor = useRunStore((s) => s.runDetectionFor)
  const updateObject = useRunStore((s) => s.updateObject)
  const deleteObject = useRunStore((s) => s.deleteObject)
  const setScreen = useRunStore((s) => s.setScreen)
  const weightFiles = useRunStore((s) => s.weightFiles)
  const loadWeightFiles = useRunStore((s) => s.loadWeightFiles)
  const width = useWidth()
  const availability = availabilityOf('detection', width)
  const canEdit = availability === 'full'
  const ctlH = controlHeight(width)

  const detectionSheets = sheets.filter((s) => s.task === 'detection')
  const sheet =
    sheets.find((s) => s.id === selectedSheetId && s.task === 'detection') ?? detectionSheets[0]

  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [draft, setDraft] = useState<DetectedObject | null>(null)
  const [zoom, setZoom] = useState<number | null>(null) // null = fit
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [busy, setBusy] = useState(false)
  // Off / class name / tag text. With OCR off the backend fills Text with a
  // placeholder ("instrument tag - no. 1"), so the class is the useful label
  // and the tag is only worth showing when OCR actually ran.
  const [labelMode, setLabelMode] = useState<LabelMode>('off')
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [editing, setEditing] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const viewportRef = useRef<HTMLDivElement>(null)
  const [viewport, setViewport] = useState({ w: 0, h: 0 })

  const objects = sheet?.detection?.objects ?? []
  const imgW = sheet?.detection?.image_width ?? sheet?.size?.width ?? 0
  const imgH = sheet?.detection?.image_height ?? sheet?.size?.height ?? 0

  const classes = useMemo(() => summarizeClasses(objects), [objects])
  /** Objects per class, for the class > object tree. */
  const byClass = useMemo(() => {
    const map = new Map<string, DetectedObject[]>()
    objects.forEach((o) => {
      const k = normalizeClass(o.Object)
      const list = map.get(k)
      if (list) list.push(o)
      else map.set(k, [o])
    })
    map.forEach((list) => list.sort((a, b) => b.Score - a.Score))
    return map
  }, [objects])
  const selected = objects.find((o) => o.Index === selectedIndex) ?? null

  useEffect(() => {
    setDraft(selected ? { ...selected } : null)
    setEditing(false)
    setConfirmDelete(false)
  }, [selectedIndex, selected?.Index])

  // The panel keeps rendering the last object while it slides back down, so
  // the content does not vanish mid-animation. It is clipped once parked.
  const [lastDraft, setLastDraft] = useState<DetectedObject | null>(null)
  useEffect(() => {
    if (draft) setLastDraft(draft)
  }, [draft])
  const shown = draft ?? lastDraft

  useEffect(() => {
    void loadWeightFiles()
  }, [loadWeightFiles])

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const measure = () => setViewport({ w: el.clientWidth, h: el.clientHeight })
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    measure()
    return () => ro.disconnect()
  }, [sheet?.id])

  const fitScale = computeFit(imgW, imgH, viewport.w, viewport.h)
  const scale = zoom ?? fitScale

  /** Every pan goes through here, so the sheet can never expose a gutter. */
  const settle = useCallback(
    (p: { x: number; y: number }, atScale: number) =>
      clampPan(p, atScale, imgW, imgH, viewport.w, viewport.h),
    [imgW, imgH, viewport.w, viewport.h]
  )

  // Re-settle whenever the frame or the scale changes, so a resize or a zoom
  // cannot leave the sheet parked off-centre with empty space beside it.
  useEffect(() => {
    setPan((p) => settle(p, scale))
  }, [settle, scale])
  const visible = objects.filter((o) => !hidden.has(normalizeClass(o.Object)))

  // Mirrors of the live view, so a zoom can read the current values without
  // nesting one state updater inside another.
  const panRef = useRef(pan)
  panRef.current = pan
  const scaleRef = useRef(scale)
  scaleRef.current = scale

  /**
   * Zoom about a point in viewport coordinates, keeping it under the cursor.
   *
   * Both values are computed up front and set separately. The previous version
   * called setZoom inside the setPan updater; updaters must be pure, and React
   * double-invokes them under StrictMode, so the zoom was applied twice per
   * event — which is what made trackpad zooming lurch.
   */
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
        ZOOM_RANGE.min,
        ZOOM_RANGE.max
      )
      if (result.scale === current) return
      // Update the mirrors immediately so several events in one frame compose.
      scaleRef.current = result.scale
      panRef.current = result.pan
      setZoom(result.scale)
      setPan(result.pan)
    },
    [imgW, imgH, viewport.w, viewport.h]
  )

  // Wheel zoom needs a non-passive listener to be able to preventDefault, so
  // the page does not scroll while zooming the sheet.
  useEffect(() => {
    const el = viewportRef.current
    if (!el) return

    // A trackpad fires wheel events far faster than the screen refreshes, and
    // each one re-renders every box. Deltas are accumulated and applied once
    // per frame instead.
    let pending = 0
    let originX = 0
    let originY = 0
    let frame = 0

    const flush = () => {
      frame = 0
      const delta = pending
      pending = 0
      if (delta !== 0) zoomAt(Math.exp(-delta * 0.0015), originX, originY)
    }

    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      originX = e.clientX - rect.left
      originY = e.clientY - rect.top
      // Clamp the accumulator too, not just each event: if the frame callback
      // is throttled (a hidden tab, a busy main thread) the deltas would
      // otherwise pile up and land as one lurch when it finally runs.
      pending = clamp(pending + wheelPixels(e.deltaY, e.deltaMode, e.ctrlKey, rect.height), -240, 240)
      if (!frame) frame = requestAnimationFrame(flush)
    }

    el.addEventListener('wheel', onWheel, { passive: false })
    return () => {
      el.removeEventListener('wheel', onWheel)
      if (frame) cancelAnimationFrame(frame)
    }
  }, [zoomAt])

  const startPan = (e: React.MouseEvent) => {
    const origin = { px: pan.x, py: pan.y, mx: e.clientX, my: e.clientY }
    // Distinguish a click from a drag: only a click clears the selection, so
    // panning the sheet never loses what you were looking at.
    let dragged = false
    const move = (ev: MouseEvent) => {
      if (Math.abs(ev.clientX - origin.mx) > 3 || Math.abs(ev.clientY - origin.my) > 3) {
        dragged = true
      }
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
      // Boxes and the toolbar stop propagation, so reaching here means empty
      // canvas. Editing is left alone: clearing it would discard the edit.
      if (!dragged && !editing) setSelectedIndex(null)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  /** Centre the view on the image point under a minimap position. */
  const panFromMinimap = useCallback(
    (mx: number, my: number) => {
      if (!imgW || !imgH) return
      const r = containRect(MINIMAP.w, MINIMAP.h, imgW, imgH)
      const fx = clamp((mx - r.x) / r.w, 0, 1)
      const fy = clamp((my - r.y) / r.h, 0, 1)
      setPan(
        settle(
          { x: viewport.w / 2 - fx * imgW * scale, y: viewport.h / 2 - fy * imgH * scale },
          scale
        )
      )
    },
    [imgW, imgH, scale, viewport.w, viewport.h, settle]
  )

  const startMinimapDrag = (e: React.MouseEvent<HTMLDivElement>) => {
    e.stopPropagation()
    const box = e.currentTarget.getBoundingClientRect()
    panFromMinimap(e.clientX - box.left, e.clientY - box.top)
    const move = (ev: MouseEvent) => panFromMinimap(ev.clientX - box.left, ev.clientY - box.top)
    const up = () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  /**
   * Drag an anchor, or the box body, in image coordinates. pan and scale are
   * captured at mousedown: panning is suppressed while a box drag is live, so
   * they cannot change underneath it.
   */
  const startBoxDrag = (mode: Handle | 'move', e: React.MouseEvent) => {
    e.stopPropagation()
    if (!canEdit || !draft) return
    const el = viewportRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const start: Box = { ...draft }
    const toImage = (ev: MouseEvent | React.MouseEvent) => ({
      x: (ev.clientX - rect.left - pan.x) / scale,
      y: (ev.clientY - rect.top - pan.y) / scale,
    })
    const origin = toImage(e)

    const move = (ev: MouseEvent) => {
      const p = toImage(ev)
      setDraft((d) =>
        d === null
          ? d
          : {
              ...d,
              ...(mode === 'move'
                ? moveBox(start, p.x - origin.x, p.y - origin.y, imgW, imgH)
                : resizeBox(start, mode, p.x, p.y, imgW, imgH)),
            }
      )
    }
    const up = () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  const toggleClass = (name: string) =>
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })

  const onExport = async (kind: 'excel' | 'json' | 'coco') => {
    if (!sheet?.detection) return
    const stem = sheet.label.replace(/\.[^.]+$/, '')
    if (kind === 'json') {
      download(
        `${stem}-detection.json`,
        new Blob([JSON.stringify(sheet.detection, null, 2)], { type: 'application/json' })
      )
      return
    }
    if (kind === 'coco') {
      exportCoco(objects, imgW, imgH, `${stem}.png`)
      return
    }
    setBusy(true)
    try {
      const blob = await exportResultsToExcel(
        [{ file_name: `${stem}.png`, objects: objects as unknown as Record<string, unknown>[] }],
        `${stem}-detection.xlsx`
      )
      download(`${stem}-detection.xlsx`, blob)
    } finally {
      setBusy(false)
    }
  }

  // The contract from the Breakpoints artboard: box editing "needs a pointer
  // and room", so this screen is not offered on a phone rather than shipping
  // a version that fails in the field.
  if (availability === 'not-offered') {
    return (
      <div className="flex h-full flex-col gap-4 p-5">
        <Card padding={20} className="flex flex-col gap-3">
          <div className="flex items-start gap-3">
            <span style={{ color: 'var(--warning)', marginTop: 2 }}>
              <Monitor size={18} strokeWidth={1.6} />
            </span>
            <div>
              <div style={{ fontSize: 14, fontWeight: 500 }}>Open this on a larger screen</div>
              <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 2 }}>
                {GIVES_WAY.detection} Detection results needs a tablet or a desktop.
              </div>
            </div>
          </div>
          <Button
            variant="secondary"
            style={{ alignSelf: 'flex-start', height: ctlH, borderRadius: 'var(--r-btn)' }}
            onPress={() => setScreen('sheets')}
          >
            Back to sheets
          </Button>
        </Card>
      </div>
    )
  }

  if (!sheet) {
    return (
      <div className="flex h-full flex-col gap-4 p-6">
        <Card padding={20}>
          <SectionHeader
            title="No detection sheet"
            description="Stage a sheet, pick Detection on the Task screen, and start the run."
          />
          <div className="mt-3">
            <Button
              variant="secondary"
              style={{ height: 32, borderRadius: 'var(--r-btn)' }}
              onPress={() => setScreen('sheets')}
            >
              Go to sheets
            </Button>
          </div>
        </Card>
      </div>
    )
  }

  // Minimap viewport indicator, in minimap pixels.
  const mm = containRect(MINIMAP.w, MINIMAP.h, imgW || 1, imgH || 1)
  const viewBoxOnMinimap = {
    left: mm.x + clamp(-pan.x / scale / (imgW || 1), 0, 1) * mm.w,
    top: mm.y + clamp(-pan.y / scale / (imgH || 1), 0, 1) * mm.h,
    width: clamp(viewport.w / scale / (imgW || 1), 0.02, 1) * mm.w,
    height: clamp(viewport.h / scale / (imgH || 1), 0.02, 1) * mm.h,
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div
        className="flex min-h-0 flex-1 gap-4"
        style={{ padding: width === 'desktop' ? '16px 24px' : '12px 16px' }}
      >
        {/*
          Canvas column — relative so the object sheet can slide up over it,
          and overflow-hidden so the sheet is genuinely gone when parked, not
          hovering over the footer below.
        */}
        <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
          <div
            ref={viewportRef}
            className="relative min-h-0 flex-1 overflow-hidden"
            style={{
              background: 'var(--surface-tertiary)',
              borderRadius: 'var(--r-table)',
              cursor: 'grab',
              // Dragging across the sheet would otherwise select the SVG label
              // text and leave it highlighted.
              userSelect: 'none',
              WebkitUserSelect: 'none',
            }}
            onMouseDown={startPan}
          >
            {sheet.progress && (
              <div
                className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-4"
                style={{
                  // A scrim over the canvas: the sheet underneath is stale
                  // while a run is in flight, so it should read as inactive.
                  background: 'color-mix(in oklab, var(--foreground) 28%, transparent)',
                  backdropFilter: 'blur(1.5px)',
                  cursor: 'progress',
                }}
                // Swallow drags and clicks so the stale canvas cannot be
                // panned or have a box selected mid-run.
                onMouseDown={(e) => e.stopPropagation()}
              >
                {/* xl is 40px, which reads as an afterthought on a canvas this
                    big, so it is scaled up — but on a WRAPPER, never on the
                    spinner itself. Its `spin` keyframes animate `transform`
                    and only declare 100%, so a transform of our own on the
                    same element becomes the implicit 0% and it interpolates
                    scale->rotate instead of spinning. */}
                <div style={{ transform: 'scale(1.6)', marginBottom: 14 }}>
                  <Spinner size="xl" color="current" style={{ color: 'var(--white)' }} />
                </div>
                <div className="flex flex-col items-center gap-1">
                  <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--white)' }}>
                    {sheet.progress.step}
                  </span>
                  <span
                    className="mono"
                    style={{ fontSize: 12, color: 'color-mix(in oklab, var(--white) 75%, transparent)' }}
                  >
                    {sheet.label}
                  </span>
                </div>
              </div>
            )}

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
                style={{
                  display: 'block',
                  background: '#ffffff',
                  // Explicit, and maxWidth:none, because Tailwind's preflight
                  // sets img{max-width:100%;height:auto} — which shrinks the
                  // raster while the SVG overlay keeps its natural size, so the
                  // boxes drift off the drawing.
                  width: imgW,
                  height: imgH,
                  maxWidth: 'none',
                }}
              />
              {imgW > 0 && (
                <svg
                  width={imgW}
                  height={imgH}
                  viewBox={`0 0 ${imgW} ${imgH}`}
                  style={{ position: 'absolute', left: 0, top: 0, overflow: 'visible' }}
                >
                  {visible.map((o) => {
                    const isSel = o.Index === selectedIndex
                    // In edit mode the box being edited comes from the draft,
                    // so the outline tracks the drag live.
                    const box: Box = isSel && editing && draft ? draft : o
                    // Everything else is muted while editing, so the box under
                    // the cursor is unambiguous.
                    const muted = editing && !isSel
                    const color = muted ? 'var(--muted)' : classColor(o.Object)
                    return (
                      <g key={o.Index} onMouseDown={(e) => e.stopPropagation()} opacity={muted ? 0.3 : 1}>
                        {/* Wide transparent stroke so thin boxes stay clickable. */}
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
                          onClick={() => !editing && setSelectedIndex(o.Index)}
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
                          fill={isSel ? `${muted ? 'transparent' : classColor(o.Object)}22` : 'transparent'}
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
                        {labelMode !== 'off' && !muted && (
                          // Stroke-then-fill gives the text a white outline so
                          // it stays readable over the drawing's own linework.
                          <text
                            x={box.Left}
                            y={box.Top - 5 / scale}
                            fontSize={13 / scale}
                            fill={color}
                            stroke="#ffffff"
                            strokeWidth={3 / scale}
                            paintOrder="stroke"
                            style={{ fontWeight: 600 }}
                            pointerEvents="none"
                          >
                            {labelMode === 'tag' ? o.Text?.trim() || o.Object : o.Object}
                          </text>
                        )}
                      </g>
                    )
                  })}
                </svg>
              )}
            </div>

            {/* Floating toolbar */}
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
              {[
                {
                  icon: <Plus size={16} strokeWidth={1.6} />,
                  label: 'Zoom in',
                  run: () => zoomAt(1.25, viewport.w / 2, viewport.h / 2),
                },
                {
                  icon: <Minus size={16} strokeWidth={1.6} />,
                  label: 'Zoom out',
                  run: () => zoomAt(1 / 1.25, viewport.w / 2, viewport.h / 2),
                },
                {
                  icon: <Maximize2 size={16} strokeWidth={1.6} />,
                  label: 'Fit',
                  run: () => {
                    setZoom(null)
                    setPan(settle({ x: 0, y: 0 }, fitScale))
                  },
                },
                {
                  icon: <Scan size={16} strokeWidth={1.6} />,
                  label: 'Actual size',
                  run: () => {
                    setZoom(1)
                    setPan(settle({ x: 0, y: 0 }, 1))
                  },
                },
              ].map((b) => (
                <button
                  key={b.label}
                  type="button"
                  title={b.label}
                  aria-label={b.label}
                  onClick={b.run}
                  className="flex items-center justify-center"
                  style={{
                    width: ctlH,
                    height: ctlH,
                    border: 0,
                    background: 'transparent',
                    borderRadius: 'var(--r-btn)',
                    color: 'var(--foreground)',
                    cursor: 'pointer',
                  }}
                >
                  {b.icon}
                </button>
              ))}
              <span
                style={{ width: 1, height: 20, background: 'var(--separator)', margin: '0 4px' }}
              />
              <button
                type="button"
                title={labelMode === 'off' ? 'Show object labels' : 'Hide object labels'}
                aria-label="Toggle labels"
                aria-pressed={labelMode !== 'off'}
                onClick={() => setLabelMode((m) => (m === 'off' ? 'class' : 'off'))}
                className="flex items-center justify-center"
                style={{
                  width: ctlH,
                  height: ctlH,
                  border: 0,
                  borderRadius: 'var(--r-btn)',
                  cursor: 'pointer',
                  background: labelMode !== 'off' ? 'var(--accent-soft)' : 'transparent',
                  color: labelMode !== 'off' ? 'var(--accent-soft-fg)' : 'var(--foreground)',
                }}
              >
                <TagIcon size={16} strokeWidth={1.6} />
              </button>
              <span
                style={{ width: 1, height: 20, background: 'var(--separator)', margin: '0 4px' }}
              />
              <span className="mono" style={{ padding: '0 8px', fontSize: 12, color: 'var(--muted)' }}>
                {Math.round(scale * 100)}%
              </span>
            </div>

            {/* Minimap — click or drag to move the view */}
            {imgW > 0 && viewport.w > 0 && (
              <div
                role="button"
                tabIndex={-1}
                aria-label="Minimap — drag to pan"
                title="Drag to pan"
                className="absolute overflow-hidden"
                style={{
                  right: 12,
                  bottom: 12,
                  width: MINIMAP.w,
                  height: MINIMAP.h,
                  background: '#ffffff',
                  borderRadius: 10,
                  boxShadow: 'inset 0 0 0 1px var(--border)',
                  cursor: 'grab',
                }}
                onMouseDown={startMinimapDrag}
              >
                <img
                  src={sheet.previewUrl}
                  alt=""
                  draggable={false}
                  style={{ width: '100%', height: '100%', objectFit: 'contain', maxWidth: 'none' }}
                />
                <div
                  style={{
                    position: 'absolute',
                    pointerEvents: 'none',
                    border: '2px solid var(--accent)',
                    background: 'color-mix(in oklab, var(--accent) 12%, transparent)',
                    borderRadius: 2,
                    left: viewBoxOnMinimap.left,
                    top: viewBoxOnMinimap.top,
                    width: viewBoxOnMinimap.width,
                    height: viewBoxOnMinimap.height,
                  }}
                />
              </div>
            )}
          </div>

          {/* Selected object — slides up from the bottom, absent when parked */}
          <div
            aria-hidden={!draft}
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              bottom: 0,
              // A fixed offset, not a percentage: the panel's own height is not
              // a reliable basis to translate by, and the parent clips it.
              transform: draft ? 'translateY(0)' : `translateY(${PARKED_OFFSET}px)`,
              transition: 'transform .22s cubic-bezier(.32,.72,0,1)',
              visibility: shown ? 'visible' : 'hidden',
              pointerEvents: draft ? 'auto' : 'none',
            }}
          >
            <div
              style={{
                background: 'var(--overlay)',
                // Tighter than a card's 32px: this is a docked strip, and a
                // large radius on a full-width panel reads as a floating pill.
                borderRadius: 14,
                padding: 16,
                boxShadow: 'inset 0 0 0 1px var(--border), 0 -8px 32px rgba(0,0,0,.16)',
              }}
            >
              {shown && (
                <>
                  <div className="mb-3 flex items-center gap-2.5">
                    <span style={{ fontSize: 14, fontWeight: 500 }}>Selected object</span>
                    <span
                      className="shrink-0"
                      style={{
                        width: 12,
                        height: 12,
                        borderRadius: 4,
                        background: classColor(shown.Object),
                      }}
                    />
                    <span className="mono" style={{ fontSize: 12, color: 'var(--muted)' }}>
                      #{shown.Index}
                    </span>
                    <Tag tone={shown.Score >= 0.8 ? 'success' : 'warning'}>
                      {shown.Score.toFixed(2)}
                    </Tag>
                    <div className="flex-1" />
                    <button
                      type="button"
                      aria-label="Close selected object"
                      title="Close"
                      onClick={() => setSelectedIndex(null)}
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

                  {!canEdit && (
                    <div
                      className="mb-2"
                      style={{ fontSize: 12, color: 'var(--warning-soft-fg)' }}
                    >
                      Read-only at this width — {GIVES_WAY.detection}
                    </div>
                  )}

                  {confirmDelete ? (
                    /* Deleting is destructive and immediate on the server, so
                       it asks first. */
                    <div className="flex flex-wrap items-center gap-3">
                      <span style={{ fontSize: 13 }}>
                        Delete this <strong>{shown.Object}</strong>
                        {shown.Text?.trim() ? ` (${shown.Text.trim()})` : ''}? This cannot be undone.
                      </span>
                      <div className="flex-1" />
                      <Button
                        variant="ghost"
                        style={{ height: ctlH, borderRadius: 'var(--r-btn)' }}
                        onPress={() => setConfirmDelete(false)}
                      >
                        Cancel
                      </Button>
                      <Button
                        variant="ghost"
                        style={{
                          height: ctlH,
                          borderRadius: 'var(--r-btn)',
                          background: 'var(--danger)',
                          color: 'var(--white)',
                        }}
                        onPress={() => {
                          void deleteObject(sheet.id, shown)
                          setConfirmDelete(false)
                          setSelectedIndex(null)
                        }}
                      >
                        <Trash2 size={15} strokeWidth={1.8} />
                        Delete
                      </Button>
                    </div>
                  ) : editing ? (
                    <div className="flex flex-wrap items-end gap-2.5">
                      <LabelledField label="Class" flex={1.6}>
                        <input
                          list="detection-classes"
                          value={shown.Object}
                          onChange={(e) => setDraft({ ...shown, Object: e.target.value })}
                          style={FIELD}
                          aria-label="Class"
                        />
                        <datalist id="detection-classes">
                          {classes.map((c) => (
                            <option key={c.name} value={c.name} />
                          ))}
                        </datalist>
                      </LabelledField>
                      <LabelledField label="Text / tag" flex={1.4}>
                        <input
                          value={shown.Text}
                          onChange={(e) => setDraft({ ...shown, Text: e.target.value })}
                          style={FIELD}
                          aria-label="Text or tag"
                        />
                      </LabelledField>
                      {(['Left', 'Top', 'Width', 'Height'] as const).map((k) => (
                        <LabelledField
                          key={k}
                          width={78}
                          label={{ Left: 'x', Top: 'y', Width: 'w', Height: 'h' }[k]}
                        >
                          <input
                            type="number"
                            aria-label={k}
                            value={shown[k]}
                            onChange={(e) => setDraft({ ...shown, [k]: Number(e.target.value) })}
                            style={FIELD}
                          />
                        </LabelledField>
                      ))}
                      <Button
                        variant="ghost"
                        style={{ height: ctlH, borderRadius: 'var(--r-btn)' }}
                        onPress={() => {
                          // Abandon the drag edits by reverting to the server copy.
                          setDraft(selected ? { ...selected } : null)
                          setEditing(false)
                        }}
                      >
                        Cancel
                      </Button>
                      <Button
                        variant="primary"
                        style={{ height: ctlH, borderRadius: 'var(--r-btn)' }}
                        onPress={() => {
                          void updateObject(sheet.id, shown)
                          setEditing(false)
                        }}
                      >
                        <Check size={15} strokeWidth={2.2} />
                        OK
                      </Button>
                    </div>
                  ) : (
                    /* Read-only by default: the values as text, nothing to
                       mistype, and the two actions. */
                    <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
                      <ReadOnly label="Class" value={shown.Object} />
                      <ReadOnly label="Text / tag" value={shown.Text?.trim() || '—'} />
                      <ReadOnly
                        label="Bounding box"
                        mono
                        value={`${shown.Left}, ${shown.Top}, ${shown.Width} × ${shown.Height}`}
                      />
                      <div className="flex-1" />
                      <Button
                        variant="ghost"
                        isDisabled={!canEdit}
                        style={{
                          height: ctlH,
                          borderRadius: 'var(--r-btn)',
                          background: 'var(--danger-soft)',
                          color: 'var(--danger-soft-fg)',
                        }}
                        onPress={() => setConfirmDelete(true)}
                      >
                        <Trash2 size={15} strokeWidth={1.8} />
                        Delete
                      </Button>
                      <Button
                        variant="primary"
                        isDisabled={!canEdit}
                        style={{ height: ctlH, borderRadius: 'var(--r-btn)' }}
                        onPress={() => setEditing(true)}
                      >
                        <Pencil size={15} strokeWidth={1.8} />
                        Edit
                      </Button>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* Right panel */}
        <div
          className="flex shrink-0 flex-col gap-3.5 overflow-y-auto"
          style={{ width: width === 'tablet' ? 288 : 340 }}
        >
          <Card className="flex shrink-0 flex-col gap-3">
            <SectionHeader
              title="Detection settings"
              actions={sheet.detection ? <Tag tone="warning">edited</Tag> : undefined}
            />
            <div className="flex items-center gap-2">
              <div
                className="flex-1"
                style={{ fontSize: 12, fontWeight: 500, color: 'var(--muted)' }}
              >
                Confidence
              </div>
              <span className="mono" style={{ fontSize: 13 }}>
                {config.confTh.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0.2}
              max={0.95}
              step={0.01}
              value={config.confTh}
              onChange={(e) => setConfig({ confTh: Number(e.target.value) })}
              style={{ width: '100%', accentColor: 'var(--accent)' }}
            />
            <div className="flex items-stretch gap-2.5">
              <LabelledField label="Tile size">
                <select
                  aria-label="Tile size"
                  value={config.imageSize}
                  onChange={(e) => setConfig({ imageSize: Number(e.target.value) })}
                  style={FIELD}
                >
                  {[320, 512, 640, 800, 1024, 1280].map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </LabelledField>
              <LabelledField label="Overlap">
                <select
                  aria-label="Overlap"
                  value={config.overlapRatio}
                  onChange={(e) => setConfig({ overlapRatio: Number(e.target.value) })}
                  style={FIELD}
                >
                  {[0, 0.1, 0.2, 0.3, 0.4, 0.5].map((v) => (
                    <option key={v} value={v}>
                      {v.toFixed(2)}
                    </option>
                  ))}
                </select>
              </LabelledField>
            </div>
            <LabelledField label="Detection weights">
              <select
                aria-label="Detection weights"
                value={config.weightFile}
                onChange={(e) => setConfig({ weightFile: e.target.value })}
                style={FIELD}
              >
                <option value="">Server default</option>
                {weightFiles.map((w) => (
                  <option key={w} value={w}>
                    {w.replace(/^.*\//, '')}
                  </option>
                ))}
              </select>
            </LabelledField>
            <LabelledField label="Object labels on canvas">
              <select
                aria-label="Object labels on canvas"
                value={labelMode}
                onChange={(e) => setLabelMode(e.target.value as LabelMode)}
                style={FIELD}
              >
                {LABEL_MODES.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.label}
                  </option>
                ))}
              </select>
            </LabelledField>
            <Toggle
              label="Text recognition (OCR)"
              checked={config.textOCR}
              onChange={(textOCR) => setConfig({ textOCR })}
            />
            <Button
              variant="secondary"
              isDisabled={Boolean(sheet.progress)}
              style={{ alignSelf: 'flex-start', height: 32, borderRadius: 'var(--r-btn)' }}
              onPress={() => {
                setSelectedIndex(null)
                void runDetectionFor(sheet.id)
              }}
            >
              <RotateCw size={15} strokeWidth={1.6} />
              {sheet.detection ? 'Re-run detection' : 'Run detection'}
            </Button>
            {sheet.error && <div style={{ fontSize: 12, color: 'var(--danger)' }}>{sheet.error}</div>}
          </Card>

          {/* Fits the window, but never collapses below CLASSES_MIN_HEIGHT */}
          <Card
            className="flex min-h-0 flex-1 flex-col gap-3"
            style={{ minHeight: CLASSES_MIN_HEIGHT }}
          >
            <SectionHeader
              title="Classes"
              description={`${objects.length} object${objects.length === 1 ? '' : 's'}`}
              actions={
                hidden.size > 0 ? (
                  <Button
                    variant="ghost"
                    style={{ height: 26, borderRadius: 'var(--r-btn)', fontSize: 12 }}
                    onPress={() => setHidden(new Set())}
                  >
                    Show all
                  </Button>
                ) : undefined
              }
            />
            <div className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto">
              {classes.length === 0 && (
                <div style={{ fontSize: 13, color: 'var(--muted)' }}>
                  No objects yet — run detection.
                </div>
              )}
              {classes.map((c) => {
                const isHidden = hidden.has(c.name)
                const isOpen = expanded.has(c.name)
                const children = byClass.get(c.name) ?? []
                return (
                  <div key={c.name} className="shrink-0">
                    <div
                      className="flex items-center gap-1.5"
                      style={{ padding: '8px 6px 8px 2px', borderRadius: 10, opacity: isHidden ? 0.45 : 1 }}
                    >
                      <button
                        type="button"
                        aria-label={isOpen ? `Collapse ${c.name}` : `Expand ${c.name}`}
                        aria-expanded={isOpen}
                        onClick={() =>
                          setExpanded((prev) => {
                            const next = new Set(prev)
                            if (next.has(c.name)) next.delete(c.name)
                            else next.add(c.name)
                            return next
                          })
                        }
                        className="flex shrink-0 items-center justify-center"
                        style={{
                          width: 22,
                          height: 22,
                          border: 0,
                          background: 'transparent',
                          borderRadius: 6,
                          color: 'var(--muted)',
                          cursor: 'pointer',
                        }}
                      >
                        {isOpen ? (
                          <ChevronDown size={14} strokeWidth={2} />
                        ) : (
                          <ChevronRight size={14} strokeWidth={2} />
                        )}
                      </button>
                      <span
                        className="shrink-0"
                        style={{ width: 20, height: 20, borderRadius: 6, background: c.color }}
                      />
                      <span className="min-w-0 flex-1 truncate" style={{ fontSize: 13 }}>
                        {c.name}
                      </span>
                      <span className="mono shrink-0" style={{ fontSize: 13, color: 'var(--muted)' }}>
                        {c.count}
                      </span>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={!isHidden}
                        aria-label={`Show ${c.name}`}
                        onClick={() => toggleClass(c.name)}
                        className="relative shrink-0"
                        style={{
                          width: 38,
                          height: 22,
                          border: 0,
                          padding: 0,
                          cursor: 'pointer',
                          borderRadius: 999,
                          background: isHidden
                            ? 'color-mix(in oklab, var(--foreground) 22%, transparent)'
                            : 'var(--accent)',
                        }}
                      >
                        <span
                          style={{
                            position: 'absolute',
                            top: 2,
                            left: isHidden ? 2 : 18,
                            width: 18,
                            height: 18,
                            borderRadius: 999,
                            background: 'var(--white)',
                            boxShadow: '0 1px 2px rgba(0,0,0,.28)',
                            transition: 'left .15s',
                          }}
                        />
                      </button>
                    </div>

                    {isOpen && (
                      <div
                        className="flex flex-col"
                        style={{ marginLeft: 32, borderLeft: '1px solid var(--separator)' }}
                      >
                        {children.map((o) => {
                          const isSel = o.Index === selectedIndex
                          return (
                            <button
                              key={o.Index}
                              type="button"
                              onClick={() => setSelectedIndex(o.Index)}
                              className="flex items-center gap-2 text-left"
                              style={{
                                padding: '6px 8px',
                                marginLeft: 6,
                                border: 0,
                                borderRadius: 8,
                                cursor: 'pointer',
                                background: isSel ? 'var(--accent-soft)' : 'transparent',
                                color: isSel ? 'var(--accent-soft-fg)' : 'var(--foreground)',
                              }}
                            >
                              <span className="mono shrink-0" style={{ fontSize: 11, color: 'var(--muted)' }}>
                                #{o.Index}
                              </span>
                              <span className="min-w-0 flex-1 truncate" style={{ fontSize: 12 }}>
                                {o.Text?.trim() || o.Object}
                              </span>
                              <span className="mono shrink-0" style={{ fontSize: 11, color: 'var(--muted)' }}>
                                {o.Score.toFixed(2)}
                              </span>
                            </button>
                          )
                        })}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </Card>
        </div>
      </div>

      {/* Footer */}
      <div
        className="flex shrink-0 items-center gap-2.5"
        style={{ height: 56, padding: '0 24px', borderTop: '1px solid var(--separator)' }}
      >
        <span className="mono" style={{ fontSize: 13, color: 'var(--muted)' }}>
          {objects.length} objects
          {hidden.size > 0 ? ` · ${objects.length - visible.length} hidden` : ''}
        </span>
        <div className="flex-1" />
        {detectionSheets.length > 1 && (
          <select
            aria-label="Sheet"
            value={sheet.id}
            onChange={(e) => {
              selectSheet(e.target.value)
              setSelectedIndex(null)
            }}
            style={{ ...FIELD, height: 32, width: 220 }}
          >
            {detectionSheets.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        )}
        {(['excel', 'json', 'coco'] as const).map((k) => (
          <Button
            key={k}
            variant="ghost"
            isDisabled={objects.length === 0 || busy}
            style={{ height: 32, borderRadius: 'var(--r-btn)', fontSize: 13 }}
            onPress={() => void onExport(k)}
          >
            {k === 'excel' ? 'Excel' : k === 'json' ? 'JSON' : 'COCO'}
          </Button>
        ))}
      </div>
    </div>
  )
}
