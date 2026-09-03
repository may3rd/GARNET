import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Spinner } from '@heroui/react'
import { Maximize2, Minus, Plus, RotateCw, Scan, X } from 'lucide-react'
import { Card, SectionHeader, Tag, Toggle } from '@/components/ui/primitives'
import { classColor, summarizeClasses, normalizeClass } from '@/lib/detectionClasses'
import { exportCoco } from '@/lib/exportFormats'
import { exportResultsToExcel } from '@/lib/api'
import { useRunStore } from '@/stores/runStore'
import type { DetectedObject } from '@/types'

const MINIMAP = { w: 150, h: 100 }
const ZOOM_RANGE = { min: 0.02, max: 8 }
/** The class legend may shrink to fit the window, but not below this. */
const CLASSES_MIN_HEIGHT = 220

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

  const detectionSheets = sheets.filter((s) => s.task === 'detection')
  const sheet =
    sheets.find((s) => s.id === selectedSheetId && s.task === 'detection') ?? detectionSheets[0]

  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [draft, setDraft] = useState<DetectedObject | null>(null)
  const [zoom, setZoom] = useState<number | null>(null) // null = fit
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [busy, setBusy] = useState(false)
  const viewportRef = useRef<HTMLDivElement>(null)
  const [viewport, setViewport] = useState({ w: 0, h: 0 })

  const objects = sheet?.detection?.objects ?? []
  const imgW = sheet?.detection?.image_width ?? sheet?.size?.width ?? 0
  const imgH = sheet?.detection?.image_height ?? sheet?.size?.height ?? 0

  const classes = useMemo(() => summarizeClasses(objects), [objects])
  const selected = objects.find((o) => o.Index === selectedIndex) ?? null

  useEffect(() => {
    setDraft(selected ? { ...selected } : null)
  }, [selectedIndex, selected?.Index])

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const measure = () => setViewport({ w: el.clientWidth, h: el.clientHeight })
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    measure()
    return () => ro.disconnect()
  }, [sheet?.id])

  const fitScale =
    imgW && imgH && viewport.w && viewport.h ? Math.min(viewport.w / imgW, viewport.h / imgH) : 1
  const scale = zoom ?? fitScale
  const visible = objects.filter((o) => !hidden.has(normalizeClass(o.Object)))

  /** Zoom about a point in viewport coordinates, keeping it under the cursor. */
  const zoomAt = useCallback(
    (factor: number, cx: number, cy: number) => {
      setPan((prevPan) => {
        const current = zoom ?? fitScale
        const next = clamp(current * factor, ZOOM_RANGE.min, ZOOM_RANGE.max)
        const imgX = (cx - prevPan.x) / current
        const imgY = (cy - prevPan.y) / current
        setZoom(next)
        return { x: cx - imgX * next, y: cy - imgY * next }
      })
    },
    [zoom, fitScale]
  )

  // Wheel zoom needs a non-passive listener to be able to preventDefault, so
  // the page does not scroll while zooming the sheet.
  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const factor = Math.exp(-e.deltaY * 0.0015)
      zoomAt(factor, e.clientX - rect.left, e.clientY - rect.top)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [zoomAt])

  const startPan = (e: React.MouseEvent) => {
    const origin = { px: pan.x, py: pan.y, mx: e.clientX, my: e.clientY }
    const move = (ev: MouseEvent) =>
      setPan({
        x: origin.px + (ev.clientX - origin.mx),
        y: origin.py + (ev.clientY - origin.my),
      })
    const up = () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
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
      setPan({
        x: viewport.w / 2 - fx * imgW * scale,
        y: viewport.h / 2 - fy * imgH * scale,
      })
    },
    [imgW, imgH, scale, viewport.w, viewport.h]
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
      <div className="flex min-h-0 flex-1 gap-4" style={{ padding: '16px 24px' }}>
        {/* Canvas column — relative so the object sheet can slide up over it */}
        <div className="relative flex min-w-0 flex-1 flex-col">
          <div
            ref={viewportRef}
            className="relative min-h-0 flex-1 overflow-hidden"
            style={{
              background: 'var(--surface-tertiary)',
              borderRadius: 'var(--r-table)',
              cursor: 'grab',
            }}
            onMouseDown={startPan}
          >
            {sheet.progress && (
              <div className="absolute inset-0 z-10 flex items-center justify-center gap-2">
                <Spinner size="sm" />
                <span style={{ fontSize: 13, color: 'var(--muted)' }}>{sheet.progress.step}</span>
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
                    const color = classColor(o.Object)
                    return (
                      <g key={o.Index} onMouseDown={(e) => e.stopPropagation()}>
                        {/* Wide transparent stroke so thin boxes stay clickable. */}
                        <rect
                          x={o.Left}
                          y={o.Top}
                          width={o.Width}
                          height={o.Height}
                          fill="transparent"
                          stroke="transparent"
                          strokeWidth={12 / scale}
                          style={{ cursor: 'pointer' }}
                          onClick={() => setSelectedIndex(o.Index)}
                        />
                        <rect
                          x={o.Left}
                          y={o.Top}
                          width={o.Width}
                          height={o.Height}
                          fill={isSel ? `${color}22` : 'transparent'}
                          stroke={color}
                          strokeWidth={(isSel ? 3 : 1.6) / scale}
                          pointerEvents="none"
                        />
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
                    setPan({ x: 0, y: 0 })
                  },
                },
                {
                  icon: <Scan size={16} strokeWidth={1.6} />,
                  label: 'Actual size',
                  run: () => {
                    setZoom(1)
                    setPan({ x: 0, y: 0 })
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
                    width: 32,
                    height: 32,
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
                  cursor: 'crosshair',
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

          {/* Selected object — slides up from the bottom */}
          <div
            aria-hidden={!draft}
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              bottom: 0,
              transform: draft ? 'translateY(0)' : 'translateY(115%)',
              transition: 'transform .22s cubic-bezier(.32,.72,0,1)',
              pointerEvents: draft ? 'auto' : 'none',
            }}
          >
            <div
              style={{
                background: 'var(--overlay)',
                borderRadius: 'var(--r-card)',
                padding: 16,
                boxShadow: 'inset 0 0 0 1px var(--border), 0 -8px 32px rgba(0,0,0,.16)',
              }}
            >
              {draft && (
                <>
                  <div className="mb-3 flex items-center gap-2.5">
                    <span style={{ fontSize: 14, fontWeight: 500 }}>Selected object</span>
                    <span
                      className="shrink-0"
                      style={{
                        width: 12,
                        height: 12,
                        borderRadius: 4,
                        background: classColor(draft.Object),
                      }}
                    />
                    <span className="mono" style={{ fontSize: 12, color: 'var(--muted)' }}>
                      #{draft.Index}
                    </span>
                    <Tag tone={draft.Score >= 0.8 ? 'success' : 'warning'}>
                      {draft.Score.toFixed(2)}
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

                  <div className="flex flex-wrap items-end gap-2.5">
                    <LabelledField label="Class" flex={1.6}>
                      <input
                        list="detection-classes"
                        value={draft.Object}
                        onChange={(e) => setDraft({ ...draft, Object: e.target.value })}
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
                        value={draft.Text}
                        onChange={(e) => setDraft({ ...draft, Text: e.target.value })}
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
                          value={draft[k]}
                          onChange={(e) => setDraft({ ...draft, [k]: Number(e.target.value) })}
                          style={FIELD}
                        />
                      </LabelledField>
                    ))}
                    <Button
                      variant="ghost"
                      style={{
                        height: 36,
                        borderRadius: 'var(--r-btn)',
                        background: 'var(--danger-soft)',
                        color: 'var(--danger-soft-fg)',
                      }}
                      onPress={() => {
                        void deleteObject(sheet.id, draft)
                        setSelectedIndex(null)
                      }}
                    >
                      Delete
                    </Button>
                    <Button
                      variant="primary"
                      style={{ height: 36, borderRadius: 'var(--r-btn)' }}
                      onPress={() => void updateObject(sheet.id, draft)}
                    >
                      Apply
                    </Button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Right panel */}
        <div
          className="flex shrink-0 flex-col gap-3.5 overflow-y-auto"
          style={{ width: 340 }}
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
                return (
                  <div
                    key={c.name}
                    className="flex shrink-0 items-center gap-2.5"
                    style={{ padding: '8px 12px', borderRadius: 10, opacity: isHidden ? 0.45 : 1 }}
                  >
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
