import { useEffect, useMemo, useRef, useState } from 'react'
import { Button, Spinner } from '@heroui/react'
import { Maximize2, Minus, Plus, RotateCw, Scan } from 'lucide-react'
import { Card, SectionHeader, Tag, Toggle } from '@/components/ui/primitives'
import { classColor, summarizeClasses, normalizeClass } from '@/lib/detectionClasses'
import { exportCoco } from '@/lib/exportFormats'
import { exportResultsToExcel } from '@/lib/api'
import { useRunStore } from '@/stores/runStore'
import type { DetectedObject } from '@/types'

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

function LabelledField({
  label,
  children,
  flex,
}: {
  label: string
  children: React.ReactNode
  flex?: number
}) {
  return (
    <div className="flex flex-col gap-1.5" style={{ flex: flex ?? 1, minWidth: 0 }}>
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

  // Default to the first detection sheet so the screen is never empty by accident.
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
    const ro = new ResizeObserver(() => {
      setViewport({ w: el.clientWidth, h: el.clientHeight })
    })
    ro.observe(el)
    setViewport({ w: el.clientWidth, h: el.clientHeight })
    return () => ro.disconnect()
  }, [sheet?.id])

  const fitScale =
    imgW && imgH && viewport.w && viewport.h
      ? Math.min(viewport.w / imgW, viewport.h / imgH)
      : 1
  const scale = zoom ?? fitScale
  const visible = objects.filter((o) => !hidden.has(normalizeClass(o.Object)))

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

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex min-h-0 flex-1 gap-4" style={{ padding: '16px 24px' }}>
        {/* Canvas */}
        <div
          ref={viewportRef}
          className="relative min-w-0 flex-1 overflow-hidden"
          style={{ background: 'var(--surface-tertiary)', borderRadius: 'var(--r-table)' }}
          onMouseDown={(e) => {
            const origin = { px: pan.x, py: pan.y, mx: e.clientX, my: e.clientY }
            const handleMove = (ev: MouseEvent) => {
              setPan({
                x: origin.px + (ev.clientX - origin.mx),
                y: origin.py + (ev.clientY - origin.my),
              })
            }
            const handleUp = () => {
              window.removeEventListener('mousemove', handleMove)
              window.removeEventListener('mouseup', handleUp)
            }
            window.addEventListener('mousemove', handleMove)
            window.addEventListener('mouseup', handleUp)
          }}
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
              width={imgW || undefined}
              height={imgH || undefined}
              style={{ display: 'block', background: '#ffffff' }}
              draggable={false}
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
          >
            {[
              { icon: <Plus size={16} strokeWidth={1.6} />, label: 'Zoom in', run: () => setZoom(scale * 1.25) },
              { icon: <Minus size={16} strokeWidth={1.6} />, label: 'Zoom out', run: () => setZoom(scale / 1.25) },
              {
                icon: <Maximize2 size={16} strokeWidth={1.6} />,
                label: 'Fit',
                run: () => {
                  setZoom(null)
                  setPan({ x: 0, y: 0 })
                },
              },
              { icon: <Scan size={16} strokeWidth={1.6} />, label: 'Actual size', run: () => setZoom(1) },
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
            <span style={{ width: 1, height: 20, background: 'var(--separator)', margin: '0 4px' }} />
            <span className="mono" style={{ padding: '0 8px', fontSize: 12, color: 'var(--muted)' }}>
              {Math.round(scale * 100)}%
            </span>
          </div>

          {/* Minimap */}
          {imgW > 0 && viewport.w > 0 && (
            <div
              className="absolute overflow-hidden"
              style={{
                right: 12,
                bottom: 12,
                width: 150,
                height: 100,
                background: '#ffffff',
                borderRadius: 10,
                boxShadow: 'inset 0 0 0 1px var(--border)',
              }}
            >
              <img
                src={sheet.previewUrl}
                alt=""
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                draggable={false}
              />
              <div
                style={{
                  position: 'absolute',
                  border: '2px solid var(--accent)',
                  borderRadius: 2,
                  left: `${Math.max(0, Math.min(100, (-pan.x / (imgW * scale)) * 100))}%`,
                  top: `${Math.max(0, Math.min(100, (-pan.y / (imgH * scale)) * 100))}%`,
                  width: `${Math.max(4, Math.min(100, (viewport.w / (imgW * scale)) * 100))}%`,
                  height: `${Math.max(4, Math.min(100, (viewport.h / (imgH * scale)) * 100))}%`,
                }}
              />
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="flex shrink-0 flex-col gap-3.5" style={{ width: 340 }}>
          <Card className="flex flex-col gap-3">
            <SectionHeader
              title="Detection settings"
              actions={sheet.detection ? <Tag tone="warning">edited</Tag> : undefined}
            />
            <div className="flex items-center gap-2">
              <div className="flex-1" style={{ fontSize: 12, fontWeight: 500, color: 'var(--muted)' }}>
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
            {sheet.error && (
              <div style={{ fontSize: 12, color: 'var(--danger)' }}>{sheet.error}</div>
            )}
          </Card>

          <Card className="flex min-h-0 flex-1 flex-col gap-3">
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
                    className="flex items-center gap-2.5"
                    style={{
                      padding: '8px 12px',
                      borderRadius: 10,
                      opacity: isHidden ? 0.45 : 1,
                    }}
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

          <Card className="flex flex-col gap-2.5">
            <SectionHeader
              title="Selected object"
              actions={
                draft ? (
                  <Tag tone={draft.Score >= 0.8 ? 'success' : 'warning'}>
                    {draft.Score.toFixed(2)}
                  </Tag>
                ) : undefined
              }
            />
            {!draft ? (
              <div style={{ fontSize: 13, color: 'var(--muted)' }}>
                Click a box on the sheet to edit its class, tag or bounds.
              </div>
            ) : (
              <>
                <div className="flex items-stretch gap-2.5">
                  <LabelledField label="Class" flex={1.4}>
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
                  <LabelledField label="Text / tag">
                    <input
                      value={draft.Text}
                      onChange={(e) => setDraft({ ...draft, Text: e.target.value })}
                      style={FIELD}
                      aria-label="Text or tag"
                    />
                  </LabelledField>
                </div>

                <div className="flex flex-col gap-1.5">
                  <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--muted)' }}>
                    Bounding box &nbsp;x, y, w, h
                  </div>
                  <div className="flex items-stretch gap-2">
                    {(['Left', 'Top', 'Width', 'Height'] as const).map((k) => (
                      <input
                        key={k}
                        type="number"
                        aria-label={k}
                        value={draft[k]}
                        onChange={(e) => setDraft({ ...draft, [k]: Number(e.target.value) })}
                        style={FIELD}
                      />
                    ))}
                  </div>
                </div>

                <div className="flex items-stretch gap-2">
                  <Button
                    variant="ghost"
                    style={{
                      flex: 1,
                      height: 32,
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
                    style={{ flex: 1, height: 32, borderRadius: 'var(--r-btn)' }}
                    onPress={() => void updateObject(sheet.id, draft)}
                  >
                    Apply
                  </Button>
                </div>
              </>
            )}
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
          {detectionSheets.length > 1 ? ` · sheet ${detectionSheets.indexOf(sheet) + 1} of ${detectionSheets.length}` : ''}
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
