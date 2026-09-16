import { useCallback, useEffect, useRef, useState } from 'react'
import { Button, Spinner } from '@heroui/react'
import { Check, Minus, Pencil, Play, Plus, Trash2, Upload, X } from 'lucide-react'
import { AiImportDialog } from '@/components/AiImportDialog'
import { Card, ResizableSidebar, SectionHeader, Tag } from '@/components/ui/primitives'
import { useResizableSidebar } from '@/hooks/useResizableSidebar'
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

/**
 * One row from stage4_objects.json / stage4_instrument_tags.json /
 * stage4_line_numbers.json. The three artifacts share id+bbox but otherwise
 * diverge (class_name vs. text) — kept loose with a passthrough index so a
 * save round-trips every field the backend wrote, not just the ones this
 * editor knows about.
 */
type Stage4Item = {
  id: string
  bbox: Bbox
  class_name?: string
  text?: string
  confidence?: number
  fused_confidence?: number
  [key: string]: unknown
}

type BucketKey = 'equipment' | 'instrument' | 'line_number'

type RawArtifact = Record<string, unknown>

const BUCKETS: Record<BucketKey, { artifact: string; itemsKey: string; label: string; hasClass: boolean }> = {
  equipment: { artifact: 'stage4_objects.json', itemsKey: 'objects', label: 'Equipment', hasClass: true },
  instrument: { artifact: 'stage4_instrument_tags.json', itemsKey: 'instrument_tags', label: 'Instrument', hasClass: false },
  line_number: { artifact: 'stage4_line_numbers.json', itemsKey: 'line_numbers', label: 'Line number', hasClass: false },
}
const BUCKET_ORDER: BucketKey[] = ['equipment', 'instrument', 'line_number']

/** Fixed pseudo-class so instrument/line-number boxes get a stable, on-brand colour from classColor's NAMED table. */
const BUCKET_COLOR_CLASS: Record<BucketKey, string> = {
  equipment: '',
  instrument: 'instrument tag',
  line_number: 'line number',
}

/**
 * The Equipment tab is scoped to major process equipment — not every
 * stage4_objects.json detection (valves, instrument-tag markers, arrows,
 * nodes, etc. stay out of view here; instrument tags get their own tab and
 * the rest aren't reviewed through Gate 1 at all). Mirrors frontend_old's
 * stage3_equipment class vocabulary, since the detector doesn't have its own
 * "equipment" class — these are hand-classified from the same fallback list.
 */
const EQUIPMENT_CLASSES = new Set([
  'blower',
  'column',
  'compressor',
  'fan',
  'heat exchanger',
  'mixer',
  'pump',
  'tank',
  'vessel',
])
const isEquipmentClass = (className: string | undefined) => EQUIPMENT_CLASSES.has(normalizeClass(className ?? ''))

/**
 * A box a reviewer drew and classified stays visible on the Equipment tab
 * even if they typed a class outside the preset vocabulary — otherwise
 * saving/editing a manually-added box makes it silently vanish from both the
 * canvas and the sidebar (it's still written to stage4_objects.json, so
 * "disappeared" reads as "wasn't saved" even though it was).
 */
const keepOnEquipmentTab = (o: Stage4Item) => isEquipmentClass(o.class_name) || o.source_model === 'hitl'

const itemsOf = (raw: RawArtifact | null, bucket: BucketKey): Stage4Item[] =>
  (raw?.[BUCKETS[bucket].itemsKey] as Stage4Item[] | undefined) ?? []

const itemConfidence = (item: Stage4Item): number => item.confidence ?? item.fused_confidence ?? 1

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
 * Gate 1 — the real stage 4 review, in its three parts: equipment/object
 * boxes, instrument tags, and line numbers. All three are stage4_* fusion
 * outputs that park at the same `stop_after` boundary (see gates.ts), so one
 * gate reviews all three rather than only the object boxes.
 */
export function Gate1Objects({ sheet, onBack }: { sheet: Sheet; onBack: () => void }) {
  const resumeGate = useRunStore((s) => s.resumeGate)
  const setScreen = useRunStore((s) => s.setScreen)
  const sidebar = useResizableSidebar(240)

  const [bucket, setBucket] = useState<BucketKey>('equipment')
  const [raw, setRaw] = useState<Record<BucketKey, RawArtifact | null>>({
    equipment: null,
    instrument: null,
    line_number: null,
  })
  const [dirty, setDirty] = useState<Record<BucketKey, boolean>>({
    equipment: false,
    instrument: false,
    line_number: false,
  })
  const [loadError, setLoadError] = useState<string | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [confirming, setConfirming] = useState(false)

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [draft, setDraft] = useState<Stage4Item | null>(null)
  const [editing, setEditing] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [drawing, setDrawing] = useState(false)
  const [drawRect, setDrawRect] = useState<Box | null>(null)
  const [importOpen, setImportOpen] = useState(false)
  const [reloadToken, setReloadToken] = useState(0)
  /** Crosshair guide while placing a new box — screen-space so it doesn't scale with zoom; image-space kept alongside for the coordinate readout. */
  const [cursorGuide, setCursorGuide] = useState<{ imageX: number; imageY: number; screenX: number; screenY: number } | null>(null)

  const [zoom, setZoom] = useState<number | null>(null)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const viewportRef = useRef<HTMLDivElement>(null)
  const [viewport, setViewport] = useState({ w: 0, h: 0 })

  // Floating dialog position, as an offset from its default top-right corner.
  const [dialogOffset, setDialogOffset] = useState({ x: 0, y: 0 })
  useEffect(() => {
    setDialogOffset({ x: 0, y: 0 })
  }, [selectedId])

  const jobId = sheet.jobId
  const imgW = sheet.size?.width ?? 0
  const imgH = sheet.size?.height ?? 0
  const bucketConfig = BUCKETS[bucket]
  const loaded = raw.equipment !== null && raw.instrument !== null && raw.line_number !== null

  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    setLoadError(null)
    Promise.all(
      BUCKET_ORDER.map((key) => getPipelineArtifactJson<RawArtifact>(jobId, BUCKETS[key].artifact))
    )
      .then(([equipment, instrument, lineNumber]) => {
        if (!cancelled) setRaw({ equipment, instrument, line_number: lineNumber })
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : 'Could not load stage 4 artifacts')
      })
    return () => {
      cancelled = true
    }
    // reloadToken lets an import pull the rewritten artifacts back in.
  }, [jobId, reloadToken])

  const items = itemsOf(raw[bucket], bucket)
  const setItems = (updater: (current: Stage4Item[]) => Stage4Item[]) => {
    setRaw((current) => ({
      ...current,
      [bucket]: { ...current[bucket], [bucketConfig.itemsKey]: updater(itemsOf(current[bucket], bucket)) },
    }))
    setDirty((current) => ({ ...current, [bucket]: true }))
  }

  const selected = items.find((o) => o.id === selectedId) ?? null
  // A plain click always lands on the read-only view; a freshly-drawn box
  // is the one exception, since it still needs classifying.
  const openInEditMode = useRef(false)
  useEffect(() => {
    setDraft(selected ? { ...selected } : null)
    setEditing(openInEditMode.current)
    openInEditMode.current = false
    setConfirmDelete(false)
  }, [selectedId, selected?.id])
  const [lastDraft, setLastDraft] = useState<Stage4Item | null>(null)
  useEffect(() => {
    if (draft) setLastDraft(draft)
  }, [draft])
  const shown = draft ?? lastDraft

  const switchBucket = (next: BucketKey) => {
    setBucket(next)
    setSelectedId(null)
    setDrawing(false)
    setDrawRect(null)
  }

  // Static equipment vocabulary, not the classes actually present — a
  // reviewer needs to classify a vessel the detector never found, so the
  // datalist can't be derived from what's already in the file.
  const classes = bucketConfig.hasClass ? [...EQUIPMENT_CLASSES].sort() : []

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const measure = () => setViewport({ w: el.clientWidth, h: el.clientHeight })
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    measure()
    return () => ro.disconnect()
    // The canvas (and viewportRef) doesn't exist until the artifacts finish
    // loading — it's behind the `if (!loaded) return <Spinner>` below — so
    // this must re-run once that flips, or el stays null forever and
    // viewport.w/h stay 0, which anchors +/- zoom at the top-left corner
    // instead of the canvas centre.
  }, [loaded])

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

  /** Pan so a box is centred in the view, without changing zoom — same as the Detection preview's sidebar-select behaviour. */
  const focusOn = useCallback(
    (o: Stage4Item) => {
      const box = boxFromBbox(o.bbox)
      const cx = box.Left + box.Width / 2
      const cy = box.Top + box.Height / 2
      const current = scaleRef.current
      const next = settle({ x: viewport.w / 2 - cx * current, y: viewport.h / 2 - cy * current }, current)
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

  const startDialogDrag = (e: React.MouseEvent) => {
    e.stopPropagation()
    const origin = { x: dialogOffset.x, y: dialogOffset.y, mx: e.clientX, my: e.clientY }
    const move = (ev: MouseEvent) => {
      setDialogOffset({ x: origin.x + (ev.clientX - origin.mx), y: origin.y + (ev.clientY - origin.my) })
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
        const created: Stage4Item = bucketConfig.hasClass
          ? { id, class_name: classes[0] ?? 'object', confidence: 1, bbox: bboxFromBox(box), source_model: 'hitl' }
          : {
              id,
              text: '',
              normalized_text: '',
              source_object_id: id,
              semantic_class: bucket === 'instrument' ? 'instrument_semantic' : 'line_number',
              fused_confidence: 1,
              ocr_confirmed: false,
              source: 'hitl',
              bbox: bboxFromBox(box),
            }
        setItems((prev) => [...prev, created])
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

  useEffect(() => {
    if (!drawing) setCursorGuide(null)
  }, [drawing])

  /** Tracks the cursor while placing a box so the crosshair guide can follow it — no-op outside drawing mode. */
  const trackCursorGuide = (e: React.MouseEvent) => {
    if (!drawing) return
    const el = viewportRef.current
    if (!el || !imgW || !imgH) return
    const rect = el.getBoundingClientRect()
    const imageX = (e.clientX - rect.left - pan.x) / scale
    const imageY = (e.clientY - rect.top - pan.y) / scale
    if (imageX < 0 || imageY < 0 || imageX > imgW || imageY > imgH) {
      setCursorGuide(null)
      return
    }
    setCursorGuide({ imageX, imageY, screenX: e.clientX - rect.left, screenY: e.clientY - rect.top })
  }

  const applyEdit = () => {
    if (!draft) return
    setItems((prev) => prev.map((o) => (o.id === draft.id ? draft : o)))
    setEditing(false)
  }

  const doDelete = () => {
    if (!shown) return
    setItems((prev) => prev.filter((o) => o.id !== shown.id))
    setConfirmDelete(false)
    setSelectedId(null)
  }

  const saveBucket = async (key: BucketKey): Promise<boolean> => {
    if (!jobId || !raw[key]) return false
    try {
      await putPipelineArtifact(jobId, BUCKETS[key].artifact, raw[key] as Record<string, unknown>)
      setDirty((current) => ({ ...current, [key]: false }))
      return true
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : `Could not save ${BUCKETS[key].label.toLowerCase()}`)
      return false
    }
  }

  const save = async (): Promise<boolean> => {
    setSaving(true)
    setSaveError(null)
    try {
      return await saveBucket(bucket)
    } finally {
      setSaving(false)
    }
  }

  const saveAndConfirm = async () => {
    setConfirming(true)
    setSaveError(null)
    try {
      for (const key of BUCKET_ORDER) {
        if (dirty[key] && !(await saveBucket(key))) return
      }
      setScreen('run')
      await resumeGate(sheet.id, 1)
    } finally {
      setConfirming(false)
    }
  }

  if (!jobId) return null

  if (loadError) {
    return (
      <div className="p-6">
        <Card padding={20}>
          <SectionHeader title="Could not load stage 4 artifacts" description={loadError} />
        </Card>
      </div>
    )
  }

  if (!loaded) {
    return (
      <div className="flex h-full items-center justify-center gap-2">
        <Spinner size="sm" />
        <span style={{ fontSize: 13, color: 'var(--muted)' }}>Loading stage 4 artifacts…</span>
      </div>
    )
  }

  const visible = bucket === 'equipment' ? items.filter(keepOnEquipmentTab) : items
  // Equipment groups by class_name (several groups); instrument/line_number
  // are already single-class artifacts, so everything lands in one group —
  // same list UI either way, just with or without multiple headers.
  const groups = Object.entries(
    visible.reduce<Record<string, Stage4Item[]>>((acc, o) => {
      const key = bucketConfig.hasClass ? normalizeClass(o.class_name ?? '') : BUCKET_COLOR_CLASS[bucket]
      ;(acc[key] ??= []).push(o)
      return acc
    }, {})
  ).sort(([a], [b]) => a.localeCompare(b))
  const countLabel =
    bucket === 'equipment' ? `${visible.length} of ${items.length} objects` : `${items.length} ${bucketConfig.label.toLowerCase()}`
  const anyDirty = BUCKET_ORDER.some((key) => dirty[key])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 flex flex-wrap gap-2" style={{ padding: '16px 24px 0' }}>
        {BUCKET_ORDER.map((key) => (
          <button
            key={key}
            type="button"
            onClick={() => switchBucket(key)}
            className="inline-flex items-center gap-1.5"
            style={{
              height: 30,
              padding: '0 12px',
              borderRadius: 999,
              border: 0,
              fontSize: 12.5,
              fontWeight: 500,
              cursor: 'pointer',
              background: bucket === key ? 'var(--accent-soft)' : 'var(--surface-secondary)',
              color: bucket === key ? 'var(--accent-soft-fg)' : 'var(--muted)',
            }}
          >
            {BUCKETS[key].label}
            <span className="mono" style={{ fontSize: 11, opacity: 0.75 }}>
              {key === 'equipment'
                ? itemsOf(raw[key], key).filter(keepOnEquipmentTab).length
                : itemsOf(raw[key], key).length}
            </span>
            {dirty[key] && (
              <span
                aria-label="Unsaved changes"
                style={{ width: 6, height: 6, borderRadius: 999, background: 'var(--warning)' }}
              />
            )}
          </button>
        ))}
      </div>

      <div className="flex min-h-0 flex-1 gap-4" style={{ padding: '16px 24px' }}>
        <div className="relative flex min-w-0 flex-[1.3] flex-col overflow-hidden">
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
            onMouseMove={trackCursorGuide}
            onMouseLeave={() => setCursorGuide(null)}
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
                    // Greyed out whenever something else is selected, regardless of
                    // edit mode — but only actively blocked from clicks while
                    // editing, so a plain selection still lets the reviewer click
                    // straight to a different box.
                    const muted = Boolean(selectedId) && !isSel
                    const blockInteraction = editing && !isSel
                    const color = muted
                      ? 'var(--muted)'
                      : classColor(bucketConfig.hasClass ? o.class_name ?? '' : BUCKET_COLOR_CLASS[bucket])
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
                            cursor: blockInteraction ? 'default' : isSel && editing ? 'move' : 'pointer',
                            pointerEvents: blockInteraction ? 'none' : undefined,
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
                          fill={isSel ? `${muted ? 'transparent' : color}22` : 'transparent'}
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

            {/* Crosshair guide while placing a box — screen-space siblings of the scaled/panned layer above, so the lines stay 1px regardless of zoom. */}
            {drawing && cursorGuide && (
              <div className="pointer-events-none absolute inset-0" style={{ zIndex: 4 }}>
                <div
                  className="absolute top-0 h-full"
                  style={{
                    left: cursorGuide.screenX,
                    width: 1,
                    background: 'var(--accent)',
                    opacity: 0.7,
                    boxShadow: '0 0 0 1px rgba(255,255,255,.35)',
                  }}
                />
                <div
                  className="absolute left-0 w-full"
                  style={{
                    top: cursorGuide.screenY,
                    height: 1,
                    background: 'var(--accent)',
                    opacity: 0.7,
                    boxShadow: '0 0 0 1px rgba(255,255,255,.35)',
                  }}
                />
                <div
                  className="mono"
                  style={{
                    position: 'absolute',
                    left: Math.min(cursorGuide.screenX + 8, Math.max(0, viewport.w - 92)),
                    top: Math.min(cursorGuide.screenY + 8, Math.max(0, viewport.h - 22)),
                    padding: '2px 6px',
                    borderRadius: 'var(--r-chip)',
                    background: 'var(--overlay)',
                    color: 'var(--foreground)',
                    fontSize: 10.5,
                    fontWeight: 600,
                    boxShadow: 'inset 0 0 0 1px var(--border), 0 2px 8px rgba(0,0,0,.2)',
                  }}
                >
                  {Math.round(cursorGuide.imageX)}, {Math.round(cursorGuide.imageY)}
                </div>
              </div>
            )}

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
                style={{
                  width: 26,
                  height: 26,
                  border: 0,
                  background: 'transparent',
                  borderRadius: 'var(--r-btn)',
                  color: 'var(--foreground)',
                  cursor: 'pointer',
                }}
              >
                <Plus size={15} strokeWidth={1.8} />
              </button>
              <button
                type="button"
                aria-label="Zoom out"
                title="Zoom out"
                onClick={() => zoomAt(1 / 1.25, viewport.w / 2, viewport.h / 2)}
                className="flex items-center justify-center"
                style={{
                  width: 26,
                  height: 26,
                  border: 0,
                  background: 'transparent',
                  borderRadius: 'var(--r-btn)',
                  color: 'var(--foreground)',
                  cursor: 'pointer',
                }}
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

          {/* Selected item — floating dialog, matching the Detection pattern.
              Position (drag offset) and the open/close animation live on
              separate layers, so dragging isn't fighting the CSS transition. */}
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
              visibility: shown ? 'visible' : 'hidden',
              pointerEvents: draft ? 'auto' : 'none',
              transform: `translate(${dialogOffset.x}px, ${dialogOffset.y}px)`,
            }}
          >
            <div
              style={{
                background: 'var(--overlay)',
                borderRadius: 'var(--r-card)',
                padding: 16,
                boxShadow:
                  'inset 0 0 0 1px var(--border), 0 1px 2px rgba(0,0,0,.24), 0 14px 32px rgba(0,0,0,.38)',
                opacity: draft ? 1 : 0,
                transform: draft ? 'scale(1)' : 'scale(0.96)',
                transition: 'opacity .16s ease, transform .16s ease',
              }}
            >
              {shown && (
                <>
                  <div
                    className="mb-3 flex items-center gap-2.5"
                    style={{ cursor: 'move' }}
                    onMouseDown={startDialogDrag}
                  >
                    <span style={{ fontSize: 14, fontWeight: 500 }}>Selected {bucketConfig.label.toLowerCase()}</span>
                    <span
                      className="shrink-0"
                      style={{
                        width: 12,
                        height: 12,
                        borderRadius: 4,
                        background: classColor(bucketConfig.hasClass ? shown.class_name ?? '' : BUCKET_COLOR_CLASS[bucket]),
                      }}
                    />
                    <Tag tone={itemConfidence(shown) >= 0.8 ? 'success' : 'warning'}>
                      {itemConfidence(shown).toFixed(2)}
                    </Tag>
                    <div className="flex-1" />
                    <button
                      type="button"
                      aria-label="Close selected item"
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
                        Delete this <strong>{bucketConfig.hasClass ? shown.text || shown.class_name : shown.text || '(blank)'}</strong>? This is
                        undone by re-opening the gate without saving.
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
                      {bucketConfig.hasClass && (
                        <Field label="Class">
                          <input
                            list="gate1-classes"
                            value={shown.class_name ?? ''}
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
                      )}
                      <Field label={bucketConfig.hasClass ? 'Label (name or tag number)' : 'Text'}>
                        <input
                          value={shown.text ?? ''}
                          onChange={(e) => setDraft({ ...shown, text: e.target.value, normalized_text: e.target.value })}
                          style={FIELD}
                          aria-label={bucketConfig.hasClass ? 'Label' : 'Text'}
                        />
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
                        {bucketConfig.hasClass && (
                          <div className="flex flex-col gap-0.5">
                            <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--muted)' }}>Class</span>
                            <span style={{ fontSize: 14, fontWeight: 500 }}>{shown.class_name}</span>
                          </div>
                        )}
                        <div className="flex flex-col gap-0.5">
                          <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--muted)' }}>
                            {bucketConfig.hasClass ? 'Label' : 'Text'}
                          </span>
                          <span className="mono" style={{ fontSize: 14, fontWeight: 500 }}>
                            {shown.text || '(blank)'}
                          </span>
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

        <ResizableSidebar
          width={sidebar.width}
          collapsed={sidebar.collapsed}
          onToggleCollapsed={sidebar.toggleCollapsed}
          onStartResize={sidebar.startResize}
        >
          <Card className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-hidden" padding={16}>
            <SectionHeader
              title={bucketConfig.label}
              description={countLabel}
              actions={bucketConfig.hasClass ? <Tag tone="neutral">{groups.length}</Tag> : undefined}
            />
            <Button
              variant={drawing ? 'secondary' : 'primary'}
              isDisabled={editing}
              aria-pressed={drawing}
              style={{ height: 30, borderRadius: 'var(--r-btn)', fontSize: 12.5 }}
              onPress={() => setDrawing((d) => !d)}
            >
              {drawing ? <X size={14} strokeWidth={2} /> : <Plus size={14} strokeWidth={2.2} />}
              {drawing ? 'Cancel' : 'Box'}
            </Button>
            <Button
              variant="secondary"
              isDisabled={editing || drawing}
              style={{ height: 30, borderRadius: 'var(--r-btn)', fontSize: 12.5 }}
              onPress={() => setImportOpen(true)}
            >
              <Upload size={13} strokeWidth={2} />
              Import JSON
            </Button>
            <div className="min-h-0 flex-1 overflow-y-auto">
              {visible.length === 0 && (
                <div style={{ fontSize: 13, color: 'var(--muted)' }}>
                  No {bucketConfig.label.toLowerCase()} yet — draw a box on the canvas.
                </div>
              )}
              {groups.map(([className, groupItems]) => (
                <div key={className} className="shrink-0" style={{ marginBottom: 10 }}>
                  {bucketConfig.hasClass && (
                    <div className="flex items-center gap-1.5" style={{ padding: '4px 2px' }}>
                      <span className="shrink-0" style={{ width: 10, height: 10, borderRadius: 3, background: classColor(className) }} />
                      <span className="min-w-0 flex-1 truncate" style={{ fontSize: 12.5, fontWeight: 500 }}>
                        {className}
                      </span>
                      <span className="mono shrink-0" style={{ fontSize: 11, color: 'var(--muted)' }}>
                        {groupItems.length}
                      </span>
                    </div>
                  )}
                  {groupItems.map((o) => {
                    const isSel = o.id === selectedId
                    return (
                      <button
                        key={o.id}
                        type="button"
                        onClick={() => {
                          setSelectedId(o.id)
                          focusOn(o)
                        }}
                        className="flex w-full items-center gap-2 text-left"
                        style={{
                          padding: '6px 8px',
                          marginLeft: bucketConfig.hasClass ? 16 : 0,
                          border: 0,
                          borderRadius: 'var(--r-chip)',
                          cursor: 'pointer',
                          background: isSel ? 'var(--accent-soft)' : 'transparent',
                          color: isSel ? 'var(--accent-soft-fg)' : 'var(--foreground)',
                        }}
                      >
                        <span className="mono min-w-0 flex-1 truncate" style={{ fontSize: 12 }}>
                          {o.text || (o.normalized_text as string | undefined) || o.id}
                        </span>
                        <span className="mono shrink-0" style={{ fontSize: 11, color: 'var(--muted)' }}>
                          {itemConfidence(o).toFixed(2)}
                        </span>
                      </button>
                    )
                  })}
                </div>
              ))}
            </div>
          </Card>
        </ResizableSidebar>
      </div>

      {/* Footer */}
      <div
        className="flex shrink-0 items-center gap-2.5"
        style={{ height: 56, padding: '0 24px', borderTop: '1px solid var(--separator)' }}
      >
        <span className="mono" style={{ fontSize: 13, color: 'var(--muted)' }}>
          {countLabel}
          {dirty[bucket] ? ' · unsaved' : ''}
        </span>
        <Button
          variant="ghost"
          isIconOnly
          aria-pressed={drawing}
          aria-label={drawing ? 'Cancel drawing (Esc)' : `Add a missed ${bucketConfig.label.toLowerCase()}`}
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
          isDisabled={!dirty[bucket] || saving}
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
          {anyDirty ? 'Save & continue' : 'Continue'}
        </Button>
      </div>

      {importOpen && jobId && (
        <AiImportDialog
          jobId={jobId}
          onClose={() => setImportOpen(false)}
          onImported={() => {
            setImportOpen(false)
            setSelectedId(null)
            // The import rewrote the artifacts server-side; pull them back in
            // rather than keeping the pre-import copy in local state.
            setDirty({ equipment: false, instrument: false, line_number: false })
            setReloadToken((token) => token + 1)
          }}
        />
      )}
    </div>
  )
}
