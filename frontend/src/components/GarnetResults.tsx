import { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '@heroui/react'
import { ArrowLeft, Maximize, Minus, Plus } from 'lucide-react'
import { Card, PageHeader, Tag } from '@/components/ui/primitives'
import type { GarnetJob } from '@/lib/api'
import { useRunStore } from '@/stores/runStore'

/**
 * Read-only view of one garnet/ job's outputs, straight from its artifact files.
 * Values are shown as written — no rounding, no text clean-up, no inferred tags.
 */

type BBox = { x_min: number; y_min: number; x_max: number; y_max: number }
type Obj = { id: string; class_name: string; confidence?: number; bbox: BBox; text?: string }
type Assoc = Obj & {
  trace_id: string | null
  distance_px: number
  reason?: string
  max_distance_px?: number
  status: 'accepted' | 'rejected'
}
type Port = { mark: string; size?: string; line_number?: string | null; side?: string; point_px: [number, number] }
type Equipment = {
  Index: number
  Tag?: string
  Equipment_type?: string
  Service?: string
  Size?: string
  Evidence?: string
  Bounding_box_px: BBox
  Ports?: Port[]
}
type Data = {
  imageUrl: string
  width: number
  height: number
  objects: Obj[]
  equipment: Equipment[]
  lineNumbers: Assoc[]
  instruments: Assoc[]
  tracesWithoutLineNumber: string[]
}
type Tab = 'objects' | 'lineNumbers' | 'equipment' | 'instruments'
const TABS: { key: Tab; label: string }[] = [
  { key: 'objects', label: 'Objects' },
  { key: 'lineNumbers', label: 'Line numbers' },
  { key: 'equipment', label: 'Equipment' },
  { key: 'instruments', label: 'Instrument tags' },
]

const url = (job: GarnetJob, name: string) => job.artifacts.find((a) => a.name === name)?.url
async function json<T>(job: GarnetJob, name: string): Promise<T | null> {
  const u = url(job, name)
  if (!u) return null
  const r = await fetch(u)
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`)
  return r.json()
}

async function loadData(job: GarnetJob): Promise<Data> {
  const imageUrl = url(job, 'input/sheet.jpg')
  if (!imageUrl) throw new Error('input/sheet.jpg missing from the job')
  const [objects, equipment, stage6] = await Promise.all([
    json<{ objects: Obj[] }>(job, 'out/stage4_objects.json'),
    json<{ objects: Equipment[] }>(job, 'input/sheet_equipment_bboxes.json'),
    json<{
      associations: Record<string, { accepted: Obj[]; rejected: Obj[] }>
      unresolved: { traces_without_line_number: string[] }
    }>(job, 'out/stage6_trace_associations.json'),
  ])
  const img = new Image()
  img.src = imageUrl
  await img.decode()
  const assoc = (key: string): Assoc[] => {
    const a = stage6?.associations[key]
    if (!a) return []
    return [
      ...a.accepted.map((o) => ({ ...(o as Assoc), status: 'accepted' as const })),
      ...a.rejected.map((o) => ({ ...(o as Assoc), status: 'rejected' as const })),
    ]
  }
  return {
    imageUrl,
    width: img.naturalWidth,
    height: img.naturalHeight,
    objects: objects?.objects ?? [],
    equipment: equipment?.objects ?? [],
    lineNumbers: assoc('line_numbers'),
    instruments: assoc('instrument_tags'),
    tracesWithoutLineNumber: stage6?.unresolved.traces_without_line_number ?? [],
  }
}

/** Stable per-class hue, so a class keeps its colour across tabs and reloads. */
function hue(s: string) {
  let h = 0
  for (const c of s) h = (h * 31 + c.charCodeAt(0)) % 360
  return h
}

type Box = { id: string; bbox: BBox; color: string; label?: string; points?: { x: number; y: number; label: string }[] }

export function GarnetResults() {
  const sheets = useRunStore((s) => s.sheets)
  const selectedSheetId = useRunStore((s) => s.selectedSheetId)
  const selectSheet = useRunStore((s) => s.selectSheet)
  const setScreen = useRunStore((s) => s.setScreen)
  const done = sheets.filter((s) => s.task === 'garnet' && s.garnet?.status === 'completed')
  const sheet = done.find((s) => s.id === selectedSheetId) ?? done[0]
  const job = sheet?.garnet ?? null

  const [tab, setTab] = useState<Tab>('objects')
  const [data, setData] = useState<Data | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const saveGarnetEdits = useRunStore((s) => s.saveGarnetEdits)
  const saved = job?.edits ?? {}
  // Unsaved corrections, layered over `saved`. Reset when the sheet changes.
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const dirty = Object.keys(draft).some((id) => draft[id] !== (saved[id] ?? null))
  const edits: Edits = {
    textOf: (id, original) => draft[id] ?? saved[id] ?? original ?? '',
    isEdited: (id) => id in draft || id in saved,
    set: (id, text) => setDraft((d) => ({ ...d, [id]: text })),
  }
  const save = async () => {
    if (!sheet) return
    setSaving(true)
    setSaveError(null)
    try {
      await saveGarnetEdits(sheet.id, { ...saved, ...draft })
      setDraft({})
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  useEffect(() => {
    if (!job) return
    let live = true
    setData(null)
    setError(null)
    setDraft({})
    loadData(job)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e instanceof Error ? e.message : String(e)))
    return () => {
      live = false
    }
  }, [job?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  const boxes: Box[] = useMemo(() => {
    if (!data) return []
    const statusColor = (a: Assoc) => (a.status === 'accepted' ? 'hsl(140 70% 40%)' : 'hsl(30 95% 50%)')
    switch (tab) {
      case 'objects':
        return data.objects
          .filter((o) => !hidden.has(o.class_name))
          .map((o) => ({ id: o.id, bbox: o.bbox, color: `hsl(${hue(o.class_name)} 80% 45%)` }))
      case 'lineNumbers':
        return data.lineNumbers.map((a) => ({ id: a.id, bbox: a.bbox, color: statusColor(a), label: edits.textOf(a.id, a.text) || undefined }))
      case 'instruments':
        return data.instruments.map((a) => ({ id: a.id, bbox: a.bbox, color: statusColor(a), label: edits.textOf(a.id, a.text) || undefined }))
      case 'equipment':
        return data.equipment.map((e) => ({
          id: String(e.Index),
          bbox: e.Bounding_box_px,
          color: 'hsl(265 70% 55%)',
          label: e.Tag,
          points: (e.Ports ?? []).map((p) => ({ x: p.point_px[0], y: p.point_px[1], label: p.mark })),
        }))
    }
  }, [data, tab, hidden, draft, saved]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!job) {
    return (
      <div className="p-6">
        <Card padding={20}>No completed GARNET job. Run a sheet on the GARNET trace screen first.</Card>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col gap-3 p-6">
      <PageHeader
        title="GARNET results"
        subtitle={`${sheet.label} · job ${job.id.slice(0, 8)}…`}
        actions={
          <>
            {done.length > 1 && (
              <select
                value={sheet.id}
                onChange={(e) => {
                  selectSheet(e.target.value)
                  setSelected(null)
                }}
                style={{ height: 36 }}
              >
                {done.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            )}
            {dirty && (
              <Button variant="ghost" style={{ height: 36, borderRadius: 'var(--r-btn)' }} isDisabled={saving} onPress={() => setDraft({})}>
                Discard
              </Button>
            )}
            <Button
              variant="primary"
              style={{ height: 36, borderRadius: 'var(--r-btn)' }}
              isDisabled={!dirty || saving}
              onPress={() => void save()}
            >
              {saving ? 'Saving…' : dirty ? 'Save edits' : 'Saved'}
            </Button>
            <Button variant="ghost" style={{ height: 36, borderRadius: 'var(--r-btn)' }} onPress={() => setScreen('garnet')}>
              <ArrowLeft size={16} strokeWidth={1.5} />
              Back to run
            </Button>
          </>
        }
      />

      <div className="flex flex-wrap gap-2" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            role="tab"
            aria-selected={tab === t.key}
            onClick={() => {
              setTab(t.key)
              setSelected(null)
            }}
            style={{ border: 0, background: 'transparent', padding: 0, cursor: 'pointer' }}
          >
            <Tag tone={tab === t.key ? 'accent' : 'outline'}>
              {t.label}
              {data && ` · ${tabCount(data, t.key)}`}
            </Tag>
          </button>
        ))}
      </div>

      {saveError && <Card padding={16} style={{ color: 'var(--danger)' }}>{`Edits not saved: ${saveError}`}</Card>}
      {error && <Card padding={16} style={{ color: 'var(--danger)' }}>{error}</Card>}
      {!data && !error && <Card padding={16}>Loading…</Card>}

      {data && (
        <div className="flex min-h-0 flex-1 gap-3 max-lg:flex-col">
          <Overlay data={data} boxes={boxes} selected={selected} onSelect={setSelected} />
          <Card padding={0} className="min-h-0 overflow-auto lg:w-[460px]">
            {tab === 'objects' && (
              <ObjectsTable edits={edits} data={data} hidden={hidden} setHidden={setHidden} selected={selected} onSelect={setSelected} />
            )}
            {tab === 'lineNumbers' && (
              <AssocTable
                edits={edits}
                rows={data.lineNumbers}
                selected={selected}
                onSelect={setSelected}
                footer={`${data.tracesWithoutLineNumber.length} trace(s) without a line number: ${data.tracesWithoutLineNumber.join(', ') || '—'}`}
              />
            )}
            {tab === 'instruments' && <AssocTable edits={edits} rows={data.instruments} selected={selected} onSelect={setSelected} showClass />}
            {tab === 'equipment' && <EquipmentTable rows={data.equipment} selected={selected} onSelect={setSelected} />}
          </Card>
        </div>
      )}
    </div>
  )
}

function tabCount(d: Data, t: Tab) {
  return { objects: d.objects.length, lineNumbers: d.lineNumbers.length, equipment: d.equipment.length, instruments: d.instruments.length }[t]
}

/** Fit-to-view scale for the raster in a viewport of size `vw`×`vh`. */
const fitScale = (d: Data, vw: number, vh: number) => Math.min(vw / d.width, vh / d.height)
type View = { s: number; x: number; y: number }

/**
 * Raster + SVG box layer under one CSS transform, in image pixels.
 * Wheel zooms about the cursor, drag pans, selecting an object zooms to it.
 */
function Overlay({
  data,
  boxes,
  selected,
  onSelect,
}: {
  data: Data
  boxes: Box[]
  selected: string | null
  onSelect: (id: string | null) => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const [view, setView] = useState<View>({ s: 0, x: 0, y: 0 })
  const drag = useRef<{ px: number; py: number; moved: boolean } | null>(null)

  const size = () => {
    const r = host.current?.getBoundingClientRect()
    return { w: r?.width ?? 1, h: r?.height ?? 1 }
  }
  const fit = () => {
    const { w, h } = size()
    const s = fitScale(data, w, h)
    setView({ s, x: (w - data.width * s) / 2, y: (h - data.height * s) / 2 })
  }
  const minScale = () => {
    const { w, h } = size()
    return fitScale(data, w, h) / 2
  }
  /** Zoom by `k` keeping screen point (cx, cy) fixed. */
  const zoomAt = (k: number, cx: number, cy: number) =>
    setView((v) => {
      const s = Math.min(8, Math.max(minScale(), v.s * k))
      const r = s / v.s
      return { s, x: cx - (cx - v.x) * r, y: cy - (cy - v.y) * r }
    })
  const zoomCenter = (k: number) => {
    const { w, h } = size()
    zoomAt(k, w / 2, h / 2)
  }

  // Fit on load and whenever the sheet changes.
  useEffect(fit, [data]) // eslint-disable-line react-hooks/exhaustive-deps

  // Wheel must be a non-passive native listener, or preventDefault is ignored and the page scrolls.
  useEffect(() => {
    const el = host.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const r = el.getBoundingClientRect()
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }) // re-bind each render so zoomAt sees current data

  // Jump to the selected object: centre it, zoomed so it fills ~1/4 of the shorter side.
  useEffect(() => {
    const b = boxes.find((x) => x.id === selected)
    if (!b) return
    const { w, h } = size()
    const bw = Math.max(b.bbox.x_max - b.bbox.x_min, 1)
    const bh = Math.max(b.bbox.y_max - b.bbox.y_min, 1)
    const s = Math.min(4, Math.max(fitScale(data, w, h), Math.min(w / bw, h / bh) / 4))
    const cx = (b.bbox.x_min + b.bbox.x_max) / 2
    const cy = (b.bbox.y_min + b.bbox.y_max) / 2
    setView({ s, x: w / 2 - cx * s, y: h / 2 - cy * s })
  }, [selected]) // eslint-disable-line react-hooks/exhaustive-deps

  const stroke = Math.max(1, 2 / (view.s || 1))

  return (
    <Card padding={0} className="relative flex min-h-[320px] min-w-0 flex-1 flex-col overflow-hidden">
      <div
        className="absolute right-2 top-2 z-10 flex items-center gap-1 rounded-lg p-1"
        style={{ background: 'var(--surface)', boxShadow: '0 1px 4px rgba(0,0,0,.15)' }}
      >
        <Button isIconOnly size="sm" variant="ghost" aria-label="Zoom out" onPress={() => zoomCenter(1 / 1.4)}>
          <Minus size={14} />
        </Button>
        <span className="mono" style={{ fontSize: 11, minWidth: 44, textAlign: 'center' }}>
          {Math.round(view.s * 100)}%
        </span>
        <Button isIconOnly size="sm" variant="ghost" aria-label="Zoom in" onPress={() => zoomCenter(1.4)}>
          <Plus size={14} />
        </Button>
        <Button isIconOnly size="sm" variant="ghost" aria-label="Fit to view" onPress={fit}>
          <Maximize size={14} />
        </Button>
        <Button isIconOnly size="sm" variant="ghost" aria-label="Actual size" onPress={() => zoomCenter(1 / view.s)}>
          <span style={{ fontSize: 10, fontWeight: 600 }}>1:1</span>
        </Button>
      </div>
      <div
        ref={host}
        className="relative min-h-0 flex-1 overflow-hidden"
        style={{ cursor: drag.current ? 'grabbing' : 'grab', touchAction: 'none', background: 'var(--surface-2, #f4f4f5)' }}
        onPointerDown={(e) => {
          drag.current = { px: e.clientX, py: e.clientY, moved: false }
          e.currentTarget.setPointerCapture(e.pointerId)
        }}
        onPointerMove={(e) => {
          const d = drag.current
          if (!d) return
          const dx = e.clientX - d.px
          const dy = e.clientY - d.py
          if (!d.moved && Math.hypot(dx, dy) < 3) return
          d.moved = true
          d.px = e.clientX
          d.py = e.clientY
          setView((v) => ({ ...v, x: v.x + dx, y: v.y + dy }))
        }}
        onPointerUp={(e) => {
          const d = drag.current
          drag.current = null
          if (d?.moved) return
          // A click, not a drag: hit-test the box under the pointer (smallest wins).
          const r = e.currentTarget.getBoundingClientRect()
          const ix = (e.clientX - r.left - view.x) / view.s
          const iy = (e.clientY - r.top - view.y) / view.s
          const hit = boxes
            .filter((b) => ix >= b.bbox.x_min && ix <= b.bbox.x_max && iy >= b.bbox.y_min && iy <= b.bbox.y_max)
            .sort((a, b) => area(a.bbox) - area(b.bbox))[0]
          onSelect(hit ? hit.id : null)
        }}
      >
        {view.s > 0 && (
          <div
            className="absolute left-0 top-0"
            style={{
              width: data.width,
              height: data.height,
              transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})`,
              transformOrigin: '0 0',
            }}
          >
            <img src={data.imageUrl} alt="sheet" width={data.width} height={data.height} draggable={false} className="block select-none" />
            <svg viewBox={`0 0 ${data.width} ${data.height}`} className="pointer-events-none absolute inset-0" width={data.width} height={data.height}>
              {boxes.map((b) => {
                const on = b.id === selected
                const { x_min, y_min, x_max, y_max } = b.bbox
                return (
                  <g key={b.id}>
                    {on && (
                      <rect
                        x={x_min - stroke * 6}
                        y={y_min - stroke * 6}
                        width={x_max - x_min + stroke * 12}
                        height={y_max - y_min + stroke * 12}
                        fill="none"
                        stroke={SELECTED}
                        strokeOpacity={0.35}
                        strokeWidth={stroke * 6}
                      />
                    )}
                    <rect
                      x={x_min}
                      y={y_min}
                      width={x_max - x_min}
                      height={y_max - y_min}
                      fill={on ? SELECTED : 'none'}
                      fillOpacity={0.15}
                      stroke={on ? SELECTED : b.color}
                      strokeWidth={on ? stroke * 2.5 : stroke}
                    />
                    {b.points?.map((p) => (
                      <g key={`${p.label}-${p.x}-${p.y}`}>
                        <circle cx={p.x} cy={p.y} r={stroke * 3} fill={b.color} />
                        {on && (
                          <text x={p.x + stroke * 4} y={p.y - stroke * 4} fontSize={stroke * 9} fill={b.color}>
                            {p.label}
                          </text>
                        )}
                      </g>
                    ))}
                  </g>
                )
              })}
            </svg>
          </div>
        )}
      </div>
    </Card>
  )
}

const SELECTED = 'hsl(212 100% 48%)'
const area = (b: BBox) => (b.x_max - b.x_min) * (b.y_max - b.y_min)

const th: React.CSSProperties = { textAlign: 'left', padding: '6px 8px', fontWeight: 500, color: 'var(--muted)', position: 'sticky', top: 0, background: 'var(--surface)' }
const td: React.CSSProperties = { padding: '5px 8px', whiteSpace: 'nowrap', borderTop: '1px solid var(--separator)', verticalAlign: 'top' }
const rowStyle = (on: boolean): React.CSSProperties => ({ cursor: 'pointer', background: on ? 'rgba(59,130,246,.18)' : undefined, boxShadow: on ? 'inset 3px 0 0 hsl(212 100% 48%)' : undefined })
const bboxText = (b: BBox) => `${b.x_min},${b.y_min} → ${b.x_max},${b.y_max}`

function ObjectsTable({
  edits,
  data,
  hidden,
  setHidden,
  selected,
  onSelect,
}: {
  edits: Edits
  data: Data
  hidden: Set<string>
  setHidden: (s: Set<string>) => void
  selected: string | null
  onSelect: (id: string) => void
}) {
  const counts = useMemo(() => {
    const m = new Map<string, number>()
    data.objects.forEach((o) => m.set(o.class_name, (m.get(o.class_name) ?? 0) + 1))
    return [...m.entries()].sort((a, b) => b[1] - a[1])
  }, [data])
  const toggle = (c: string) => {
    const next = new Set(hidden)
    if (next.has(c)) next.delete(c)
    else next.add(c)
    setHidden(next)
  }
  return (
    <div style={{ fontSize: 12 }}>
      <div className="flex flex-wrap gap-1.5 p-3">
        {counts.map(([c, n]) => (
          <button key={c} type="button" onClick={() => toggle(c)} aria-pressed={!hidden.has(c)} style={{ border: 0, background: 'transparent', padding: 0, cursor: 'pointer', opacity: hidden.has(c) ? 0.4 : 1 }}>
            <Tag dot={`hsl(${hue(c)} 80% 45%)`}>{`${c} · ${n}`}</Tag>
          </button>
        ))}
      </div>
      <table className="w-full" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr><th style={th}>id</th><th style={th}>class</th><th style={th}>conf</th><th style={th}>text</th><th style={th}>bbox</th></tr>
        </thead>
        <tbody>
          {data.objects.filter((o) => !hidden.has(o.class_name)).map((o) => (
            <tr key={o.id} style={rowStyle(o.id === selected)} onClick={() => onSelect(o.id)}>
              <td style={td} className="mono">{o.id}</td>
              <td style={td}>{o.class_name}</td>
              <td style={td} className="mono">{o.confidence ?? '—'}</td>
              <TextCell edits={edits} id={o.id} original={o.text} on={o.id === selected} />
              <td style={td} className="mono">{bboxText(o.bbox)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function AssocTable({
  edits,
  rows,
  selected,
  onSelect,
  showClass,
  footer,
}: {
  edits: Edits
  rows: Assoc[]
  selected: string | null
  onSelect: (id: string) => void
  showClass?: boolean
  footer?: string
}) {
  const accepted = rows.filter((r) => r.status === 'accepted').length
  return (
    <div style={{ fontSize: 12 }}>
      <div className="flex gap-2 p-3">
        <Tag tone="success">{`${accepted} attached`}</Tag>
        <Tag tone="warning">{`${rows.length - accepted} unattached`}</Tag>
      </div>
      <table className="w-full" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={th}>text</th>
            {showClass && <th style={th}>class</th>}
            <th style={th}>trace</th>
            <th style={th}>dist px</th>
            <th style={th}>status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} style={rowStyle(r.id === selected)} onClick={() => onSelect(r.id)}>
              <TextCell edits={edits} id={r.id} original={r.text} on={r.id === selected} />
              {showClass && <td style={td}>{r.class_name}</td>}
              <td style={td} className="mono">{r.trace_id ?? '—'}</td>
              <td style={td} className="mono">{r.distance_px}</td>
              <td style={td}>
                {r.status === 'accepted' ? (
                  <Tag tone="success">attached</Tag>
                ) : (
                  <span title={`max ${r.max_distance_px ?? '?'} px`}><Tag tone="warning">{r.reason ?? 'rejected'}</Tag></span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {footer && <div className="p-3" style={{ color: 'var(--muted)' }}>{footer}</div>}
    </div>
  )
}

function EquipmentTable({
  rows,
  selected,
  onSelect,
}: {
  rows: Equipment[]
  selected: string | null
  onSelect: (id: string) => void
}) {
  const current = rows.find((r) => String(r.Index) === selected)
  return (
    <div style={{ fontSize: 12 }}>
      <table className="w-full" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr><th style={th}>tag</th><th style={th}>type</th><th style={th}>service</th><th style={th}>ports</th></tr>
        </thead>
        <tbody>
          {rows.map((e) => (
            <tr key={e.Index} style={rowStyle(String(e.Index) === selected)} onClick={() => onSelect(String(e.Index))}>
              <td style={td} className="mono">{e.Tag || '—'}</td>
              <td style={td}>{e.Equipment_type || '—'}</td>
              <td style={td}>{e.Service || '—'}</td>
              <td style={td} className="mono">{e.Ports?.length ?? 0}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {current && (
        <div className="flex flex-col gap-2 p-3" style={{ borderTop: '1px solid var(--separator)' }}>
          <div style={{ fontWeight: 600 }}>{current.Tag}</div>
          <div style={{ color: 'var(--muted)' }}>
            {current.Size || '—'} · evidence: {current.Evidence || '—'} · bbox <span className="mono">{bboxText(current.Bounding_box_px)}</span>
          </div>
          <table className="w-full" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr><th style={th}>mark</th><th style={th}>size</th><th style={th}>side</th><th style={th}>line number</th><th style={th}>point</th></tr>
            </thead>
            <tbody>
              {(current.Ports ?? []).map((p, i) => (
                <tr key={`${p.mark}-${i}`}>
                  <td style={td} className="mono">{p.mark}</td>
                  <td style={td} className="mono">{p.size || '—'}</td>
                  <td style={td}>{p.side || '—'}</td>
                  <td style={td} className="mono">{p.line_number || '—'}</td>
                  <td style={td} className="mono">{p.point_px.join(',')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

type Edits = {
  textOf: (id: string, original: string | undefined) => string
  isEdited: (id: string) => boolean
  set: (id: string, text: string) => void
}

/**
 * Text cell: read-only until its row is selected, then an input. Changes go to the draft only;
 * nothing reaches the server until Save. An edited value is marked and shows the original.
 */
function TextCell({ edits, id, original, on }: { edits: Edits; id: string; original?: string; on: boolean }) {
  const value = edits.textOf(id, original)
  const edited = edits.isEdited(id) && value !== (original ?? '')
  return (
    <td style={td} className="mono" title={edited ? `was: ${original || '(empty)'}` : undefined}>
      {on ? (
        <input
          autoFocus
          aria-label={`Text for ${id}`}
          value={value}
          onChange={(e) => edits.set(id, e.target.value)}
          onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => e.stopPropagation()}
          className="mono"
          style={{ width: 220, font: 'inherit', padding: '2px 6px', border: '1px solid var(--accent)', borderRadius: 4, background: 'var(--surface)' }}
        />
      ) : value ? (
        <>
          {value}
          {edited && <span style={{ color: 'var(--accent)', marginLeft: 6 }}>●</span>}
        </>
      ) : (
        <span style={{ color: 'var(--muted)' }}>{`(no text) ${id}`}</span>
      )}
    </td>
  )
}
