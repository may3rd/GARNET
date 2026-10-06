import { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '@heroui/react'
import { ArrowLeft, Minus, Plus } from 'lucide-react'
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

  useEffect(() => {
    if (!job) return
    let live = true
    setData(null)
    setError(null)
    loadData(job)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e instanceof Error ? e.message : String(e)))
    return () => {
      live = false
    }
  }, [job])

  const boxes: Box[] = useMemo(() => {
    if (!data) return []
    const statusColor = (a: Assoc) => (a.status === 'accepted' ? 'hsl(140 70% 40%)' : 'hsl(30 95% 50%)')
    switch (tab) {
      case 'objects':
        return data.objects
          .filter((o) => !hidden.has(o.class_name))
          .map((o) => ({ id: o.id, bbox: o.bbox, color: `hsl(${hue(o.class_name)} 80% 45%)` }))
      case 'lineNumbers':
        return data.lineNumbers.map((a) => ({ id: a.id, bbox: a.bbox, color: statusColor(a), label: a.text || undefined }))
      case 'instruments':
        return data.instruments.map((a) => ({ id: a.id, bbox: a.bbox, color: statusColor(a), label: a.text || undefined }))
      case 'equipment':
        return data.equipment.map((e) => ({
          id: String(e.Index),
          bbox: e.Bounding_box_px,
          color: 'hsl(265 70% 55%)',
          label: e.Tag,
          points: (e.Ports ?? []).map((p) => ({ x: p.point_px[0], y: p.point_px[1], label: p.mark })),
        }))
    }
  }, [data, tab, hidden])

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

      {error && <Card padding={16} style={{ color: 'var(--danger)' }}>{error}</Card>}
      {!data && !error && <Card padding={16}>Loading…</Card>}

      {data && (
        <div className="flex min-h-0 flex-1 gap-3 max-lg:flex-col">
          <Overlay data={data} boxes={boxes} selected={selected} onSelect={setSelected} />
          <Card padding={0} className="min-h-0 overflow-auto lg:w-[460px]">
            {tab === 'objects' && (
              <ObjectsTable data={data} hidden={hidden} setHidden={setHidden} selected={selected} onSelect={setSelected} />
            )}
            {tab === 'lineNumbers' && (
              <AssocTable
                rows={data.lineNumbers}
                selected={selected}
                onSelect={setSelected}
                footer={`${data.tracesWithoutLineNumber.length} trace(s) without a line number: ${data.tracesWithoutLineNumber.join(', ') || '—'}`}
              />
            )}
            {tab === 'instruments' && <AssocTable rows={data.instruments} selected={selected} onSelect={setSelected} showClass />}
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

/** Raster with an SVG box layer in image pixels; zoom is a width scale, pan is native scroll. */
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
  const [zoom, setZoom] = useState(1)
  const scroller = useRef<HTMLDivElement>(null)
  const stroke = Math.max(2, data.width / 1500)

  // Bring a box picked from the table into view.
  useEffect(() => {
    const el = selected ? scroller.current?.querySelector(`[data-box="${CSS.escape(selected)}"]`) : null
    el?.scrollIntoView({ block: 'center', inline: 'center', behavior: 'smooth' })
  }, [selected])

  return (
    <Card padding={0} className="relative flex min-h-[320px] min-w-0 flex-1 flex-col overflow-hidden">
      <div className="absolute right-2 top-2 z-10 flex gap-1">
        <Button isIconOnly size="sm" variant="secondary" aria-label="Zoom out" onPress={() => setZoom((z) => Math.max(1, z / 1.5))}>
          <Minus size={14} />
        </Button>
        <Button isIconOnly size="sm" variant="secondary" aria-label="Zoom in" onPress={() => setZoom((z) => Math.min(12, z * 1.5))}>
          <Plus size={14} />
        </Button>
      </div>
      <div ref={scroller} className="min-h-0 flex-1 overflow-auto">
        <div className="relative" style={{ width: `${zoom * 100}%` }}>
          <img src={data.imageUrl} alt="sheet" className="block w-full" draggable={false} />
          <svg
            viewBox={`0 0 ${data.width} ${data.height}`}
            className="absolute inset-0 h-full w-full"
            onClick={() => onSelect(null)}
          >
            {boxes.map((b) => {
              const on = b.id === selected
              const { x_min, y_min, x_max, y_max } = b.bbox
              return (
                <g key={b.id} data-box={b.id} onClick={(e) => { e.stopPropagation(); onSelect(b.id) }} style={{ cursor: 'pointer' }}>
                  <rect
                    x={x_min}
                    y={y_min}
                    width={x_max - x_min}
                    height={y_max - y_min}
                    fill={on ? b.color : 'transparent'}
                    fillOpacity={on ? 0.25 : 0}
                    stroke={b.color}
                    strokeWidth={on ? stroke * 2 : stroke}
                    pointerEvents="all"
                  >
                    <title>{b.label ?? b.id}</title>
                  </rect>
                  {b.points?.map((p) => (
                    <g key={`${p.label}-${p.x}-${p.y}`}>
                      <circle cx={p.x} cy={p.y} r={stroke * 3} fill={b.color} />
                      {on && (
                        <text x={p.x + stroke * 4} y={p.y - stroke * 4} fontSize={stroke * 10} fill={b.color}>
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
      </div>
    </Card>
  )
}

const th: React.CSSProperties = { textAlign: 'left', padding: '6px 8px', fontWeight: 500, color: 'var(--muted)', position: 'sticky', top: 0, background: 'var(--surface)' }
const td: React.CSSProperties = { padding: '5px 8px', whiteSpace: 'nowrap', borderTop: '1px solid var(--separator)', verticalAlign: 'top' }
const rowStyle = (on: boolean): React.CSSProperties => ({ cursor: 'pointer', background: on ? 'var(--accent-soft, rgba(59,130,246,.12))' : undefined })
const bboxText = (b: BBox) => `${b.x_min},${b.y_min} → ${b.x_max},${b.y_max}`

function ObjectsTable({
  data,
  hidden,
  setHidden,
  selected,
  onSelect,
}: {
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
              <td style={td} className="mono">{o.text || '—'}</td>
              <td style={td} className="mono">{bboxText(o.bbox)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function AssocTable({
  rows,
  selected,
  onSelect,
  showClass,
  footer,
}: {
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
              <td style={td} className="mono">{r.text || <span style={{ color: 'var(--muted)' }}>{`(no text) ${r.id}`}</span>}</td>
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
