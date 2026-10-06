import { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '@heroui/react'
import { ArrowLeft, Maximize, Minus, Plus } from 'lucide-react'
import { Card, PageHeader, Tag } from '@/components/ui/primitives'
import type { GarnetGate, GarnetJob } from '@/lib/api'
import { useRunStore } from '@/stores/runStore'

/**
 * Read-only view of one garnet/ job's outputs, straight from its artifact files.
 * Values are shown as written — no rounding, no text clean-up, no inferred tags.
 */

type BBox = { x_min: number; y_min: number; x_max: number; y_max: number }
type Obj = { id: string; class_name: string; confidence?: number; bbox: BBox; text?: string }
type Assoc = Obj & {
  trace_id: string | null
  projected_xy?: [number, number]
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
type Segment = { x1: number; y1: number; x2: number; y2: number; length_px?: number }
type Trace = { id: string; kind: 'port' | 'branch'; terminal_type?: string; segments: Segment[]; bbox: BBox }
type Data = {
  traces: Trace[]
  imageUrl: string
  width: number
  height: number
  objects: Obj[]
  equipment: Equipment[]
  lineNumbers: Assoc[]
  instruments: Assoc[]
  tracesWithoutLineNumber: string[]
  /** Traces Stage 6 associated against (Gate 2 rejections excluded): the valid reassign targets. */
  stage6TraceIds: Set<string>
}
type Tab = 'objects' | 'traces' | 'lineNumbers' | 'equipment' | 'instruments'
const TABS: { key: Tab; label: string }[] = [
  { key: 'objects', label: 'Objects' },
  { key: 'traces', label: 'Traces' },
  { key: 'lineNumbers', label: 'Line numbers' },
  { key: 'equipment', label: 'Equipment' },
  { key: 'instruments', label: 'Instrument tags' },
]
/** What each gate reviews. The first tab opens by default. */
const GATE_TABS: Record<GarnetGate, Tab[]> = {
  1: ['objects', 'equipment'],
  2: ['traces', 'objects', 'equipment'],
  3: ['lineNumbers', 'instruments', 'traces'],
}
const GATE_COPY: Record<GarnetGate, { title: string; hint: string }> = {
  1: { title: 'Gate 1 · Objects', hint: 'Reject false detections and fix tag text. Rejected objects are not traced.' },
  2: { title: 'Gate 2 · Traces', hint: 'Reject traces that follow the wrong pipe. Stage 6 runs without them.' },
  3: { title: 'Gate 3 · Line numbers', hint: 'Accept or reject each line-number attachment and fix text.' },
}

function segBox(segments: Segment[]): BBox {
  const xs = segments.flatMap((g) => [g.x1, g.x2])
  const ys = segments.flatMap((g) => [g.y1, g.y2])
  return { x_min: Math.min(...xs), y_min: Math.min(...ys), x_max: Math.max(...xs), y_max: Math.max(...ys) }
}

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
  // stage 4 exists only after trace; before that (Gate 1) the staged objects are the source.
  const objectsFile = url(job, 'out/stage4_objects.json') ? 'out/stage4_objects.json' : 'input/sheet_objects.json'
  // Branch geometry: 5c (post-merge) first, the same preference Stage 6 uses.
  const branchFile = url(job, 'out/stage5c_branch_trace_results.json')
    ? 'out/stage5c_branch_trace_results.json'
    : 'out/stage5b_branch_trace_results.json'
  const [objects, equipment, stage6, portTraces, branches] = await Promise.all([
    json<{ objects: Obj[] }>(job, objectsFile),
    json<{ objects: Equipment[] }>(job, 'input/sheet_equipment_bboxes.json'),
    json<{
      associations: Record<string, { accepted: Obj[]; rejected: Obj[] }>
      unresolved: { traces_without_line_number: string[] }
      trace_edges: { trace_id: string }[]
    }>(job, 'out/stage6_trace_associations.json'),
    json<Record<string, { terminal_type?: string; segments?: Segment[] }>>(job, 'out/stage5b_trace_results.json'),
    json<{ branches: Record<string, { terminal_type?: string; segments?: Segment[] }> }>(job, branchFile),
  ])
  const traces: Trace[] = [
    ...Object.entries(portTraces ?? {}).map(([id, t]) => ({ id, kind: 'port' as const, ...t })),
    ...Object.entries(branches?.branches ?? {}).map(([id, t]) => ({ id, kind: 'branch' as const, ...t })),
  ]
    .filter((t) => t.segments && t.segments.length > 0)
    .map((t) => ({ id: t.id, kind: t.kind, terminal_type: t.terminal_type, segments: t.segments!, bbox: segBox(t.segments!) }))
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
    traces,
    imageUrl,
    width: img.naturalWidth,
    height: img.naturalHeight,
    objects: objects?.objects ?? [],
    equipment: equipment?.objects ?? [],
    lineNumbers: assoc('line_numbers'),
    instruments: assoc('instrument_tags'),
    tracesWithoutLineNumber: stage6?.unresolved.traces_without_line_number ?? [],
    stage6TraceIds: new Set((stage6?.trace_edges ?? []).map((e) => e.trace_id)),
  }
}

/** Stable per-class hue, so a class keeps its colour across tabs and reloads. */
function hue(s: string) {
  let h = 0
  for (const c of s) h = (h * 31 + c.charCodeAt(0)) % 360
  return h
}

type Box = {
  id: string
  bbox: BBox
  color: string
  label?: string
  points?: { x: number; y: number; label: string }[]
  /** Polyline drawn instead of the rectangle (traces); bbox stays for hit-test and jump. */
  lines?: Segment[]
  /** Rejected at a gate: drawn grey and dashed. */
  dim?: boolean
  /** Region to frame when this box is selected, if wider than the box (label + its trace). */
  focus?: BBox
}

const union = (a: BBox, b: BBox): BBox => ({
  x_min: Math.min(a.x_min, b.x_min),
  y_min: Math.min(a.y_min, b.y_min),
  x_max: Math.max(a.x_max, b.x_max),
  y_max: Math.max(a.y_max, b.y_max),
})

export function GarnetResults() {
  const sheets = useRunStore((s) => s.sheets)
  const selectedSheetId = useRunStore((s) => s.selectedSheetId)
  const selectSheet = useRunStore((s) => s.selectSheet)
  const setScreen = useRunStore((s) => s.setScreen)
  const done = sheets.filter(
    (s) => s.task === 'garnet' && (s.garnet?.status === 'completed' || s.garnet?.status === 'awaiting_gate')
  )
  const sheet = done.find((s) => s.id === selectedSheetId) ?? done[0]
  const job = sheet?.garnet ?? null

  const gate = job?.status === 'awaiting_gate' ? job.gate : null
  const tabs = gate ? TABS.filter((t) => GATE_TABS[gate].includes(t.key)) : TABS
  const [tab, setTab] = useState<Tab>('objects')
  // Gate decisions, cleared whenever the job moves on.
  const [rejected, setRejected] = useState<Set<string>>(new Set())
  const [decisions, setDecisions] = useState<Record<string, 'accepted' | 'rejected'>>({})
  // Gate 1 geometry drafts: new geometry for existing ids, and boxes drawn here (ids `new_N`).
  const [moved, setMoved] = useState<Record<string, BBox>>({})
  const [reclassed, setReclassed] = useState<Record<string, string>>({})
  // Gate 3: line number -> trace it belongs on; `picking` arms the next canvas click.
  const [reassigned, setReassigned] = useState<Record<string, string>>({})
  const [picking, setPicking] = useState(false)
  const [added, setAdded] = useState<Obj[]>([])
  const [addMode, setAddMode] = useState(false)
  const [addClass, setAddClass] = useState('instrument tag')
  const nextNew = useRef(0)
  const confirmGarnetGate = useRunStore((s) => s.confirmGarnetGate)
  const toggleReject = (id: string) => {
    // A box drawn in this session has nothing upstream to reject: just remove it.
    if (added.some((o) => o.id === id)) {
      setAdded((a) => a.filter((o) => o.id !== id))
      setSelected(null)
      return
    }
    setRejected((r) => {
      const next = new Set(r)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }
  const confirm = async () => {
    if (!sheet || !gate) return
    setSaving(true)
    setSaveError(null)
    try {
      const isNew = (id: string) => id.startsWith('new_')
      await confirmGarnetGate(sheet.id, gate, {
        rejected: gate === 3 ? [] : [...rejected],
        edits: gate === 2 ? {} : Object.fromEntries(Object.entries(draft).filter(([id]) => !isNew(id))),
        added: gate === 1 ? added.map((o) => ({ class_name: o.class_name, bbox: o.bbox, text: draft[o.id] ?? '' })) : [],
        moved: gate === 1 ? Object.fromEntries(Object.entries(moved).filter(([id]) => !isNew(id))) : {},
        reclassed: gate === 1 ? reclassed : {},
        reassigned: gate === 3 ? reassigned : {},
        decisions: gate === 3 ? decisions : {},
      })
      setScreen('garnet')
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }
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
    setRejected(new Set())
    setDecisions({})
    setMoved({})
    setReclassed({})
    setReassigned({})
    setPicking(false)
    setAdded([])
    setAddMode(false)
    setSelected(null)
    setTab(gate ? GATE_TABS[gate][0] : 'objects')
    loadData(job)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e instanceof Error ? e.message : String(e)))
    return () => {
      live = false
    }
  }, [job?.id, job?.status, job?.gate]) // eslint-disable-line react-hooks/exhaustive-deps

  const view: Data | null = useMemo(
    () =>
      data && {
        ...data,
        objects: [
          ...data.objects.map((o) => ({
            ...o,
            bbox: moved[o.id] ?? o.bbox,
            class_name: reclassed[o.id] ?? o.class_name,
          })),
          ...added,
        ],
      },
    [data, moved, added, reclassed]
  )
  const onDraw = (bbox: BBox) => {
    const cls = addClass.trim()
    if (!cls) return
    nextNew.current += 1
    const id = `new_${nextNew.current}`
    setAdded((a) => [...a, { id, class_name: cls, confidence: 1, bbox }])
    setHidden((h) => {
      const next = new Set(h)
      next.delete(cls)
      return next
    })
    setAddMode(false)
    setSelected(id)
  }
  /** Class edits: a drawn box changes in place; an existing one records an override (cleared when reverted). */
  const onReclass = (id: string, cls: string) => {
    if (added.some((o) => o.id === id)) {
      setAdded((a) => a.map((o) => (o.id === id ? { ...o, class_name: cls } : o)))
      return
    }
    const original = data?.objects.find((o) => o.id === id)?.class_name
    setReclassed((r) => {
      const { [id]: _, ...rest } = r
      return cls === original ? rest : { ...rest, [id]: cls }
    })
  }
  /** Point a line number at another trace; picking its own system trace clears the override. */
  const reassign = (id: string, traceId: string) => {
    const original = data?.lineNumbers.find((a) => a.id === id)?.trace_id
    setReassigned((r) => {
      const { [id]: _, ...rest } = r
      return traceId === original ? rest : { ...rest, [id]: traceId }
    })
    // Reassigning means "this label is right, on that pipe": a reject verdict would contradict it.
    if (traceId !== original)
      setDecisions((d) => {
        const { [id]: _, ...rest } = d
        return rest
      })
  }
  /** Canvas selection; while picking, a click on a pipe completes the reassignment instead. */
  const onCanvasSelect = (id: string | null) => {
    if (picking) {
      if (id?.startsWith('trace:') && selected) reassign(selected, id.slice('trace:'.length))
      setPicking(false)
      return
    }
    setSelected(id)
  }
  const onUnassignSilently = (id: string) =>
    setReassigned((r) => {
      const { [id]: _, ...rest } = r
      return rest
    })
  const traceOf = (a: Assoc) => reassigned[a.id] ?? a.trace_id
  const onMove = (id: string, bbox: BBox) => {
    if (added.some((o) => o.id === id)) setAdded((a) => a.map((o) => (o.id === id ? { ...o, bbox } : o)))
    else setMoved((m) => ({ ...m, [id]: bbox }))
  }

  const traceById = useMemo(() => new Map((data?.traces ?? []).map((t) => [t.id, t])), [data])
  // The selected label's trace: shown on the line-number and instrument tabs.
  const highlight = useMemo(() => {
    if (!data || (tab !== 'lineNumbers' && tab !== 'instruments')) return null
    const a = (tab === 'lineNumbers' ? data.lineNumbers : data.instruments).find((x) => x.id === selected)
    const tid = a ? traceOf(a) : null
    const t = tid ? traceById.get(tid) : undefined
    // A reassigned label has no projected point on its new trace; don't show the old one.
    return t ? { segments: t.segments, point: a && reassigned[a.id] ? undefined : a!.projected_xy } : null
  }, [data, tab, selected, traceById, reassigned]) // eslint-disable-line react-hooks/exhaustive-deps

  const boxes: Box[] = useMemo(() => {
    if (!data || !view) return []
    const statusColor = (a: Assoc) =>
      (reassigned[a.id] ? 'accepted' : (decisions[a.id] ?? a.status)) === 'accepted' ? 'hsl(140 70% 40%)' : 'hsl(30 95% 50%)'
    switch (tab) {
      case 'objects':
        return view.objects
          .filter((o) => !hidden.has(o.class_name))
          .map((o) => ({
            id: o.id,
            bbox: o.bbox,
            color: `hsl(${hue(o.class_name)} 80% 45%)`,
            dim: rejected.has(o.id),
            label: edits.textOf(o.id, o.text) || undefined,
          }))
      case 'traces':
        return data.traces.map((t) => ({
          id: t.id,
          bbox: t.bbox,
          color: t.kind === 'port' ? 'hsl(165 85% 32%)' : 'hsl(320 75% 45%)',
          lines: t.segments,
          dim: rejected.has(t.id),
        }))
      case 'lineNumbers':
      case 'instruments':
        return (tab === 'lineNumbers' ? data.lineNumbers : data.instruments).map((a): Box => {
          const tid = traceOf(a)
          const t = tid ? traceById.get(tid) : undefined
          return {
            id: a.id,
            bbox: a.bbox,
            color: statusColor(a),
            label: edits.textOf(a.id, a.text) || undefined,
            focus: t ? union(a.bbox, t.bbox) : undefined,
          }
        }).concat(
          // While picking, every eligible pipe is a click target.
          picking && tab === 'lineNumbers'
            ? data.traces
                .filter((t) => data.stage6TraceIds.has(t.id))
                .map((t) => ({ id: `trace:${t.id}`, bbox: t.bbox, color: 'hsl(265 80% 55%)', lines: t.segments }))
            : []
        )
      case 'equipment':
        return data.equipment.map((e) => ({
          id: String(e.Index),
          bbox: e.Bounding_box_px,
          color: 'hsl(265 70% 55%)',
          label: e.Tag,
          points: (e.Ports ?? []).map((p) => ({ x: p.point_px[0], y: p.point_px[1], label: p.mark })),
        }))
    }
  }, [data, view, tab, hidden, draft, saved, rejected, decisions, traceById, reassigned, picking]) // eslint-disable-line react-hooks/exhaustive-deps

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
        title={gate ? `GARNET review · Gate ${gate}` : 'GARNET results'}
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
            {gate ? (
              <Button
                variant="primary"
                style={{ height: 36, borderRadius: 'var(--r-btn)' }}
                isDisabled={saving || !data}
                onPress={() => void confirm()}
              >
                {saving ? 'Confirming…' : `Confirm Gate ${gate} & continue`}
              </Button>
            ) : (
              <>
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
              </>
            )}
            <Button variant="ghost" style={{ height: 36, borderRadius: 'var(--r-btn)' }} onPress={() => setScreen('garnet')}>
              <ArrowLeft size={16} strokeWidth={1.5} />
              Back to run
            </Button>
          </>
        }
      />

      {gate && (
        <Card padding={14} className="flex flex-wrap items-center gap-3">
          <Tag tone="warning">{GATE_COPY[gate].title}</Tag>
          <span style={{ fontSize: 13, color: 'var(--muted)' }}>{GATE_COPY[gate].hint}</span>
          <span className="mono" style={{ fontSize: 12, marginLeft: 'auto' }}>
            {gate === 3
              ? `${Object.keys(decisions).length} verdict(s) · ${Object.keys(reassigned).length} reassigned · ${Object.keys(draft).length} text fix(es)`
              : `${rejected.size} rejected${
                  gate === 1
                    ? ` · ${added.length} added · ${Object.keys(moved).length} moved · ${Object.keys(reclassed).length} reclassed · ${Object.keys(draft).length} text fix(es)`
                    : ''
                }`}
          </span>
          {gate === 1 && tab === 'objects' && (
            <div className="flex w-full items-center gap-2">
              <Button
                size="sm"
                variant={addMode ? 'primary' : 'secondary'}
                aria-pressed={addMode}
                isDisabled={!addClass.trim()}
                onPress={() => setAddMode((m) => !m)}
              >
                <Plus size={14} />
                {addMode ? 'Drag on the drawing…' : 'Add box'}
              </Button>
              <label className="flex items-center gap-1.5" style={{ fontSize: 12 }}>
                class
                {/* Suggestions are this sheet's own classes; free text is allowed for a class it lacks. */}
                <input
                  list="garnet-classes"
                  value={addClass}
                  onChange={(e) => setAddClass(e.target.value)}
                  style={{ height: 28, padding: '0 6px', border: '1px solid var(--border)', borderRadius: 4, background: 'var(--surface)' }}
                />
                <datalist id="garnet-classes">
                  {[...new Set(data?.objects.map((o) => o.class_name) ?? [])].sort().map((c) => (
                    <option key={c} value={c} />
                  ))}
                </datalist>
              </label>
              <span style={{ fontSize: 12, color: 'var(--muted)' }}>
                Select a box to drag it or pull its corners.
              </span>
              {selected && selected in moved && (
                <Button
                  size="sm"
                  variant="ghost"
                  onPress={() =>
                    setMoved((m) => {
                      const { [selected]: _, ...rest } = m
                      return rest
                    })
                  }
                >
                  Reset box
                </Button>
              )}
            </div>
          )}
        </Card>
      )}

      <div className="flex flex-wrap gap-2" role="tablist">
        {tabs.map((t) => (
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
              {view && ` · ${tabCount(view, t.key)}`}
            </Tag>
          </button>
        ))}
      </div>

      {saveError && <Card padding={16} style={{ color: 'var(--danger)' }}>{`Edits not saved: ${saveError}`}</Card>}
      {error && <Card padding={16} style={{ color: 'var(--danger)' }}>{error}</Card>}
      {!data && !error && <Card padding={16}>Loading…</Card>}

      {data && (
        <div className="flex min-h-0 flex-1 gap-3 max-lg:flex-col">
          <Overlay
            data={data}
            boxes={boxes}
            selected={selected}
            onSelect={onCanvasSelect}
            editable={gate === 1 && tab === 'objects'}
            addMode={addMode}
            onDraw={onDraw}
            onMove={onMove}
            highlight={highlight}
          />
          <Card padding={0} className="min-h-0 overflow-auto lg:w-[460px]">
            {tab === 'objects' && (
              <ObjectsTable edits={edits} rejectable={gate === 1} rejected={rejected} onReject={toggleReject} data={view!} geometryEdited={(id) => id in moved || id.startsWith('new_')}
                onReclass={gate === 1 ? onReclass : undefined}
                classEdited={(id) => id in reclassed} hidden={hidden} setHidden={setHidden} selected={selected} onSelect={setSelected} />
            )}
            {tab === 'lineNumbers' && (
              <AssocTable
                edits={edits}
                decisions={decisions}
                reassigned={reassigned}
                picking={picking}
                onPick={gate === 3 ? () => setPicking((x) => !x) : undefined}
                onUnassign={gate === 3 ? (id) => reassign(id, data.lineNumbers.find((a) => a.id === id)?.trace_id ?? '') : undefined}
                onDecide={
                  gate === 3
                    ? (id, v, system) => {
                        if (v === 'rejected') onUnassignSilently(id)
                        setDecisions((d) => {
                          // Matching the system's verdict is not an override: drop it.
                          const { [id]: _, ...rest } = d
                          return v === system ? rest : { ...rest, [id]: v }
                        })
                      }
                    : undefined
                }
                rows={data.lineNumbers}
                selected={selected}
                onSelect={setSelected}
                footer={`${data.tracesWithoutLineNumber.length} trace(s) without a line number: ${data.tracesWithoutLineNumber.join(', ') || '—'}`}
              />
            )}
            {tab === 'instruments' && <AssocTable edits={edits} decisions={{}} rows={data.instruments} selected={selected} onSelect={setSelected} showClass />}
            {tab === 'traces' && (
              <TracesTable
                rows={data.traces}
                rejectable={gate === 2}
                rejected={rejected}
                onReject={toggleReject}
                selected={selected}
                onSelect={setSelected}
              />
            )}
            {tab === 'equipment' && <EquipmentTable rows={data.equipment} selected={selected} onSelect={setSelected} />}
          </Card>
        </div>
      )}
    </div>
  )
}

function tabCount(d: Data, t: Tab) {
  return { objects: d.objects.length, traces: d.traces.length, lineNumbers: d.lineNumbers.length, equipment: d.equipment.length, instruments: d.instruments.length }[t]
}

/** Fit-to-view scale for the raster in a viewport of size `vw`×`vh`. */
const fitScale = (d: Data, vw: number, vh: number) => Math.min(vw / d.width, vh / d.height)
type View = { s: number; x: number; y: number }

/**
 * Raster + SVG box layer under one CSS transform, in image pixels.
 * Wheel zooms about the cursor, drag pans, selecting an object zooms to it.
 */
type Corner = 'nw' | 'ne' | 'sw' | 'se'
/** What a pointer drag is doing. Pan is the default; the rest exist only in edit mode. */
type DragMode =
  | { kind: 'pan' }
  | { kind: 'draw'; x0: number; y0: number }
  | { kind: 'move'; id: string; x0: number; y0: number; start: BBox }
  | { kind: 'resize'; id: string; corner: Corner; start: BBox }

function Overlay({
  data,
  boxes,
  selected,
  onSelect,
  editable = false,
  addMode = false,
  onDraw,
  onMove,
  highlight,
}: {
  data: Data
  boxes: Box[]
  selected: string | null
  onSelect: (id: string | null) => void
  /** Gate 1: the selected box can be dragged and resized. */
  editable?: boolean
  /** Next drag on empty canvas draws a new box instead of panning. */
  addMode?: boolean
  onDraw?: (bbox: BBox) => void
  onMove?: (id: string, bbox: BBox) => void
  /** Context drawn under the boxes: the trace a selected label attached to, and where. */
  highlight?: { segments: Segment[]; point?: [number, number] } | null
}) {
  const host = useRef<HTMLDivElement>(null)
  const [view, setView] = useState<View>({ s: 0, x: 0, y: 0 })
  const drag = useRef<{ px: number; py: number; moved: boolean; mode: DragMode } | null>(null)
  // Geometry being dragged right now; committed to the parent on pointer-up only.
  const [live, setLive] = useState<{ id: string; bbox: BBox } | null>(null)

  /** Screen event -> image pixel, clamped to the raster. */
  const toImg = (e: React.PointerEvent, el: HTMLElement) => {
    const r = el.getBoundingClientRect()
    const x = (e.clientX - r.left - view.x) / view.s
    const y = (e.clientY - r.top - view.y) / view.s
    return { x: Math.min(data.width, Math.max(0, x)), y: Math.min(data.height, Math.max(0, y)) }
  }
  const norm = (x0: number, y0: number, x1: number, y1: number): BBox => ({
    x_min: Math.round(Math.min(x0, x1)),
    y_min: Math.round(Math.min(y0, y1)),
    x_max: Math.round(Math.max(x0, x1)),
    y_max: Math.round(Math.max(y0, y1)),
  })
  const startMode = (pt: { x: number; y: number }): DragMode => {
    const sel = editable ? boxes.find((b) => b.id === selected && !b.lines) : undefined
    if (sel) {
      const b = sel.bbox
      const tol = 10 / view.s
      const corners: [Corner, number, number][] = [
        ['nw', b.x_min, b.y_min], ['ne', b.x_max, b.y_min], ['sw', b.x_min, b.y_max], ['se', b.x_max, b.y_max],
      ]
      const hit = corners.find(([, cx, cy]) => Math.abs(pt.x - cx) <= tol && Math.abs(pt.y - cy) <= tol)
      if (hit) return { kind: 'resize', id: sel.id, corner: hit[0], start: b }
      if (pt.x >= b.x_min && pt.x <= b.x_max && pt.y >= b.y_min && pt.y <= b.y_max)
        return { kind: 'move', id: sel.id, x0: pt.x, y0: pt.y, start: b }
    }
    if (editable && addMode) return { kind: 'draw', x0: pt.x, y0: pt.y }
    return { kind: 'pan' }
  }
  const dragBox = (mode: DragMode, pt: { x: number; y: number }): { id: string; bbox: BBox } | null => {
    if (mode.kind === 'draw') return { id: '__draw__', bbox: norm(mode.x0, mode.y0, pt.x, pt.y) }
    if (mode.kind === 'move') {
      const b = mode.start
      const w = b.x_max - b.x_min
      const h = b.y_max - b.y_min
      // Keep the whole box on the raster while moving.
      const x = Math.min(data.width - w, Math.max(0, b.x_min + pt.x - mode.x0))
      const y = Math.min(data.height - h, Math.max(0, b.y_min + pt.y - mode.y0))
      return { id: mode.id, bbox: norm(x, y, x + w, y + h) }
    }
    if (mode.kind === 'resize') {
      const b = mode.start
      const ax = mode.corner.endsWith('w') ? b.x_max : b.x_min // fixed corner opposite the handle
      const ay = mode.corner.startsWith('n') ? b.y_max : b.y_min
      return { id: mode.id, bbox: norm(ax, ay, pt.x, pt.y) }
    }
    return null
  }

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
    const f = b.focus ?? b.bbox
    const bw = Math.max(f.x_max - f.x_min, 1)
    const bh = Math.max(f.y_max - f.y_min, 1)
    // A focus region is framed whole (with margin); a lone box fills ~1/4 of the view.
    const want = b.focus ? Math.min(w / bw, h / bh) * 0.85 : Math.min(w / bw, h / bh) / 4
    const s = Math.min(4, Math.max(fitScale(data, w, h), want))
    const cx = (f.x_min + f.x_max) / 2
    const cy = (f.y_min + f.y_max) / 2
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
        style={{
          cursor: editable && addMode ? 'crosshair' : drag.current ? 'grabbing' : 'grab',
          touchAction: 'none',
          background: 'var(--surface-2, #f4f4f5)',
        }}
        onPointerDown={(e) => {
          const mode = startMode(toImg(e, e.currentTarget))
          drag.current = { px: e.clientX, py: e.clientY, moved: false, mode }
          e.currentTarget.setPointerCapture(e.pointerId)
        }}
        onPointerMove={(e) => {
          const d = drag.current
          if (!d) return
          const dx = e.clientX - d.px
          const dy = e.clientY - d.py
          if (!d.moved && Math.hypot(dx, dy) < 3) return
          d.moved = true
          if (d.mode.kind !== 'pan') {
            setLive(dragBox(d.mode, toImg(e, e.currentTarget)))
            return
          }
          d.px = e.clientX
          d.py = e.clientY
          setView((v) => ({ ...v, x: v.x + dx, y: v.y + dy }))
        }}
        onPointerUp={(e) => {
          const d = drag.current
          drag.current = null
          const done = d && d.moved && d.mode.kind !== 'pan' ? dragBox(d.mode, toImg(e, e.currentTarget)) : null
          setLive(null)
          if (done) {
            const { bbox } = done
            // A slip of the mouse is not a box: need a few image pixels each way.
            if (bbox.x_max - bbox.x_min < 4 || bbox.y_max - bbox.y_min < 4) return
            if (d!.mode.kind === 'draw') onDraw?.(bbox)
            else onMove?.(done.id, bbox)
            return
          }
          if (d?.moved) return
          // A click, not a drag: hit-test the box under the pointer (smallest wins).
          const r = e.currentTarget.getBoundingClientRect()
          const ix = (e.clientX - r.left - view.x) / view.s
          const iy = (e.clientY - r.top - view.y) / view.s
          // Polylines hit by distance to the drawn line (nearest wins); rectangles by
          // containment (smallest wins). A pipe near the pointer beats the box it runs through.
          const tol = 8 / view.s
          const lineHit = boxes
            .filter((b) => b.lines)
            .map((b) => ({ b, d: Math.min(...b.lines!.map((g) => segDist(ix, iy, g))) }))
            .filter((x) => x.d <= tol)
            .sort((a, b) => a.d - b.d)[0]?.b
          const hit =
            lineHit ??
            boxes
              .filter((b) => !b.lines && ix >= b.bbox.x_min && ix <= b.bbox.x_max && iy >= b.bbox.y_min && iy <= b.bbox.y_max)
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
              {highlight && (
                <g>
                  {highlight.segments.map((g, k) => (
                    <line
                      key={k}
                      x1={g.x1}
                      y1={g.y1}
                      x2={g.x2}
                      y2={g.y2}
                      stroke={TRACE_HIGHLIGHT}
                      strokeOpacity={0.85}
                      strokeWidth={stroke * 5}
                      strokeLinecap="round"
                    />
                  ))}
                  {highlight.point && (
                    <circle cx={highlight.point[0]} cy={highlight.point[1]} r={stroke * 7} fill="white" stroke={TRACE_HIGHLIGHT} strokeWidth={stroke * 3} />
                  )}
                </g>
              )}
              {boxes.map((b) => {
                const on = b.id === selected
                const { x_min, y_min, x_max, y_max } = live?.id === b.id ? live.bbox : b.bbox
                return (
                  <g key={b.id}>
                    {on && !b.lines && (
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
                    {b.lines ? (
                      b.lines.map((g, k) => (
                        <line
                          key={k}
                          x1={g.x1}
                          y1={g.y1}
                          x2={g.x2}
                          y2={g.y2}
                          stroke={on ? SELECTED : b.dim ? 'hsl(0 0% 55%)' : b.color}
                          strokeWidth={on ? stroke * 4 : stroke * 1.5}
                          strokeDasharray={b.dim ? `${stroke * 6} ${stroke * 4}` : undefined}
                          strokeLinecap="round"
                        />
                      ))
                    ) : (
                      <rect
                        x={x_min}
                        y={y_min}
                        width={x_max - x_min}
                        height={y_max - y_min}
                        fill={on ? SELECTED : 'none'}
                        fillOpacity={0.15}
                        stroke={on ? SELECTED : b.dim ? 'hsl(0 0% 55%)' : b.color}
                        strokeWidth={on ? stroke * 2.5 : stroke}
                        strokeDasharray={b.dim ? `${stroke * 4} ${stroke * 3}` : undefined}
                      />
                    )}
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
              {live?.id === '__draw__' && (
                <rect
                  x={live.bbox.x_min}
                  y={live.bbox.y_min}
                  width={live.bbox.x_max - live.bbox.x_min}
                  height={live.bbox.y_max - live.bbox.y_min}
                  fill={SELECTED}
                  fillOpacity={0.12}
                  stroke={SELECTED}
                  strokeWidth={stroke * 2}
                  strokeDasharray={`${stroke * 5} ${stroke * 3}`}
                />
              )}
              {editable &&
                (() => {
                  const b = boxes.find((x) => x.id === selected && !x.lines)
                  if (!b) return null
                  const g = live?.id === b.id ? live.bbox : b.bbox
                  const k = 8 / view.s
                  return (
                    [[g.x_min, g.y_min], [g.x_max, g.y_min], [g.x_min, g.y_max], [g.x_max, g.y_max]] as const
                  ).map(([cx, cy], i) => (
                    <rect key={i} x={cx - k / 2} y={cy - k / 2} width={k} height={k} fill="white" stroke={SELECTED} strokeWidth={stroke * 1.5} />
                  ))
                })()}
            </svg>
          </div>
        )}
      </div>
    </Card>
  )
}

const SELECTED = 'hsl(212 100% 48%)'
const TRACE_HIGHLIGHT = 'hsl(330 95% 50%)'
/** Distance from point (x, y) to segment g, in image pixels. */
function segDist(x: number, y: number, g: Segment) {
  const dx = g.x2 - g.x1
  const dy = g.y2 - g.y1
  const len2 = dx * dx + dy * dy
  const t = len2 ? Math.max(0, Math.min(1, ((x - g.x1) * dx + (y - g.y1) * dy) / len2)) : 0
  return Math.hypot(x - (g.x1 + t * dx), y - (g.y1 + t * dy))
}
const area = (b: BBox) => (b.x_max - b.x_min) * (b.y_max - b.y_min)

const th: React.CSSProperties = { textAlign: 'left', padding: '6px 8px', fontWeight: 500, color: 'var(--muted)', position: 'sticky', top: 0, background: 'var(--surface)' }
const td: React.CSSProperties = { padding: '5px 8px', whiteSpace: 'nowrap', borderTop: '1px solid var(--separator)', verticalAlign: 'top' }
const rowStyle = (on: boolean): React.CSSProperties => ({ cursor: 'pointer', background: on ? 'rgba(59,130,246,.18)' : undefined, boxShadow: on ? 'inset 3px 0 0 hsl(212 100% 48%)' : undefined })
const bboxText = (b: BBox) => `${b.x_min},${b.y_min} → ${b.x_max},${b.y_max}`

function ObjectsTable({
  edits,
  geometryEdited,
  onReclass,
  classEdited,
  rejectable,
  rejected,
  onReject,
  data,
  hidden,
  setHidden,
  selected,
  onSelect,
}: {
  edits: Edits
  geometryEdited?: (id: string) => boolean
  /** Gate 1: the selected row's class becomes an input. */
  onReclass?: (id: string, cls: string) => void
  classEdited?: (id: string) => boolean
  rejectable: boolean
  rejected: Set<string>
  onReject: (id: string) => void
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
          <tr>
            {rejectable && <th style={th} />}
            <th style={th}>id</th><th style={th}>class</th><th style={th}>conf</th><th style={th}>text</th><th style={th}>bbox</th>
          </tr>
        </thead>
        <tbody>
          {data.objects.filter((o) => !hidden.has(o.class_name)).map((o) => (
            <tr key={o.id} style={{ ...rowStyle(o.id === selected), ...(rejected.has(o.id) ? REJECTED_ROW : {}) }} onClick={() => onSelect(o.id)}>
              {rejectable && <RejectCell on={rejected.has(o.id)} onToggle={() => onReject(o.id)} />}
              <td style={td} className="mono">{o.id}</td>
              <td style={td}>
                {onReclass && o.id === selected ? (
                  <ClassInput id={o.id} value={o.class_name} onCommit={(cls) => onReclass(o.id, cls)} />
                ) : (
                  o.class_name
                )}
                {classEdited?.(o.id) && <span style={{ color: 'var(--accent)', marginLeft: 6 }}>●</span>}
              </td>
              <td style={td} className="mono">{o.confidence ?? '—'}</td>
              <TextCell edits={edits} id={o.id} original={o.text} on={o.id === selected} />
              <td style={td} className="mono">
                {bboxText(o.bbox)}
                {geometryEdited?.(o.id) && <span style={{ color: 'var(--accent)', marginLeft: 6 }}>●</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function AssocTable({
  edits,
  decisions,
  reassigned = {},
  picking = false,
  onPick,
  onUnassign,
  onDecide,
  rows,
  selected,
  onSelect,
  showClass,
  footer,
}: {
  edits: Edits
  decisions: Record<string, 'accepted' | 'rejected'>
  reassigned?: Record<string, string>
  picking?: boolean
  /** Gate 3: arm/disarm picking a trace on the canvas for the selected row. */
  onPick?: () => void
  onUnassign?: (id: string) => void
  /** Present at Gate 3: each row gets accept / reject buttons. */
  onDecide?: (id: string, verdict: 'accepted' | 'rejected', system: 'accepted' | 'rejected') => void
  rows: Assoc[]
  selected: string | null
  onSelect: (id: string) => void
  showClass?: boolean
  footer?: string
}) {
  const eff = (r: Assoc) => (reassigned[r.id] ? 'accepted' : (decisions[r.id] ?? r.status))
  const accepted = rows.filter((r) => eff(r) === 'accepted').length
  return (
    <div style={{ fontSize: 12 }}>
      <div className="flex gap-2 p-3">
        <Tag tone="success">{`${accepted} attached`}</Tag>
        <Tag tone="warning">{`${rows.length - accepted} unattached`}</Tag>
      </div>
      <table className="w-full" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={th}>status</th>
            <th style={th}>text</th>
            {showClass && <th style={th}>class</th>}
            <th style={th}>trace</th>
            <th style={th}>dist px</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} style={rowStyle(r.id === selected)} onClick={() => onSelect(r.id)}>
              <td style={td}>
                {onDecide ? (
                  <span className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                    {(['accepted', 'rejected'] as const).map((v) => {
                      const current = eff(r) === v
                      return (
                        <button
                          key={v}
                          type="button"
                          aria-pressed={current}
                          onClick={() => onDecide(r.id, v, r.status)}
                          style={{ border: 0, background: 'transparent', padding: 0, cursor: 'pointer', opacity: current ? 1 : 0.35 }}
                        >
                          <Tag tone={v === 'accepted' ? 'success' : 'warning'}>{v === 'accepted' ? 'accept' : 'reject'}</Tag>
                        </button>
                      )
                    })}
                    {decisions[r.id] && <span style={{ color: 'var(--accent)' }} title={`system: ${r.status}${r.reason ? ` (${r.reason})` : ''}`}>●</span>}
                  </span>
                ) : (decisions[r.id] ?? r.status) === 'accepted' ? (
                  <Tag tone="success">attached</Tag>
                ) : (
                  <span title={`max ${r.max_distance_px ?? '?'} px`}><Tag tone="warning">{r.reason ?? 'rejected'}</Tag></span>
                )}
              </td>
              <TextCell edits={edits} id={r.id} original={r.text} on={r.id === selected} />
              {showClass && <td style={td}>{r.class_name}</td>}
              <td style={td} className="mono" title={reassigned[r.id] ? `system: ${r.trace_id ?? 'none'}` : undefined}>
                {reassigned[r.id] ?? r.trace_id ?? '—'}
                {reassigned[r.id] && <span style={{ color: 'var(--accent)', marginLeft: 6 }}>●</span>}
                {onPick && r.id === selected && (
                  <span className="ml-2 inline-flex gap-1" onClick={(e) => e.stopPropagation()}>
                    <button type="button" aria-pressed={picking} onClick={onPick} style={{ border: 0, background: 'transparent', padding: 0, cursor: 'pointer' }}>
                      <Tag tone={picking ? 'accent' : 'outline'}>{picking ? 'click a pipe…' : 'reassign'}</Tag>
                    </button>
                    {reassigned[r.id] && onUnassign && (
                      <button type="button" onClick={() => onUnassign(r.id)} style={{ border: 0, background: 'transparent', padding: 0, cursor: 'pointer' }}>
                        <Tag>undo</Tag>
                      </button>
                    )}
                  </span>
                )}
              </td>
              <td style={td} className="mono">{r.distance_px}</td>
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

const REJECTED_ROW: React.CSSProperties = { textDecoration: 'line-through', color: 'var(--muted)' }

function RejectCell({ on, onToggle }: { on: boolean; onToggle: () => void }) {
  return (
    <td style={td} onClick={(e) => e.stopPropagation()}>
      <button
        type="button"
        aria-pressed={on}
        onClick={onToggle}
        style={{ border: 0, background: 'transparent', padding: 0, cursor: 'pointer', textDecoration: 'none' }}
      >
        <Tag tone={on ? 'warning' : 'outline'}>{on ? 'restore' : 'reject'}</Tag>
      </button>
    </td>
  )
}

function TracesTable({
  rows,
  rejectable,
  rejected,
  onReject,
  selected,
  onSelect,
}: {
  rows: Trace[]
  rejectable: boolean
  rejected: Set<string>
  onReject: (id: string) => void
  selected: string | null
  onSelect: (id: string) => void
}) {
  return (
    <div style={{ fontSize: 12 }}>
      <div className="flex gap-2 p-3">
        <Tag dot="hsl(165 85% 32%)">{`${rows.filter((r) => r.kind === 'port').length} port`}</Tag>
        <Tag dot="hsl(320 75% 45%)">{`${rows.filter((r) => r.kind === 'branch').length} branch`}</Tag>
        {rejected.size > 0 && <Tag tone="warning">{`${[...rejected].filter((id) => rows.some((r) => r.id === id)).length} rejected`}</Tag>}
      </div>
      <table className="w-full" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {rejectable && <th style={th} />}
            <th style={th}>trace</th><th style={th}>kind</th><th style={th}>terminal</th><th style={th}>segments</th><th style={th}>length px</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t) => (
            <tr key={t.id} style={{ ...rowStyle(t.id === selected), ...(rejected.has(t.id) ? REJECTED_ROW : {}) }} onClick={() => onSelect(t.id)}>
              {rejectable && <RejectCell on={rejected.has(t.id)} onToggle={() => onReject(t.id)} />}
              <td style={td} className="mono">{t.id}</td>
              <td style={td}>{t.kind}</td>
              <td style={td}>{t.terminal_type ?? '—'}</td>
              <td style={td} className="mono">{t.segments.length}</td>
              <td style={td} className="mono">{t.segments.reduce((n, g) => n + (g.length_px ?? 0), 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Class field: free typing locally, commits only non-empty values (an object always has a class). */
function ClassInput({ id, value, onCommit }: { id: string; value: string; onCommit: (cls: string) => void }) {
  const [text, setText] = useState(value)
  return (
    <input
      list="garnet-classes"
      aria-label={`Class for ${id}`}
      value={text}
      onChange={(e) => {
        setText(e.target.value)
        if (e.target.value.trim()) onCommit(e.target.value.trim())
      }}
      onBlur={() => setText(value)}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => e.stopPropagation()}
      style={{ width: 160, font: 'inherit', padding: '2px 6px', border: '1px solid var(--accent)', borderRadius: 4, background: 'var(--surface)' }}
    />
  )
}
