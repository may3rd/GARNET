import { Button } from '@heroui/react'
import { useEffect, useRef, useState } from 'react'
import { CheckCircle2, Container, FolderOpen, Layers, Hash, History, Pause, Play, RotateCcw, Save, Shapes, Square, Upload, X } from 'lucide-react'
import { Card, PageHeader, SectionHeader, SelectField, Tag, Toggle } from '@/components/ui/primitives'
import type { GarnetGate, GarnetInputs, GarnetStepStatus } from '@/lib/api'
import { useRunStore, type Sheet } from '@/stores/runStore'

const STEP_TONE: Record<GarnetStepStatus, 'success' | 'warning' | 'accent' | 'outline'> = {
  completed: 'success',
  failed: 'warning',
  running: 'accent',
  skipped: 'outline',
  pending: 'outline',
}

/** The three inputs, and what happens when each is missing. */
const INPUTS: { key: keyof GarnetInputs; label: string; missing: string; icon: React.ReactNode }[] = [
  { key: 'objects', label: 'Objects', missing: 'YOLO will detect symbols', icon: <Shapes size={18} /> },
  { key: 'equipment', label: 'Equipment', missing: 'Vision LLM will find equipment', icon: <Container size={18} /> },
  { key: 'lineNumbers', label: 'Line numbers', missing: 'No source — traces stay unlabelled', icon: <Hash size={18} /> },
]

/**
 * Which slot a JSON file belongs in, from its CONTENT first (the three formats are distinct), then
 * its filename. Content wins because exported files are often renamed.
 */
function classify(name: string, payload: unknown): keyof GarnetInputs | null {
  const first = (payload as { objects?: Record<string, unknown>[] })?.objects?.[0]
  if (first) {
    if ('Equipment_type' in first || 'Bounding_box_px' in first) return 'equipment'
    if (String(first.Object ?? '').toLowerCase() === 'line number') return 'lineNumbers'
    if ('class_name' in first && 'bbox' in first) return 'objects'
  }
  const n = name.toLowerCase()
  if (n.includes('line_number')) return 'lineNumbers'
  if (n.includes('equipment')) return 'equipment'
  if (n.includes('objects')) return 'objects'
  return null
}

type Parsed = { count: number } | { error: string }

async function parse(file: File): Promise<{ payload: unknown; info: Parsed }> {
  try {
    const payload = JSON.parse(await file.text())
    const objs = (payload as { objects?: unknown })?.objects
    if (!Array.isArray(objs)) return { payload, info: { error: "no 'objects' list" } }
    return { payload, info: { count: objs.length } }
  } catch {
    return { payload: null, info: { error: 'not valid JSON' } }
  }
}

/** Drop zone + three slots. Drop all three files at once; each lands in its slot by content. */
function JsonInputs({ sheet, disabled }: { sheet: Sheet; disabled: boolean }) {
  const setGarnetInput = useRunStore((s) => s.setGarnetInput)
  const [info, setInfo] = useState<Partial<Record<keyof GarnetInputs, Parsed>>>({})
  const [over, setOver] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const picker = useRef<HTMLInputElement>(null)

  // Re-read attached files (e.g. after navigating back) so counts survive remounts.
  useEffect(() => {
    INPUTS.forEach(({ key }) => {
      const f = sheet.garnetInputs[key]
      if (f && !info[key]) void parse(f).then(({ info: i }) => setInfo((x) => ({ ...x, [key]: i })))
    })
  }, [sheet.garnetInputs]) // eslint-disable-line react-hooks/exhaustive-deps

  const accept = async (files: File[]) => {
    const skipped: string[] = []
    for (const f of files) {
      const { payload, info: i } = await parse(f)
      const slot = classify(f.name, payload)
      if (!slot) {
        skipped.push(`${f.name} (${'error' in i ? i.error : 'unrecognised format'})`)
        continue
      }
      setGarnetInput(sheet.id, slot, f)
      setInfo((x) => ({ ...x, [slot]: i }))
    }
    setNotice(skipped.length ? `Not used: ${skipped.join(', ')}` : null)
  }

  return (
    <div className="flex flex-col gap-3">
      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        onClick={() => !disabled && picker.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !disabled && picker.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          if (!disabled) void accept([...e.dataTransfer.files])
        }}
        className="flex items-center justify-center gap-3"
        style={{
          minHeight: 72,
          padding: 16,
          borderRadius: 12,
          border: `2px dashed ${over ? 'var(--accent)' : 'var(--border)'}`,
          background: over ? 'color-mix(in srgb, var(--accent) 8%, transparent)' : 'var(--surface-2, transparent)',
          cursor: disabled ? 'not-allowed' : 'pointer',
          opacity: disabled ? 0.5 : 1,
          transition: 'border-color .15s, background .15s',
        }}
      >
        <Upload size={22} style={{ color: over ? 'var(--accent)' : 'var(--muted)' }} />
        <div>
          <div style={{ fontSize: 14, fontWeight: 600 }}>Drop JSON files here, or click to browse</div>
          <div style={{ fontSize: 12, color: 'var(--muted)' }}>
            Objects, equipment and line-number files, together or one at a time. Each is recognised by its content.
          </div>
        </div>
        <input
          ref={picker}
          type="file"
          multiple
          accept=".json,application/json"
          hidden
          onChange={(e) => {
            void accept([...(e.target.files ?? [])])
            e.target.value = ''
          }}
        />
      </div>

      <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
        {INPUTS.map(({ key, label, missing, icon }) => {
          const file = sheet.garnetInputs[key]
          const i = info[key]
          const bad = i && 'error' in i
          return (
            <div
              key={key}
              className="flex items-center gap-3"
              style={{
                padding: '10px 12px',
                borderRadius: 10,
                boxShadow: `inset 0 0 0 ${file ? 1.5 : 1}px ${bad ? 'hsl(0 75% 50%)' : file ? 'hsl(140 60% 40%)' : 'var(--border)'}`,
                background: file && !bad ? 'color-mix(in srgb, hsl(140 60% 40%) 7%, transparent)' : undefined,
              }}
            >
              <span
                className="flex shrink-0 items-center justify-center"
                style={{
                  width: 34,
                  height: 34,
                  borderRadius: 8,
                  color: file && !bad ? 'hsl(140 60% 35%)' : 'var(--muted)',
                  background: 'var(--surface)',
                  boxShadow: 'inset 0 0 0 1px var(--border)',
                }}
              >
                {file && !bad ? <CheckCircle2 size={18} /> : icon}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2" style={{ fontSize: 13, fontWeight: 600 }}>
                  {label}
                  {file && i && !bad && <Tag tone="success">{`${(i as { count: number }).count} objects`}</Tag>}
                  {bad && <Tag tone="warning">{(i as { error: string }).error}</Tag>}
                </div>
                <div className="truncate" style={{ fontSize: 12, color: 'var(--muted)' }} title={file?.name}>
                  {file ? file.name : missing}
                </div>
              </div>
              {file && !disabled && (
                <button
                  type="button"
                  aria-label={`Remove ${label} file`}
                  onClick={() => {
                    setGarnetInput(sheet.id, key, undefined)
                    setInfo((x) => ({ ...x, [key]: undefined }))
                  }}
                  style={{ border: 0, background: 'transparent', cursor: 'pointer', color: 'var(--muted)', padding: 4 }}
                >
                  <X size={16} />
                </button>
              )}
            </div>
          )
        })}
      </div>
      {notice && <div style={{ fontSize: 12, color: 'hsl(0 70% 45%)' }}>{notice}</div>}
    </div>
  )
}

/** garnet/ chain: stage fixtures, optional ML, trace, review gate, Stage 6. */
export function GarnetRun() {
  const sheets = useRunStore((s) => s.sheets).filter((s) => s.task === 'garnet')
  const config = useRunStore((s) => s.config)
  const setConfig = useRunStore((s) => s.setConfig)
  const weightFiles = useRunStore((s) => s.weightFiles)
  const loadWeightFiles = useRunStore((s) => s.loadWeightFiles)
  useEffect(() => {
    void loadWeightFiles()
  }, [loadWeightFiles])
  const setScreen = useRunStore((s) => s.setScreen)
  const loadGarnetBundle = useRunStore((s) => s.loadGarnetBundle)
  const runGarnetFor = useRunStore((s) => s.runGarnetFor)
  // Never-started sheets; Run all starts them together (each with whatever JSON it has).
  const unstarted = sheets.filter((s) => !s.garnet && s.progress === null)
  const mergeable = sheets.filter((s) => s.garnet?.status === 'completed' || (s.garnet?.status === 'awaiting_gate' && s.garnet.gate === 3)).length
  const [loadError, setLoadError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const loader = useRef<HTMLInputElement>(null)
  const onLoad = async (file: File | undefined) => {
    if (!file) return
    setLoading(true)
    setLoadError(null)
    try {
      await loadGarnetBundle(file)
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-6">
      <PageHeader
        title="GARNET trace"
        subtitle={`${sheets.length} sheet${sheets.length === 1 ? '' : 's'} · POST /api/garnet/jobs`}
        actions={
          <>
            {unstarted.length > 1 && (
              <Button
                variant="primary"
                style={{ height: 36, borderRadius: 'var(--r-btn)' }}
                onPress={() => {
                  const missing = unstarted.filter((s) => !s.garnetInputs.objects || !s.garnetInputs.equipment).length
                  if (missing && !window.confirm(`${missing} of ${unstarted.length} sheets are missing objects or equipment JSON; those will run YOLO / the equipment LLM. Run all anyway?`)) return
                  unstarted.forEach((s) => void runGarnetFor(s.id))
                }}
              >
                <Play size={15} /> {`Run all (${unstarted.length})`}
              </Button>
            )}
            {sheets.length > 1 && (
              <Button
                variant="secondary"
                isDisabled={mergeable === 0}
                style={{ height: 36, borderRadius: 'var(--r-btn)' }}
                onPress={() => setScreen('merge')}
              >
                <Layers size={16} strokeWidth={1.5} />
                {`Merge sheets (${mergeable})`}
              </Button>
            )}
            <Button
              variant="secondary"
              isDisabled={loading}
              style={{ height: 36, borderRadius: 'var(--r-btn)' }}
              onPress={() => loader.current?.click()}
            >
              <FolderOpen size={16} strokeWidth={1.5} />
              {loading ? 'Loading…' : 'Load saved result'}
            </Button>
            <input
              ref={loader}
              type="file"
              accept=".zip,application/zip"
              hidden
              onChange={(e) => {
                void onLoad(e.target.files?.[0])
                e.target.value = ''
              }}
            />
            <Button variant="ghost" style={{ height: 36, borderRadius: 'var(--r-btn)' }} onPress={() => setScreen('sheets')}>
              Back to sheets
            </Button>
          </>
        }
      />
      {loadError && <Card padding={14} style={{ color: 'var(--danger)', fontSize: 13 }}>{`Could not load: ${loadError}`}</Card>}
      <Card padding={20} className="flex flex-col gap-3">
        {/* Same store field and options as the extraction picker on the Task screen. */}
        <SelectField
          label="Detection weights (unused for sheets with an objects JSON)"
          value={config.weightFile}
          options={[{ key: '', label: 'Server default' }, ...weightFiles.map((w) => ({ key: w, label: w.replace(/^.*\//, '') }))]}
          onChange={(weightFile) => setConfig({ weightFile })}
        />
        <Toggle
          label="Review gate calls the vision model"
          description="Off passes --no-llm: the residual is kept, nothing leaves this machine."
          checked={config.reviewLlm}
          onChange={(reviewLlm) => setConfig({ reviewLlm })}
        />
        <Toggle
          label="OCR instrument tags and off-page connectors"
          description="EasyOCR on instrument bubbles and connectors that have no text. Connector text (e.g. 25-0003) is what links sheets. Existing text is never overwritten."
          checked={config.ocrInstruments}
          onChange={(ocrInstruments) => setConfig({ ocrInstruments })}
        />
        <Toggle
          label="Pause at every gate"
          description="Gate 1 objects, Gate 2 traces, Gate 3 line numbers. Off runs straight through."
          checked={config.pauseAtEveryGate}
          onChange={(pauseAtEveryGate) => setConfig({ pauseAtEveryGate })}
        />
      </Card>
      {sheets.length > 1 && <BatchJsonDrop sheets={sheets} />}

      {sheets.length === 0 ? (
        <Card padding={20}>
          <SectionHeader title="No GARNET sheets" description="Pick the GARNET trace task on the Task screen." />
        </Card>
      ) : (
        sheets.map((sheet) => <SheetCard key={sheet.id} sheet={sheet} />)
      )}
    </div>
  )
}

function SheetCard({ sheet }: { sheet: Sheet }) {
  const runGarnetFor = useRunStore((s) => s.runGarnetFor)
  const control = useRunStore((s) => s.controlGarnetJob)
  const selectSheet = useRunStore((s) => s.selectSheet)
  const setScreen = useRunStore((s) => s.setScreen)
  const job = sheet.garnet
  const busy = sheet.progress !== null
  const live = job?.status === 'queued' || job?.status === 'running'
  const resting = job?.status === 'awaiting_gate' || job?.status === 'paused' || job?.status === 'completed' || job?.status === 'failed'
  const overlays = job?.artifacts.filter((a) => a.name.startsWith('out/') && a.name.endsWith('.png')) ?? []
  const jsons = job?.artifacts.filter((a) => a.name.startsWith('out/') && a.name.endsWith('.json')) ?? []

  return (
    <Card padding={20} className="flex flex-col gap-3.5">
      <SectionHeader
        title={sheet.label}
        description={
          job
            ? `job ${job.id.slice(0, 8)}… · ${job.status}`
            : 'Waiting for input — attach any JSON you have, then Run. Missing objects / equipment JSON is produced by YOLO / the LLM.'
        }
        actions={
          <div className="flex gap-2">
          {live && (
            <>
              <Button variant="secondary" style={{ height: 32, borderRadius: 'var(--r-btn)' }} onPress={() => void control(sheet.id, 'pause')}>
                <Pause size={14} /> Pause
              </Button>
              <Button
                variant="danger"
                style={{ height: 32, borderRadius: 'var(--r-btn)' }}
                onPress={() => window.confirm('Stop this run now? The running step is killed; you can Restart or Reopen afterwards.') && void control(sheet.id, 'stop')}
              >
                <Square size={14} /> Stop
              </Button>
            </>
          )}
          {job?.status === 'paused' && !busy && (
            <Button variant="primary" style={{ height: 32, borderRadius: 'var(--r-btn)' }} onPress={() => void control(sheet.id, 'resume')}>
              <Play size={14} /> Resume
            </Button>
          )}
          {resting && !busy && (
            <Button
              variant="secondary"
              style={{ height: 32, borderRadius: 'var(--r-btn)' }}
              onPress={() =>
                window.confirm('Restart this run on the same inputs? Confirmed Gate 1 edits are kept; later outputs and gate decisions are archived.') &&
                void control(sheet.id, 'restart')
              }
            >
              <RotateCcw size={14} /> Restart
            </Button>
          )}
          {job?.status === 'awaiting_gate' && !busy && (
            <Button
              variant="primary"
              style={{ height: 32, borderRadius: 'var(--r-btn)' }}
              onPress={() => {
                selectSheet(sheet.id)
                setScreen('garnetResults')
              }}
            >
              Review Gate {job.gate}
            </Button>
          )}
          {resting && !busy && (
            // The whole job folder as a zip: reload it later with "Load saved result".
            <a
              href={`/api/garnet/jobs/${job.id}/bundle`}
              download
              title="Save this result (inputs, outputs, gate decisions, history) as a .zip"
              className="inline-flex items-center gap-1.5"
              style={{ height: 32, padding: '0 12px', borderRadius: 'var(--r-btn)', boxShadow: 'inset 0 0 0 1px var(--border)', color: 'inherit', fontSize: 13, fontWeight: 500, textDecoration: 'none' }}
            >
              <Save size={14} /> Save
            </a>
          )}
          {job?.status === 'completed' && !busy && (
            <Button
              variant="secondary"
              style={{ height: 32, borderRadius: 'var(--r-btn)' }}
              onPress={() => {
                selectSheet(sheet.id)
                setScreen('garnetResults')
              }}
            >
              Open results
            </Button>
          )}
          <Button
            variant={job?.status === 'awaiting_gate' ? 'secondary' : 'primary'}
            isDisabled={busy}
            style={{ height: 32, borderRadius: 'var(--r-btn)' }}
            onPress={() => void runGarnetFor(sheet.id)}
          >
            <Play size={14} strokeWidth={1.8} />
            {job ? 'Re-run' : 'Run'}
          </Button>
          </div>
        }
      />

      <JsonInputs sheet={sheet} disabled={busy} />

      {job && <GateTimeline sheet={sheet} busy={busy} />}

      {sheet.error && <div style={{ fontSize: 13, color: 'var(--danger)' }}>{sheet.error}</div>}

      {job && (
        <div className="flex flex-col gap-1.5">
          {job.steps.map((step) => (
            <details key={step.name}>
              <summary className="flex cursor-pointer items-center gap-2" style={{ fontSize: 13 }}>
                <Tag tone={STEP_TONE[step.status]}>{step.status}</Tag>
                <span className="mono">{step.name}</span>
                {step.note && <span style={{ color: 'var(--muted)' }}>{step.note}</span>}
              </summary>
              {step.log && (
                <pre className="mono" style={{ fontSize: 11, maxHeight: 240, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
                  {step.log}
                </pre>
              )}
            </details>
          ))}
        </div>
      )}

      {overlays.length > 0 && (
        <div className="flex flex-wrap gap-3">
          {overlays.map((a) => (
            <a key={a.url} href={a.url} target="_blank" rel="noreferrer" style={{ fontSize: 11 }}>
              <img src={a.url} alt={a.name} loading="lazy" style={{ width: 220, borderRadius: 6, outline: '1px solid var(--border)' }} />
              <div className="mono truncate" style={{ width: 220 }}>{a.name.slice(4)}</div>
            </a>
          ))}
        </div>
      )}
      {jsons.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {jsons.map((a) => (
            <a key={a.url} href={a.url} target="_blank" rel="noreferrer" className="mono" style={{ fontSize: 12 }}>
              {a.name.slice(4)}
            </a>
          ))}
        </div>
      )}
    </Card>
  )
}

const GATE_NAMES: Record<GarnetGate, string> = { 1: 'Objects', 2: 'Traces', 3: 'Line numbers' }

/**
 * Gate 1 -> Gate 2 -> Gate 3 -> Done. A confirmed gate can be reopened: the job parks there
 * again and everything produced after it is archived by the server (history/), not deleted.
 */
function GateTimeline({ sheet, busy }: { sheet: Sheet; busy: boolean }) {
  const reopen = useRunStore((s) => s.reopenGarnetGate)
  const selectSheet = useRunStore((s) => s.selectSheet)
  const setScreen = useRunStore((s) => s.setScreen)
  const job = sheet.garnet!
  // A failed job can still go back to a gate it already passed.
  const resting = job.status === 'awaiting_gate' || job.status === 'paused' || job.status === 'completed' || job.status === 'failed'
  const done = job.status === 'completed'

  const go = async (gate: GarnetGate) => {
    const later = gate === 1 ? 'the trace, Stage 6 and every review after detection' : gate === 2 ? 'Stage 6 and the Gate 3 review' : 'the Gate 3 review and text edits'
    if (!window.confirm(`Go back to Gate ${gate} · ${GATE_NAMES[gate]}?\n\nThis discards ${later}. Nothing is deleted: it is archived in the job's history folder.`)) return
    // On failure stay here: the run screen is where sheet.error is shown.
    if (!(await reopen(sheet.id, gate))) return
    selectSheet(sheet.id)
    setScreen('garnetResults')
  }

  const node = (gate: GarnetGate) => {
    const confirmed = job.gates_confirmed.includes(gate)
    const current = job.status === 'awaiting_gate' && job.gate === gate
    // Passed = the run is beyond this gate, reviewed or not (a run with gates off passes all three).
    // Same rule as the API's reopen check: every step up to the gate's boundary completed or skipped.
    const boundary = job.steps.findIndex((st) => st.gate_after === gate)
    const reached = boundary >= 0 && job.steps.slice(0, boundary + 1).every((st) => st.status === 'completed' || st.status === 'skipped')
    const passed = confirmed || done || reached || (job.status === 'awaiting_gate' && (job.gate ?? 0) > gate)
    const at = [...job.history].reverse().find((h) => h.event === 'confirm' && h.gate === gate)?.at
    const tone = current ? 'var(--accent)' : confirmed ? 'hsl(140 60% 38%)' : passed ? 'hsl(38 90% 45%)' : 'var(--border)'
    return (
      <div key={gate} className="flex min-w-[150px] flex-1 flex-col gap-1.5" style={{ padding: '10px 12px', borderRadius: 10, boxShadow: `inset 0 0 0 ${current ? 2 : 1}px ${tone}` }}>
        <div className="flex items-center gap-2" style={{ fontSize: 13, fontWeight: 600 }}>
          <span className="flex items-center justify-center" style={{ width: 20, height: 20, borderRadius: 999, fontSize: 11, color: 'white', background: current || passed ? tone : 'var(--muted)' }}>
            {confirmed && !current ? '✓' : gate}
          </span>
          {`Gate ${gate} · ${GATE_NAMES[gate]}`}
        </div>
        <div style={{ fontSize: 11.5, color: 'var(--muted)' }}>
          {current
            ? 'Waiting for review'
            : confirmed
              ? `Confirmed ${at ? new Date(at).toLocaleString() : ''}`
              : passed
                ? 'Passed without review'
                : 'Not reached'}
        </div>
        {passed && !current && resting && !busy && (
          <button type="button" onClick={() => void go(gate)} style={{ alignSelf: 'flex-start', border: 0, background: 'transparent', padding: 0, cursor: 'pointer' }}>
            <Tag>↶ Reopen</Tag>
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-stretch gap-2">
        {([1, 2, 3] as GarnetGate[]).map(node)}
        <div
          className="flex min-w-[110px] flex-col justify-center gap-1"
          style={{ padding: '10px 12px', borderRadius: 10, boxShadow: `inset 0 0 0 1px ${done ? 'hsl(140 60% 38%)' : 'var(--border)'}` }}
        >
          <div style={{ fontSize: 13, fontWeight: 600, color: done ? 'hsl(140 60% 35%)' : undefined }}>{done ? '✓ Done' : 'Done'}</div>
          <div style={{ fontSize: 11.5, color: 'var(--muted)' }}>{done ? 'Test systems ready' : '—'}</div>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        {resting && !busy && (job.gates_confirmed.length > 0 || done) && (
          <button
            type="button"
            onClick={() => void go(1)}
            className="inline-flex items-center gap-1.5"
            style={{ border: 0, background: 'transparent', padding: 0, cursor: 'pointer', fontSize: 12, color: 'hsl(0 70% 45%)' }}
          >
            <RotateCcw size={13} /> Reset to start
          </button>
        )}
        {job.history.length > 0 && (
          <details style={{ fontSize: 12 }}>
            <summary className="inline-flex cursor-pointer items-center gap-1.5" style={{ color: 'var(--muted)' }}>
              <History size={13} /> History ({job.history.length})
            </summary>
            <ol className="mt-1.5 flex flex-col gap-1" style={{ paddingLeft: 18 }}>
              {job.history.map((h, i) => (
                <li key={i} className="mono" style={{ listStyle: 'decimal' }}>
                  {new Date(h.at).toLocaleString()} · {h.event === 'confirm' ? `confirmed Gate ${h.gate}` : `reopened Gate ${h.gate}`}
                  {h.event === 'confirm' && Object.keys(h.changes).length > 0 && (
                    <span style={{ color: 'var(--muted)' }}>
                      {' '}({Object.entries(h.changes).map(([k, n]) => `${n} ${k}`).join(', ')})
                    </span>
                  )}
                  {h.event === 'reopen' && h.archive && <span style={{ color: 'var(--muted)' }}> → {h.archive}</span>}
                </li>
              ))}
            </ol>
          </details>
        )}
      </div>
    </div>
  )
}

const stemOf = (name: string) => name.replace(/\.[^.]+$/, '')

/**
 * One drop for many sheets: each file goes to the sheet whose raster name is the LONGEST prefix
 * of the file name (so `..-0004A_objects.json` goes to 0004A, not 0004), and into its slot by
 * content. Files that match no sheet, or no slot, are listed, not guessed.
 */
function BatchJsonDrop({ sheets }: { sheets: Sheet[] }) {
  const setGarnetInput = useRunStore((s) => s.setGarnetInput)
  const [over, setOver] = useState(false)
  const [report, setReport] = useState<string | null>(null)
  const picker = useRef<HTMLInputElement>(null)
  const open = sheets.filter((s) => !s.garnet && s.progress === null)

  const accept = async (files: File[]) => {
    let placed = 0
    const misses: string[] = []
    for (const f of files) {
      const target = open
        .filter((s) => f.name.startsWith(stemOf(s.file.name)))
        .sort((a, b) => stemOf(b.file.name).length - stemOf(a.file.name).length)[0]
      if (!target) {
        misses.push(`${f.name} (no sheet named like it)`)
        continue
      }
      const { payload, info } = await parse(f)
      const slot = classify(f.name, payload)
      if (!slot || 'error' in info) {
        misses.push(`${f.name} (${'error' in info ? info.error : 'unrecognised format'})`)
        continue
      }
      setGarnetInput(target.id, slot, f)
      placed += 1
    }
    setReport(`${placed} file(s) attached${misses.length ? ` · not used: ${misses.join(', ')}` : ''}`)
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => picker.current?.click()}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && picker.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        void accept([...e.dataTransfer.files])
      }}
      className="flex flex-wrap items-center gap-3"
      style={{
        padding: '12px 16px',
        borderRadius: 12,
        border: `2px dashed ${over ? 'var(--accent)' : 'var(--border)'}`,
        background: over ? 'color-mix(in srgb, var(--accent) 8%, transparent)' : 'var(--surface)',
        cursor: 'pointer',
      }}
    >
      <Layers size={20} style={{ color: over ? 'var(--accent)' : 'var(--muted)' }} />
      <div className="min-w-0 flex-1">
        <div style={{ fontSize: 14, fontWeight: 600 }}>Drop JSON for all sheets</div>
        <div style={{ fontSize: 12, color: 'var(--muted)' }}>
          {report ?? 'Files are matched to sheets by name (e.g. 14780-…-0002A_objects.json → 14780-…-0002A.jpg) and to slots by content.'}
        </div>
      </div>
      <input
        ref={picker}
        type="file"
        multiple
        accept=".json,application/json"
        hidden
        onChange={(e) => {
          void accept([...(e.target.files ?? [])])
          e.target.value = ''
        }}
      />
    </div>
  )
}
