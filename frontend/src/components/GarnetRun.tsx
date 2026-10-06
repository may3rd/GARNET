import { Button } from '@heroui/react'
import { Play } from 'lucide-react'
import { Card, PageHeader, SectionHeader, Tag, Toggle } from '@/components/ui/primitives'
import type { GarnetInputs, GarnetStepStatus } from '@/lib/api'
import { useRunStore, type Sheet } from '@/stores/runStore'

const STEP_TONE: Record<GarnetStepStatus, 'success' | 'warning' | 'accent' | 'outline'> = {
  completed: 'success',
  failed: 'warning',
  running: 'accent',
  skipped: 'outline',
  pending: 'outline',
}

/** Which uploaded JSON replaces which ML step. Line numbers have no ML step at all. */
const INPUTS: { key: keyof GarnetInputs; label: string; replaces: string }[] = [
  { key: 'objects', label: 'Objects JSON', replaces: 'skips YOLO detection' },
  { key: 'equipment', label: 'Equipment JSON', replaces: 'skips the equipment LLM' },
  { key: 'lineNumbers', label: 'Line numbers JSON', replaces: 'no script exists — traces stay unlabelled without it' },
]

/** garnet/ chain: stage fixtures, optional ML, trace, review gate, Stage 6. */
export function GarnetRun() {
  const sheets = useRunStore((s) => s.sheets).filter((s) => s.task === 'garnet')
  const config = useRunStore((s) => s.config)
  const setConfig = useRunStore((s) => s.setConfig)
  const setScreen = useRunStore((s) => s.setScreen)

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-6">
      <PageHeader
        title="GARNET trace"
        subtitle={`${sheets.length} sheet${sheets.length === 1 ? '' : 's'} · POST /api/garnet/jobs`}
        actions={
          <Button variant="ghost" style={{ height: 36, borderRadius: 'var(--r-btn)' }} onPress={() => setScreen('sheets')}>
            Back to sheets
          </Button>
        }
      />
      <Card padding={20}>
        <Toggle
          label="Review gate calls the vision model"
          description="Off passes --no-llm: the residual is kept, nothing leaves this machine."
          checked={config.reviewLlm}
          onChange={(reviewLlm) => setConfig({ reviewLlm })}
        />
      </Card>
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
  const setGarnetInput = useRunStore((s) => s.setGarnetInput)
  const runGarnetFor = useRunStore((s) => s.runGarnetFor)
  const job = sheet.garnet
  const busy = sheet.progress !== null
  const overlays = job?.artifacts.filter((a) => a.name.startsWith('out/') && a.name.endsWith('.png')) ?? []
  const jsons = job?.artifacts.filter((a) => a.name.startsWith('out/') && a.name.endsWith('.json')) ?? []

  return (
    <Card padding={20} className="flex flex-col gap-3.5">
      <SectionHeader
        title={sheet.label}
        description={job ? `job ${job.id.slice(0, 8)}… · ${job.status}` : 'Not started'}
        actions={
          <Button
            variant="primary"
            isDisabled={busy}
            style={{ height: 32, borderRadius: 'var(--r-btn)' }}
            onPress={() => void runGarnetFor(sheet.id)}
          >
            <Play size={14} strokeWidth={1.8} />
            {job ? 'Re-run' : 'Run'}
          </Button>
        }
      />

      <div className="flex flex-wrap gap-4">
        {INPUTS.map(({ key, label, replaces }) => (
          <label key={key} className="flex min-w-[220px] flex-1 flex-col gap-1" style={{ fontSize: 12 }}>
            <span style={{ fontWeight: 500 }}>{label}</span>
            <input
              type="file"
              accept=".json,application/json"
              disabled={busy}
              onChange={(e) => setGarnetInput(sheet.id, key, e.target.files?.[0])}
            />
            <span style={{ color: 'var(--muted)' }}>
              {sheet.garnetInputs[key] ? `using ${sheet.garnetInputs[key]!.name}` : replaces}
            </span>
          </label>
        ))}
      </div>

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
