import { Button } from '@heroui/react'
import { ArrowLeft, ArrowRight, Check, ScanSearch, Workflow } from 'lucide-react'
import {
  Card,
  IconTile,
  PageHeader,
  SectionHeader,
  SelectField,
  Separator,
  Tag,
  Toggle,
} from '@/components/ui/primitives'
import { useRunStore, type TaskKind } from '@/stores/runStore'
import type { OcrRoute } from '@/types'

const OCR_ROUTES: { key: OcrRoute; label: string }[] = [
  { key: 'ocrmac', label: 'ocrmac — Apple Vision (default)' },
  { key: 'easyocr', label: 'easyocr — local, CPU/GPU' },
  { key: 'paddleocr', label: 'paddleocr — detection only' },
  { key: 'gemini', label: 'gemini — via OpenRouter' },
]

/** Only the stop_after values the API accepts. Stage 3 does not exist. */
const STOP_AFTER_STAGES: { value: number; label: string }[] = [
  { value: 1, label: '1 · Normalise' },
  { value: 2, label: '2 · OCR' },
  { value: 4, label: '4 · Detect' },
  { value: 5, label: '5 · Pipe mask' },
  { value: 6, label: '6 · Associate' },
  { value: 7, label: '7 · Graph' },
  { value: 8, label: '8 · QA' },
  { value: 9, label: '9 · Apply review' },
  { value: 10, label: '10 · Exports' },
  { value: 11, label: '11 · Overlay' },
]

type TaskCardProps = {
  task: TaskKind
  selected: boolean
  onSelect: () => void
  icon: React.ReactNode
  title: string
  endpoint: string
  blurb: string
  bullets: string[]
  timing: string
  gates: { label: string; tone: 'success' | 'warning' }
}

function TaskCard({
  selected,
  onSelect,
  icon,
  title,
  endpoint,
  blurb,
  bullets,
  timing,
  gates,
}: TaskCardProps) {
  return (
    <div className="min-w-0 flex-1">
      <button
        type="button"
        onClick={onSelect}
        aria-pressed={selected}
        className="flex w-full flex-col gap-4 text-left"
        style={{
          height: 436,
          padding: 20,
          boxSizing: 'border-box',
          background: 'var(--surface)',
          borderRadius: 'var(--r-card)',
          border: 0,
          cursor: 'pointer',
          boxShadow: selected ? 'inset 0 0 0 1.5px var(--accent)' : 'inset 0 0 0 1px var(--border)',
        }}
      >
        <div className="flex w-full items-start gap-3">
          <IconTile>{icon}</IconTile>
          <div className="min-w-0 flex-1">
            <div style={{ fontSize: 16, fontWeight: 600 }}>{title}</div>
            <div className="mono" style={{ fontSize: 12, color: 'var(--muted)' }}>
              {endpoint}
            </div>
          </div>
          {selected ? (
            <span
              className="flex shrink-0 items-center justify-center"
              style={{
                width: 20,
                height: 20,
                borderRadius: 999,
                background: 'var(--accent)',
                color: 'var(--accent-foreground)',
              }}
            >
              <Check size={13} strokeWidth={2.6} />
            </span>
          ) : (
            <span
              className="shrink-0"
              style={{
                width: 20,
                height: 20,
                borderRadius: 999,
                boxShadow: 'inset 0 0 0 1.5px var(--border)',
              }}
            />
          )}
        </div>

        <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: '19px' }}>{blurb}</div>

        <Separator />

        <div className="flex flex-col gap-2">
          {bullets.map((b) => (
            <div key={b} className="flex items-start gap-2.5">
              <span className="flex shrink-0" style={{ color: 'var(--accent)', marginTop: 2 }}>
                <Check size={14} strokeWidth={2.2} />
              </span>
              <span style={{ fontSize: 13, lineHeight: '19px' }}>{b}</span>
            </div>
          ))}
        </div>

        <div className="flex-1" />

        <div className="flex items-stretch gap-2">
          <Tag dot="var(--muted)">{timing}</Tag>
          <Tag tone={gates.tone}>{gates.label}</Tag>
        </div>
      </button>
    </div>
  )
}

export function TaskFork() {
  const task = useRunStore((s) => s.task)
  const setTask = useRunStore((s) => s.setTask)
  const config = useRunStore((s) => s.config)
  const setConfig = useRunStore((s) => s.setConfig)
  const sheets = useRunStore((s) => s.sheets)
  const setScreen = useRunStore((s) => s.setScreen)
  const startRun = useRunStore((s) => s.startRun)

  const extractionSheets = sheets.filter((s) => s.task === 'extraction').length

  const onContinue = async () => {
    // Follow the chosen task: detection has its own results screen and never
    // goes through the run monitor, which is extraction's stage tracker.
    setScreen(task === 'detection' ? 'detection' : 'run')
    void startRun()
  }

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-6">
      <PageHeader
        title="Choose a task"
        subtitle="Set once for the run, override per sheet"
        actions={
          <>
            <Button
              variant="ghost"
              style={{ height: 36, borderRadius: 'var(--r-btn)' }}
              onPress={() => setScreen('sheets')}
            >
              <ArrowLeft size={16} strokeWidth={1.5} />
              Back
            </Button>
            <Button
              variant="primary"
              isDisabled={sheets.length === 0}
              style={{ height: 36, borderRadius: 'var(--r-btn)' }}
              onPress={onContinue}
            >
              Continue to run
              <ArrowRight size={16} strokeWidth={1.5} />
            </Button>
          </>
        }
      />

      <div className="flex items-stretch gap-4">
        <TaskCard
          task="detection"
          selected={task === 'detection'}
          onSelect={() => setTask('detection')}
          icon={<ScanSearch size={20} strokeWidth={1.5} />}
          title="Detection"
          endpoint="POST /api/detect"
          blurb="One pass of YOLOv11 + SAHI over the sheet, with optional OCR. Boxes land in an editable results canvas — no graph, no connectivity, no review gates."
          bullets={[
            'Tunable confidence, tile size and overlap ratio',
            'Add, move, retype and delete boxes on the canvas',
            'Exports to Excel, JSON and COCO',
            'Use it for symbol counts, take-offs and training-data QA',
          ]}
          timing="~40 s per sheet"
          gates={{ label: 'no gates', tone: 'success' }}
        />
        <TaskCard
          task="extraction"
          selected={task === 'extraction'}
          onSelect={() => setTask('extraction')}
          icon={<Workflow size={20} strokeWidth={1.5} />}
          title="Extraction"
          endpoint="POST /api/pipeline/jobs"
          blurb="The full pid_extractor rebuild: normalise, OCR, detect, mask, trace, associate, assemble the graph, then process exports. Four gates need a human before the graph is trusted."
          bullets={[
            '13 inspectable stages, every one writing artifacts',
            'Resume from any stage after a parameter change',
            'Gates 1–4: objects, traces, line association, graph QA',
            'Exports line list, equipment connectivity, MTO, instrument index',
          ]}
          timing="~8 min per sheet + review"
          gates={{ label: '4 human gates', tone: 'warning' }}
        />
      </div>

      {task === 'extraction' && (
        <Card padding={20} className="flex flex-col gap-3.5">
          <SectionHeader
            title="Extraction parameters"
            description={`Applied to all ${extractionSheets} extraction sheet${
              extractionSheets === 1 ? '' : 's'
            }. Override per sheet on the Sheets screen.`}
          />

          <div className="flex items-stretch gap-3.5">
            <SelectField
              label="OCR route"
              value={config.ocrRoute}
              options={OCR_ROUTES}
              onChange={(ocrRoute) => setConfig({ ocrRoute })}
            />
            <SelectField
              label="Detection weights"
              value={config.weightFile}
              options={[{ key: '', label: 'Server default' }]}
              flex={1.2}
              onChange={(weightFile) => setConfig({ weightFile })}
            />
            <SelectField
              label="Gemini match threshold"
              value={config.geminiPostprocessMatchThreshold.toFixed(2)}
              options={['0.00', '0.10', '0.25', '0.50', '0.75', '1.00'].map((v) => ({
                key: v,
                label: v,
              }))}
              hint="0–1"
              flex={0.7}
              disabled={config.ocrRoute !== 'gemini'}
              onChange={(v) => setConfig({ geminiPostprocessMatchThreshold: Number(v) })}
            />
          </div>

          <Separator />

          <div className="flex flex-col gap-2">
            <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--muted)' }}>
              Stop after stage
            </div>
            <div className="flex flex-wrap gap-2">
              {STOP_AFTER_STAGES.map((s) => {
                const selected = config.stopAfterStage === s.value
                return (
                  <button
                    key={s.value}
                    type="button"
                    aria-pressed={selected}
                    onClick={() => setConfig({ stopAfterStage: s.value })}
                    style={{ border: 0, background: 'transparent', padding: 0, cursor: 'pointer' }}
                  >
                    <Tag tone={selected ? 'accent' : 'outline'}>{s.label}</Tag>
                  </button>
                )
              })}
            </div>
          </div>

          <Separator />

          <div className="flex items-center gap-6">
            <Toggle
              label="Debug artifacts"
              description="Writes intermediate masks and overlays. Slower, large output."
              checked={config.debugArtifacts}
              onChange={(debugArtifacts) => setConfig({ debugArtifacts })}
            />
            <Toggle
              label="Pause at every gate"
              description="Off runs straight to Stage 8 and queues all gates at once."
              checked={config.pauseAtEveryGate}
              onChange={(pauseAtEveryGate) => setConfig({ pauseAtEveryGate })}
            />
          </div>
        </Card>
      )}
    </div>
  )
}
