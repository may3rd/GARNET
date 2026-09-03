import { Button } from '@heroui/react'
import { ArrowRight, Download, FileWarning, Play } from 'lucide-react'
import { Card, PageHeader, SectionHeader, Separator, Tag } from '@/components/ui/primitives'
import { GATES, isRunComplete, type GateId } from '@/lib/gates'
import type { Screen } from '@/lib/nav'
import { useRunStore, type Sheet } from '@/stores/runStore'

/**
 * Destinations for rail entries whose artboard has not been built yet. Better
 * than a dead icon: navigation works, and the screen says what it needs.
 */
export function NotBuilt({
  title,
  artboard,
  needs,
  goTo,
}: {
  title: string
  artboard: string
  needs: string
  goTo?: { label: string; screen: Screen }
}) {
  const setScreen = useRunStore((s) => s.setScreen)
  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <PageHeader title={title} subtitle={`Canvas artboard: ${artboard}`} />
      <Card padding={20} className="flex flex-col gap-3">
        <div className="flex items-start gap-3">
          <span style={{ color: 'var(--warning)', marginTop: 2 }}>
            <FileWarning size={18} strokeWidth={1.6} />
          </span>
          <div>
            <div style={{ fontSize: 14, fontWeight: 500 }}>Not built yet</div>
            <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 2 }}>{needs}</div>
          </div>
        </div>
        {goTo && (
          <>
            <Separator />
            <Button
              variant="secondary"
              style={{ alignSelf: 'flex-start', height: 32, borderRadius: 'var(--r-btn)' }}
              onPress={() => setScreen(goTo.screen)}
            >
              {goTo.label}
              <ArrowRight size={15} strokeWidth={1.5} />
            </Button>
          </>
        )}
      </Card>
    </div>
  )
}

function stageLine(sheet: Sheet) {
  const done = sheet.stages.filter((s) => s.status === 'completed').length
  const current = sheet.stages.find((s) => s.status === 'started' || s.status === 'running')
  if (current) return `${done} done · running ${current.name}`
  if (sheet.stages.length) return `${done} of ${sheet.stages.length} stages complete`
  return 'not started'
}

/** 4 · ExtractionRun — live stage state for every sheet in the run. */
export function RunMonitor() {
  const allSheets = useRunStore((s) => s.sheets)
  const setScreen = useRunStore((s) => s.setScreen)
  const selectSheet = useRunStore((s) => s.selectSheet)
  const gateFor = useRunStore((s) => s.gateFor)

  // Extraction only: detection sheets have no stage manifest and no gates,
  // so listing them here would show every one as "not started" forever.
  const sheets = allSheets.filter((s) => s.task === 'extraction')
  const detectionCount = allSheets.length - sheets.length
  const anyRunning = sheets.some((s) => s.progress !== null)

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-6">
      <PageHeader
        title="Extraction run"
        subtitle={
          sheets.length === 0
            ? 'No sheets in this run'
            : `${sheets.length} sheet${sheets.length === 1 ? '' : 's'} · ${
                anyRunning ? 'running' : 'idle'
              }`
        }
        actions={
          <Button
            variant="ghost"
            style={{ height: 36, borderRadius: 'var(--r-btn)' }}
            onPress={() => setScreen('sheets')}
          >
            Back to sheets
          </Button>
        }
      />

      {sheets.length === 0 ? (
        <Card padding={20}>
          <SectionHeader
            title="Nothing to show"
            description={
              detectionCount > 0
                ? `This run has ${detectionCount} detection sheet${
                    detectionCount === 1 ? '' : 's'
                  } and no extraction sheets — detection results live on the Detection screen.`
                : 'Stage some sheets and start a run first.'
            }
            actions={
              detectionCount > 0 ? (
                <Button
                  variant="secondary"
                  style={{ height: 32, borderRadius: 'var(--r-btn)' }}
                  onPress={() => setScreen('detection')}
                >
                  Detection results
                  <ArrowRight size={15} strokeWidth={1.5} />
                </Button>
              ) : undefined
            }
          />
        </Card>
      ) : (
        <Card className="flex flex-col gap-3.5">
          <SectionHeader
            title="Sheets"
            description="Each sheet is its own job with its own stage manifest and gate queue."
          />
          <div className="flex flex-col gap-2">
            {sheets.map((sheet) => {
              const gate = gateFor(sheet.id)
              const complete = isRunComplete(sheet.stages)
              return (
                <div
                  key={sheet.id}
                  className="flex items-center gap-3"
                  style={{
                    padding: '12px 14px',
                    borderRadius: 'var(--r-field)',
                    background: 'var(--surface-secondary)',
                  }}
                >
                  <img
                    src={sheet.previewUrl}
                    alt=""
                    className="shrink-0"
                    style={{
                      width: 30,
                      height: 30,
                      objectFit: 'cover',
                      borderRadius: 6,
                      outline: '1px solid var(--border)',
                    }}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="truncate" style={{ fontSize: 13, fontWeight: 500 }}>
                      {sheet.label}
                    </div>
                    <div className="mono" style={{ fontSize: 11.5, color: 'var(--muted)' }}>
                      {sheet.error ?? sheet.progress?.step ?? stageLine(sheet)}
                    </div>
                  </div>
                  {sheet.progress && <Tag tone="accent">{sheet.progress.percent}%</Tag>}
                  {sheet.error && <Tag tone="danger">failed</Tag>}
                  {complete && <Tag tone="success">complete</Tag>}
                  {gate && (
                    <Button
                      variant="secondary"
                      style={{ height: 30, borderRadius: 'var(--r-btn)', fontSize: 13 }}
                      onPress={() => {
                        selectSheet(sheet.id)
                        setScreen('review')
                      }}
                    >
                      Gate {gate}
                      <ArrowRight size={14} strokeWidth={1.6} />
                    </Button>
                  )}
                </div>
              )
            })}
          </div>
        </Card>
      )}
    </div>
  )
}

/**
 * 5-8 · Gate1..Gate4 — the queue across sheets. The per-gate review surfaces
 * are still to be built; this is the real gate state and the real resume call.
 */
export function ReviewQueue() {
  const sheets = useRunStore((s) => s.sheets)
  const selectedSheetId = useRunStore((s) => s.selectedSheetId)
  const selectSheet = useRunStore((s) => s.selectSheet)
  const gateFor = useRunStore((s) => s.gateFor)
  const resumeGate = useRunStore((s) => s.resumeGate)
  const setScreen = useRunStore((s) => s.setScreen)

  const waiting = sheets
    .map((sheet) => ({ sheet, gate: gateFor(sheet.id) }))
    .filter((x): x is { sheet: Sheet; gate: GateId } => x.gate !== null)

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-6">
      <PageHeader
        title="Review queue"
        subtitle={
          waiting.length === 0
            ? 'No gate is open'
            : `${waiting.length} sheet${waiting.length === 1 ? '' : 's'} waiting on a human`
        }
        actions={
          <Button
            variant="ghost"
            style={{ height: 36, borderRadius: 'var(--r-btn)' }}
            onPress={() => setScreen('run')}
          >
            Run monitor
          </Button>
        }
      />

      {waiting.length === 0 ? (
        <Card padding={20}>
          <SectionHeader
            title="Nothing waiting"
            description="A gate opens when a job comes to rest at a stage boundary that needs a decision — after stage 4, 5b, 6 or 8."
          />
        </Card>
      ) : (
        <Card className="flex flex-col gap-3.5">
          <SectionHeader
            title="Open gates"
            description="Confirming a gate resumes the job from the stage that gate feeds. Gate 4 resumes through stage 11 so exports and the overlay are regenerated."
          />
          <div className="flex flex-col gap-2">
            {waiting.map(({ sheet, gate }) => {
              const def = GATES[gate]
              const isSelected = sheet.id === selectedSheetId
              return (
                <div
                  key={sheet.id}
                  className="flex items-center gap-3"
                  style={{
                    padding: '12px 14px',
                    borderRadius: 'var(--r-field)',
                    background: 'var(--surface-secondary)',
                    boxShadow: isSelected ? 'inset 0 0 0 1.5px var(--accent)' : undefined,
                  }}
                >
                  <Tag tone="warning">Gate {gate}</Tag>
                  <div className="min-w-0 flex-1">
                    <div className="truncate" style={{ fontSize: 13, fontWeight: 500 }}>
                      {sheet.label}
                    </div>
                    <div className="mono" style={{ fontSize: 11.5, color: 'var(--muted)' }}>
                      {def.label} · {def.stageLabel} → resumes at {def.resumeStage}
                      {def.resumeStopAfter ? ` (stop_after ${def.resumeStopAfter})` : ''}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    style={{ height: 30, borderRadius: 'var(--r-btn)', fontSize: 13 }}
                    onPress={() => selectSheet(sheet.id)}
                  >
                    Focus
                  </Button>
                  <Button
                    variant="primary"
                    isDisabled={Boolean(sheet.progress)}
                    style={{ height: 30, borderRadius: 'var(--r-btn)', fontSize: 13 }}
                    onPress={() => {
                      selectSheet(sheet.id)
                      void resumeGate(sheet.id, gate)
                    }}
                  >
                    <Play size={14} strokeWidth={1.6} />
                    Confirm &amp; continue
                  </Button>
                </div>
              )
            })}
          </div>
        </Card>
      )}

      <Card padding={20}>
        <SectionHeader
          title="Per-gate review surfaces still to build"
          description="Gate 1 objects and equipment, Gate 2 traced paths, Gate 3 line association, Gate 4 graph QA — each has its own artboard with the sheet canvas and its own decision controls. Confirming from here accepts the stage as-is."
        />
      </Card>
    </div>
  )
}

/** 9 · Exports — the artifacts each job actually wrote. */
export function ExportsView() {
  const sheets = useRunStore((s) => s.sheets)
  const withJobs = sheets.filter((s) => s.job && s.job.artifacts.length > 0)

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-6">
      <PageHeader
        title="Exports"
        subtitle={
          withJobs.length === 0
            ? 'No artifacts yet'
            : `${withJobs.length} sheet${withJobs.length === 1 ? '' : 's'} with artifacts`
        }
      />

      {withJobs.length === 0 ? (
        <Card padding={20}>
          <SectionHeader
            title="Nothing written yet"
            description="Every stage writes inspectable artifacts as it completes. Start a run and they show up here."
          />
        </Card>
      ) : (
        withJobs.map((sheet) => (
          <Card key={sheet.id} className="flex flex-col gap-3">
            <SectionHeader
              title={sheet.label}
              description={`${sheet.job!.artifacts.length} artifacts`}
              actions={
                isRunComplete(sheet.stages) ? (
                  <Tag tone="success">run complete</Tag>
                ) : (
                  <Tag tone="warning">partial</Tag>
                )
              }
            />
            <div className="flex flex-wrap gap-2">
              {sheet.job!.artifacts.map((a) => (
                <a
                  key={a.name}
                  href={a.url}
                  target="_blank"
                  rel="noreferrer"
                  className="mono inline-flex items-center gap-1.5"
                  style={{
                    padding: '5px 10px',
                    borderRadius: 'var(--r-chip)',
                    fontSize: 11.5,
                    background: 'var(--surface-secondary)',
                    color: 'var(--foreground)',
                    textDecoration: 'none',
                  }}
                >
                  <Download size={12} strokeWidth={1.8} style={{ color: 'var(--muted)' }} />
                  {a.name}
                </a>
              ))}
            </div>
          </Card>
        ))
      )}
    </div>
  )
}
