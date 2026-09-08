import { Button, Spinner } from '@heroui/react'
import { AlertTriangle, ArrowRight, Check, Download, FileWarning, Play, X } from 'lucide-react'
import { ArtifactsPanel } from '@/components/ArtifactsPanel'
import { Card, PageHeader, SectionHeader, Separator, Tag } from '@/components/ui/primitives'
import { Gate1Objects } from '@/components/Gate1Objects'
import { Gate2Traces } from '@/components/Gate2Traces'
import { Gate3LineAssociation } from '@/components/Gate3LineAssociation'
import { Gate4GraphQA } from '@/components/Gate4GraphQA'
import { GATES, isRunComplete, type GateId } from '@/lib/gates'
import type { Screen } from '@/lib/nav'
import type { PipelineStageManifest } from '@/types'
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

/** "stage5b_pipe_trace" -> "Pipe trace". Strips the leading stage-number token. */
function stageTitle(name: string): string {
  const rest = name.replace(/^stage\d+[a-z]?_/, '').replaceAll('_', ' ')
  return rest.charAt(0).toUpperCase() + rest.slice(1)
}

/** Which gate (if any) this manifest row's stage number is the boundary for, and whether the run has already moved past it. */
function gateBadgeForStage(
  stages: PipelineStageManifest[],
  stageNum: number
): { gate: GateId; cleared: boolean } | null {
  const gate = ([1, 2, 3, 4] as GateId[]).find((g) => GATES[g].stopAfter === stageNum)
  if (!gate) return null
  if (!stages.some((s) => s.num === stageNum && s.status === 'completed')) return null
  const cleared = stages.some((s) => (s.num ?? 0) > stageNum && s.status !== 'pending')
  return { gate, cleared }
}

const STAGE_ICON_BG: Record<PipelineStageManifest['status'], string> = {
  completed: 'var(--success-soft)',
  started: 'var(--accent-soft)',
  running: 'var(--accent-soft)',
  failed: 'var(--danger-soft)',
  stale: 'var(--warning-soft)',
  pending: 'var(--surface-tertiary)',
}

function StageIcon({ status, num }: { status: PipelineStageManifest['status']; num: number }) {
  const style: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 26,
    height: 26,
    borderRadius: 999,
    flexShrink: 0,
    background: STAGE_ICON_BG[status],
  }
  if (status === 'completed') return <span style={{ ...style, color: 'var(--success)' }}><Check size={14} strokeWidth={2.4} /></span>
  if (status === 'started' || status === 'running') return <span style={style}><Spinner size="sm" /></span>
  if (status === 'failed') return <span style={{ ...style, color: 'var(--danger)' }}><X size={14} strokeWidth={2.4} /></span>
  if (status === 'stale') return <span style={{ ...style, color: 'var(--warning)' }}><AlertTriangle size={13} strokeWidth={2} /></span>
  return (
    <span className="mono" style={{ ...style, color: 'var(--muted)', fontSize: 10.5 }}>
      {num}
    </span>
  )
}

/** 4 · ExtractionRun — live stage state for every sheet in the run. */
export function RunMonitor() {
  const allSheets = useRunStore((s) => s.sheets)
  const setScreen = useRunStore((s) => s.setScreen)
  const selectSheet = useRunStore((s) => s.selectSheet)
  const selectedSheetId = useRunStore((s) => s.selectedSheetId)
  const gateFor = useRunStore((s) => s.gateFor)

  // Extraction only: detection sheets have no stage manifest and no gates,
  // so listing them here would show every one as "not started" forever.
  const sheets = allSheets.filter((s) => s.task === 'extraction')
  const detectionCount = allSheets.length - sheets.length
  const anyRunning = sheets.some((s) => s.progress !== null)

  const focused =
    sheets.find((s) => s.id === selectedSheetId) ?? sheets.find((s) => s.progress !== null) ?? sheets[0]
  const artifacts = focused?.job?.artifacts ?? []

  return (
    <div className="flex h-full flex-col gap-4 p-6">
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
        <div className="flex min-h-0 flex-1 gap-4">
          {/* Stage timeline — the sheet in focus. Every field here comes straight
              off the real stage manifest; nothing is simulated. */}
          <Card className="flex min-h-0 flex-[1.3] flex-col gap-3 overflow-hidden" padding={20}>
            {focused && (
              <>
                <SectionHeader
                  title="Stage timeline"
                  description={`${focused.label}${focused.jobId ? ` · job ${focused.jobId.slice(0, 8)}…` : ''}`}
                />
                <div className="min-h-0 flex-1 overflow-y-auto">
                  {focused.stages.length === 0 ? (
                    <div style={{ fontSize: 13, color: 'var(--muted)', padding: '8px 0' }}>
                      {focused.error ?? focused.progress?.step ?? 'Not started yet.'}
                    </div>
                  ) : (
                    <div className="flex flex-col">
                      {focused.stages.map((stage, i) => {
                        // Several manifest rows can share one stage number (stage 4 has
                        // three); the gate boundary is the last row at that number.
                        const isLastAtNum = focused.stages
                          .slice(i + 1)
                          .every((s) => s.num !== stage.num)
                        const badge = isLastAtNum ? gateBadgeForStage(focused.stages, stage.num ?? 0) : null
                        const isLast = i === focused.stages.length - 1
                        return (
                          <div key={`${stage.name}-${i}`} className="relative flex items-start gap-3" style={{ padding: '4px 0' }}>
                            {!isLast && (
                              <div
                                style={{
                                  position: 'absolute',
                                  left: 12,
                                  top: 28,
                                  bottom: -10,
                                  width: 2,
                                  background: 'var(--separator)',
                                }}
                              />
                            )}
                            <div style={{ position: 'relative', zIndex: 1 }}>
                              <StageIcon status={stage.status} num={stage.num ?? i + 1} />
                            </div>
                            <div className="min-w-0 flex-1" style={{ paddingTop: 2 }}>
                              <div className="flex items-center gap-2">
                                <span style={{ fontSize: 13, fontWeight: 500 }}>{stageTitle(stage.name)}</span>
                                <span className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>
                                  stage {stage.num}
                                </span>
                                {badge && (
                                  <Tag tone={badge.cleared ? 'success' : 'warning'}>
                                    Gate {badge.gate} {badge.cleared ? 'cleared' : 'next'}
                                  </Tag>
                                )}
                              </div>
                              <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 1 }}>
                                {stage.status === 'failed' && stage.error
                                  ? stage.error
                                  : stage.status === 'stale'
                                    ? `stale${stage.stale_source_artifact ? ` — invalidated by ${stage.stale_source_artifact}` : ''}`
                                    : `${stage.name}${stage.artifacts?.length ? ` · ${stage.artifacts.length} artifact${stage.artifacts.length === 1 ? '' : 's'}` : ''}`}
                              </div>
                            </div>
                            {stage.duration_sec != null && (
                              <span
                                className="mono"
                                style={{
                                  flexShrink: 0,
                                  fontSize: 12,
                                  color: 'var(--muted)',
                                  boxShadow: 'inset 0 0 0 1px var(--border)',
                                  borderRadius: 'var(--r-chip)',
                                  padding: '2px 8px',
                                }}
                              >
                                {stage.duration_sec.toFixed(1)}s
                              </span>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              </>
            )}
          </Card>

          {/* Right column: per-sheet queue + the focused sheet's real artifacts. */}
          <div className="flex min-h-0 flex-1 flex-col gap-3.5" style={{ maxWidth: 360 }}>
            <Card className="flex flex-col gap-2.5" padding={16}>
              <SectionHeader title={`Run · ${sheets.length} job${sheets.length === 1 ? '' : 's'}`} description="Each sheet runs and is reviewed on its own" />
              <div className="flex flex-col gap-1.5" style={{ maxHeight: 260, overflowY: 'auto' }}>
                {sheets.map((sheet) => {
                  const gate = gateFor(sheet.id)
                  const complete = isRunComplete(sheet.stages)
                  const isFocused = sheet.id === focused?.id
                  return (
                    <button
                      key={sheet.id}
                      type="button"
                      onClick={() => selectSheet(sheet.id)}
                      className="flex flex-col gap-1.5 text-left"
                      style={{
                        padding: '9px 10px',
                        borderRadius: 14,
                        background: isFocused ? 'var(--surface-secondary)' : 'transparent',
                        border: 0,
                        cursor: 'pointer',
                        font: 'inherit',
                        color: 'inherit',
                      }}
                    >
                      <div className="flex items-center gap-2">
                        <span className="truncate" style={{ flex: 1, fontSize: 13, fontWeight: 500 }}>
                          {sheet.label}
                        </span>
                        {sheet.error ? (
                          <Tag tone="danger">failed</Tag>
                        ) : complete ? (
                          <Tag tone="success">complete</Tag>
                        ) : gate ? (
                          <Tag tone="warning">Gate {gate}</Tag>
                        ) : sheet.progress ? (
                          <Tag tone="accent">running</Tag>
                        ) : (
                          <Tag tone="neutral">queued</Tag>
                        )}
                      </div>
                      <div className="mono" style={{ fontSize: 11.5, color: 'var(--muted)' }}>
                        {sheet.error ?? sheet.progress?.step ?? (gate ? `Awaiting Gate ${gate}` : stageLine(sheet))}
                      </div>
                      <div style={{ width: '100%', height: 4, borderRadius: 999, background: 'color-mix(in oklab, var(--foreground) 12%, transparent)', overflow: 'hidden' }}>
                        <div
                          style={{
                            width: `${sheet.progress?.percent ?? (complete ? 100 : 0)}%`,
                            height: '100%',
                            borderRadius: 999,
                            background: sheet.error ? 'var(--danger)' : gate ? 'var(--warning)' : 'var(--accent)',
                          }}
                        />
                      </div>
                    </button>
                  )
                })}
              </div>
            </Card>

            <ArtifactsPanel jobId={focused?.jobId} artifacts={artifacts} />
          </div>
        </div>
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

  const focused = waiting.find((w) => w.sheet.id === selectedSheetId)
  // All four gates now have real review screens.
  if (focused && focused.gate === 1) {
    return <Gate1Objects sheet={focused.sheet} onBack={() => selectSheet(null)} />
  }
  if (focused && focused.gate === 2) {
    return <Gate2Traces sheet={focused.sheet} onBack={() => selectSheet(null)} />
  }
  if (focused && focused.gate === 3) {
    return <Gate3LineAssociation sheet={focused.sheet} onBack={() => selectSheet(null)} />
  }
  if (focused && focused.gate === 4) {
    return <Gate4GraphQA sheet={focused.sheet} onBack={() => selectSheet(null)} />
  }

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
                    {gate === 1
                      ? 'Review objects'
                      : gate === 2
                        ? 'Review traces'
                        : gate === 3
                          ? 'Review associations'
                          : 'Review graph QA'}
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
