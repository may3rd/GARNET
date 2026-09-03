import { AppShell } from '@/components/AppShell'
import { SheetsIntake } from '@/components/SheetsIntake'
import { TaskFork } from '@/components/TaskFork'
import { useRunStore } from '@/stores/runStore'

/**
 * Screen order follows the canvas: 1 Sheets intake, 2 Task fork, then the run
 * and its four gates. The rail highlight per screen matches the artboards.
 */
export default function App() {
  const screen = useRunStore((s) => s.screen)

  if (screen === 'task') {
    return (
      <AppShell active="detection" breadcrumb={['GARNET', 'New run', 'Task']}>
        <TaskFork />
      </AppShell>
    )
  }

  if (screen === 'run') {
    return (
      <AppShell active="extraction" breadcrumb={['GARNET', 'New run', 'Run']}>
        <RunPlaceholder />
      </AppShell>
    )
  }

  return (
    <AppShell active="sheets" breadcrumb={['Projects', 'GARNET', 'New run']}>
      <SheetsIntake />
    </AppShell>
  )
}

function RunPlaceholder() {
  const sheets = useRunStore((s) => s.sheets)
  const setScreen = useRunStore((s) => s.setScreen)

  return (
    <div className="flex flex-col gap-4 p-6">
      <div style={{ fontSize: 22, fontWeight: 600 }}>Run monitor</div>
      <div style={{ fontSize: 14, color: 'var(--muted)' }}>
        Not built yet — next screen from the canvas (ExtractionRun). Jobs below are live.
      </div>
      <div className="flex flex-col gap-2">
        {sheets.map((s) => (
          <div
            key={s.id}
            className="flex items-center gap-3"
            style={{ padding: 12, background: 'var(--surface)', borderRadius: 'var(--r-field)' }}
          >
            <span style={{ fontSize: 13, fontWeight: 500 }}>{s.label}</span>
            <span className="mono" style={{ fontSize: 12, color: 'var(--muted)' }}>
              {s.error ?? s.progress?.step ?? s.job?.status ?? 'queued'}
              {s.progress ? ` · ${s.progress.percent}%` : ''}
            </span>
          </div>
        ))}
      </div>
      <button
        type="button"
        onClick={() => setScreen('sheets')}
        style={{
          alignSelf: 'flex-start',
          padding: '8px 16px',
          borderRadius: 'var(--r-btn)',
          border: 0,
          background: 'var(--surface-secondary)',
          color: 'var(--foreground)',
          cursor: 'pointer',
        }}
      >
        Back to sheets
      </button>
    </div>
  )
}
