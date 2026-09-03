import { AppShell } from '@/components/AppShell'
import { SheetsIntake } from '@/components/SheetsIntake'
import { TaskFork } from '@/components/TaskFork'
import { DetectionResults } from '@/components/DetectionResults'
import { ExportsView, NotBuilt, ReviewQueue, RunMonitor } from '@/components/screens'
import { useRunStore } from '@/stores/runStore'

/** Screen order follows the canvas; AppShell derives its own chrome from it. */
export default function App() {
  const screen = useRunStore((s) => s.screen)

  return (
    <AppShell>
      {screen === 'sheets' && <SheetsIntake />}
      {screen === 'task' && <TaskFork />}
      {screen === 'run' && <RunMonitor />}
      {screen === 'review' && <ReviewQueue />}
      {screen === 'exports' && <ExportsView />}
      {screen === 'detection' && <DetectionResults />}
      {screen === 'merge' && (
        <NotBuilt
          title="Merge"
          artboard="Exports and merge"
          needs="Needs POST /api/pipeline/merge and at least two sheets past Gate 4, so off-page connectors can be matched across sheets."
          goTo={{ label: 'See exports', screen: 'exports' }}
        />
      )}
    </AppShell>
  )
}
