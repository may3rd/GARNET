import { AppShell } from '@/components/AppShell'
import { SheetsIntake } from '@/components/SheetsIntake'
import { TaskFork } from '@/components/TaskFork'
import { GarnetRun } from '@/components/GarnetRun'
import { GarnetResults } from '@/components/GarnetResults'
import { GarnetMerge } from '@/components/GarnetMerge'
import { DetectionResults } from '@/components/DetectionResults'
import { ExportsView, NotBuilt, ReviewQueue, RunMonitor } from '@/components/screens'
import { useRunStore } from '@/stores/runStore'

/** Screen order follows the canvas; AppShell derives its own chrome from it. */
export default function App() {
  const screen = useRunStore((s) => s.screen)
  // Merge serves the garnet/ path when there are GARNET sheets; the backend pipeline merge is not built.
  const hasGarnet = useRunStore((s) => s.sheets.some((x) => x.task === 'garnet'))

  return (
    <AppShell>
      {screen === 'sheets' && <SheetsIntake />}
      {screen === 'task' && <TaskFork />}
      {screen === 'run' && <RunMonitor />}
      {screen === 'garnet' && <GarnetRun />}
      {screen === 'garnetResults' && <GarnetResults />}
      {screen === 'review' && <ReviewQueue />}
      {screen === 'exports' && <ExportsView />}
      {screen === 'detection' && <DetectionResults />}
      {screen === 'merge' && hasGarnet && <GarnetMerge />}
      {screen === 'merge' && !hasGarnet && (
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
