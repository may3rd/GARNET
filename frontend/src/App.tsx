import { AppShell } from '@/components/AppShell'
import { SheetsIntake } from '@/components/SheetsIntake'

export default function App() {
  return (
    <AppShell active="sheets" breadcrumb={['Projects', 'GARNET', 'New run']}>
      <SheetsIntake />
    </AppShell>
  )
}
