import { FileImage, Trash2 } from 'lucide-react'
import { useAppStore } from '@/stores/appStore'
import { Button } from '@/components/ui/button'

export function PipelineSystemSetupView() {
  const items = useAppStore((state) => state.batch.items)
  const sheetIds = useAppStore((state) => state.pipelineSheetIds)
  const setSheetId = useAppStore((state) => state.setPipelineSheetId)
  const removeItem = useAppStore((state) => state.removeBatchItem)
  const normalized = items.map((item) => (sheetIds[item.id] || '').trim().toLocaleLowerCase().replace(/\s+/g, ' '))
  const valid = items.length >= 2 && items.length <= 50 && normalized.every(Boolean) && new Set(normalized).size === normalized.length

  return (
    <section className="flex-1 overflow-y-auto bg-[var(--bg-canvas)] p-6">
      <div className="mx-auto max-w-4xl">
        <h1 className="text-xl font-semibold">Pipeline system pages</h1>
        <p className="mt-1 text-sm text-[var(--text-secondary)]">
          Confirm the drawing or sheet ID printed on each P&amp;ID. These IDs are used to resolve cross-page connectors.
        </p>
        <div className="mt-6 space-y-3">
          {items.map((item, index) => (
            <div key={item.id} className="flex items-center gap-3 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-4">
              <FileImage className="h-5 w-5 shrink-0 text-[var(--accent)]" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">{index + 1}. {item.fileName}</div>
                <label className="mt-2 block text-xs text-[var(--text-secondary)]">
                  Drawing / sheet ID
                  <input
                    value={sheetIds[item.id] || ''}
                    onChange={(event) => setSheetId(item.id, event.target.value)}
                    className="mt-1 w-full rounded-md border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm text-[var(--text-primary)]"
                  />
                </label>
              </div>
              <Button variant="ghost" size="icon" onClick={() => removeItem(item.id)} aria-label={`Remove ${item.fileName}`}>
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          ))}
        </div>
        {!valid && (
          <div className="mt-4 rounded-lg border border-[var(--danger)]/40 bg-[var(--danger)]/5 p-3 text-xs text-[var(--danger)]">
            A system needs 2–50 pages, each with a non-empty, unique drawing/sheet ID.
          </div>
        )}
      </div>
    </section>
  )
}
