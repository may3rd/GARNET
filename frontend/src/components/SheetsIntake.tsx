import { useRef, useState } from 'react'
import { Button, Card, Chip, Spinner } from '@heroui/react'
import { Play, Trash2, Upload, X } from 'lucide-react'
import { useRunStore, type TaskKind } from '@/stores/runStore'
import type { OcrRoute } from '@/types'

/**
 * A native select styled with the design tokens. HeroUI's Select is a popover
 * compound — too heavy for a table cell that toggles a few fixed values, and
 * one popover per row per column in a scrolling table is asking for trouble.
 */
function TokenSelect<T extends string>({
  label,
  value,
  options,
  disabled,
  onChange,
}: {
  label: string
  value: T
  options: { key: T; label: string }[]
  disabled?: boolean
  onChange: (value: T) => void
}) {
  return (
    <select
      aria-label={label}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value as T)}
      className="h-8 px-2 text-[13px]"
      style={{
        borderRadius: 'var(--r-field)',
        background: 'var(--field-background)',
        color: 'var(--foreground)',
        boxShadow: 'inset 0 0 0 1px var(--border)',
        border: 0,
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {options.map((o) => (
        <option key={o.key} value={o.key}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

const OCR_ROUTES: { key: OcrRoute; label: string }[] = [
  { key: 'ocrmac', label: 'ocrmac' },
  { key: 'easyocr', label: 'easyocr' },
  { key: 'paddleocr', label: 'paddleocr' },
  { key: 'gemini', label: 'gemini' },
]

const TASKS: { key: TaskKind; label: string }[] = [
  { key: 'extraction', label: 'Extraction' },
  { key: 'detection', label: 'Detection' },
]

function statusChip(sheet: ReturnType<typeof useRunStore.getState>['sheets'][number]) {
  if (sheet.error) {
    return (
      <Chip style={{ background: 'var(--danger-soft)', color: 'var(--danger-soft-fg)' }}>
        {sheet.error.length > 40 ? 'Failed' : sheet.error}
      </Chip>
    )
  }
  if (sheet.progress) {
    return (
      <Chip style={{ background: 'var(--accent-soft)', color: 'var(--accent-soft-fg)' }}>
        {sheet.progress.step} · {sheet.progress.percent}%
      </Chip>
    )
  }
  if (sheet.job?.status === 'completed') {
    return (
      <Chip style={{ background: 'var(--success-soft)', color: 'var(--success-soft-fg)' }}>
        Gate 1 open
      </Chip>
    )
  }
  return <Chip>Queued</Chip>
}

export function SheetsIntake() {
  const sheets = useRunStore((s) => s.sheets)
  const isExtracting = useRunStore((s) => s.isExtracting)
  const intakeError = useRunStore((s) => s.intakeError)
  const addFiles = useRunStore((s) => s.addFiles)
  const removeSheet = useRunStore((s) => s.removeSheet)
  const clearSheets = useRunStore((s) => s.clearSheets)
  const setSheetTask = useRunStore((s) => s.setSheetTask)
  const setSheetOcrRoute = useRunStore((s) => s.setSheetOcrRoute)
  const startRun = useRunStore((s) => s.startRun)

  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [starting, setStarting] = useState(false)

  const staged = sheets.filter((s) => !s.jobId).length
  const anyRunning = sheets.some((s) => s.progress !== null)

  const onStart = async () => {
    setStarting(true)
    try {
      await startRun()
    } finally {
      setStarting(false)
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* Page header */}
      <div
        className="flex shrink-0 items-center gap-5 px-6 py-4"
        style={{ borderBottom: '1px solid var(--separator)' }}
      >
        <div className="min-w-0">
          <div className="text-base font-semibold">New run</div>
          <div className="text-xs" style={{ color: 'var(--muted)' }}>
            {sheets.length === 0
              ? 'No sheets staged yet'
              : `${sheets.length} sheet${sheets.length === 1 ? '' : 's'} staged · ${staged} not started`}
          </div>
        </div>
        <div className="flex-1" />
        <Button variant="ghost" isDisabled={sheets.length === 0} onPress={clearSheets}>
          <Trash2 size={16} strokeWidth={1.5} />
          Clear
        </Button>
        <Button
          variant="primary"
          isDisabled={staged === 0 || starting || anyRunning}
          onPress={onStart}
        >
          {starting || anyRunning ? <Spinner size="sm" /> : <Play size={16} strokeWidth={1.5} />}
          {staged === 0 ? 'Start jobs' : `Start ${staged} job${staged === 1 ? '' : 's'}`}
        </Button>
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-6">
        {/* Dropzone */}
        <div
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              inputRef.current?.click()
            }
          }}
          onDragOver={(e) => e.preventDefault()}
          onDragEnter={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={(e) => {
            e.preventDefault()
            setDragging(false)
          }}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            if (e.dataTransfer.files.length) void addFiles(Array.from(e.dataTransfer.files))
          }}
          className="flex cursor-pointer flex-col items-center justify-center gap-3 px-6 py-10 text-center"
          style={{
            borderRadius: 'var(--r-card)',
            border: `2px dashed ${dragging ? 'var(--accent)' : 'var(--border)'}`,
            background: dragging ? 'var(--accent-soft)' : 'var(--surface)',
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.png,.jpg,.jpeg,.webp"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.length) void addFiles(Array.from(e.target.files))
              e.target.value = ''
            }}
          />
          <div
            className="flex h-12 w-12 items-center justify-center"
            style={{ borderRadius: 14, background: 'var(--accent-soft)', color: 'var(--accent-soft-fg)' }}
          >
            {isExtracting ? <Spinner size="sm" /> : <Upload size={22} strokeWidth={1.5} />}
          </div>
          <div>
            <div className="text-[15px] font-medium">
              {isExtracting ? 'Splitting PDF into sheets…' : 'Drop P&ID sheets, or browse'}
            </div>
            <div className="mt-1 text-xs" style={{ color: 'var(--muted)' }}>
              PDF, PNG, JPG · multi-page PDFs are split into one sheet per page
            </div>
          </div>
          <Button variant="secondary">Browse files</Button>
        </div>

        {intakeError && (
          <div
            className="px-4 py-3 text-sm"
            style={{
              borderRadius: 'var(--r-field)',
              background: 'var(--danger-soft)',
              color: 'var(--danger-soft-fg)',
            }}
          >
            {intakeError}
          </div>
        )}

        {/* Staged sheets */}
        {sheets.length > 0 && (
          <Card className="flex flex-col gap-3 p-4" style={{ borderRadius: 'var(--r-card)' }}>
            <div>
              <div className="text-sm font-medium">Staged sheets</div>
              <div className="text-xs" style={{ color: 'var(--muted)' }}>
                One sheet = one pipeline job. Sheets stay independent through all four gates;
                cross-sheet connectivity is resolved last, on the Merge screen.
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full" style={{ borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    {['Sheet', 'Resolution', 'Task', 'OCR route', 'Status', ''].map((h) => (
                      <th
                        key={h}
                        className="px-3 pb-2 text-left text-[11px] font-semibold uppercase tracking-wide"
                        style={{ color: 'var(--muted)' }}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sheets.map((sheet) => (
                    <tr key={sheet.id} style={{ borderTop: '1px solid var(--separator)' }}>
                      <td className="px-3 py-2.5">
                        <div className="flex items-center gap-2.5">
                          <img
                            src={sheet.previewUrl}
                            alt=""
                            className="h-9 w-9 shrink-0 object-cover"
                            style={{ borderRadius: 8, outline: '1px solid var(--border)' }}
                          />
                          <span className="text-[13px] font-medium">{sheet.label}</span>
                        </div>
                      </td>
                      <td className="mono px-3 py-2.5 text-[12px]" style={{ color: 'var(--muted)' }}>
                        {sheet.size ? `${sheet.size.width} × ${sheet.size.height}` : '—'}
                      </td>
                      <td className="px-3 py-2.5">
                        <TokenSelect
                          label={`Task for ${sheet.label}`}
                          value={sheet.task}
                          options={TASKS}
                          disabled={Boolean(sheet.jobId)}
                          onChange={(task) => setSheetTask(sheet.id, task)}
                        />
                      </td>
                      <td className="px-3 py-2.5">
                        <TokenSelect
                          label={`OCR route for ${sheet.label}`}
                          value={sheet.ocrRoute}
                          options={OCR_ROUTES}
                          disabled={Boolean(sheet.jobId) || sheet.task === 'detection'}
                          onChange={(route) => setSheetOcrRoute(sheet.id, route)}
                        />
                      </td>
                      <td className="px-3 py-2.5">{statusChip(sheet)}</td>
                      <td className="px-3 py-2.5">
                        <Button
                          variant="ghost"
                          aria-label={`Remove ${sheet.label}`}
                          isDisabled={Boolean(sheet.progress)}
                          onPress={() => removeSheet(sheet.id)}
                        >
                          <X size={16} strokeWidth={1.5} />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>
    </div>
  )
}
