import { useRef, useState } from 'react'
import { Button } from '@heroui/react'
import { ArrowRight, Eye, FileUp, Trash2, Upload, X } from 'lucide-react'
import { Card, PageHeader, SectionHeader, Tag } from '@/components/ui/primitives'
import { useRunStore, type Sheet } from '@/stores/runStore'

/** Column widths are the artboard's: 300/130/130/120/300/110/108. */
const COLS = [
  { key: 'sheet', label: 'Sheet', w: 300 },
  { key: 'resolution', label: 'Resolution', w: 130 },
  { key: 'task', label: 'Task', w: 130 },
  { key: 'ocr', label: 'OCR route', w: 120 },
  { key: 'weights', label: 'Weights', w: 300 },
  { key: 'status', label: 'Status', w: 110 },
  { key: 'actions', label: '', w: 108 },
]

function Cell({
  width,
  children,
  align,
}: {
  width: number
  children?: React.ReactNode
  align?: 'end'
}) {
  return (
    <div
      className="flex items-center gap-2"
      style={{
        width,
        flexShrink: 0,
        padding: '12px 16px',
        boxSizing: 'border-box',
        fontSize: 14,
        justifyContent: align === 'end' ? 'flex-end' : undefined,
      }}
    >
      {children}
    </div>
  )
}

function StatusCell({ sheet }: { sheet: Sheet }) {
  if (sheet.error) return <Tag tone="danger">Failed</Tag>
  if (sheet.progress) return <Tag tone="accent">{sheet.progress.percent}%</Tag>
  if (sheet.job?.status === 'completed') return <Tag tone="warning">Gate 1</Tag>
  return <Tag dot="var(--muted)">Queued</Tag>
}

export function SheetsIntake() {
  const sheets = useRunStore((s) => s.sheets)
  const isExtracting = useRunStore((s) => s.isExtracting)
  const intakeError = useRunStore((s) => s.intakeError)
  const config = useRunStore((s) => s.config)
  const addFiles = useRunStore((s) => s.addFiles)
  const removeSheet = useRunStore((s) => s.removeSheet)
  const clearSheets = useRunStore((s) => s.clearSheets)
  const setScreen = useRunStore((s) => s.setScreen)

  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  /** Sheets that came from one PDF, for the page-picker panel. */
  const pdfGroups = new Map<string, Sheet[]>()
  sheets.forEach((s) => {
    const m = s.label.match(/^(.*\.pdf) · p\.\d+$/i)
    if (m) {
      const list = pdfGroups.get(m[1]) ?? []
      list.push(s)
      pdfGroups.set(m[1], list)
    }
  })
  const [pdfName, pdfSheets] = [...pdfGroups.entries()][0] ?? [null, []]

  const open = () => inputRef.current?.click()

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-6">
      <PageHeader
        title="New run"
        subtitle={
          sheets.length === 0
            ? 'No sheets staged yet'
            : `${sheets.length} sheet${sheets.length === 1 ? '' : 's'} staged`
        }
        actions={
          <>
            <Button
              variant="ghost"
              isDisabled={sheets.length === 0}
              style={{ height: 36, borderRadius: 'var(--r-btn)' }}
              onPress={clearSheets}
            >
              <Trash2 size={16} strokeWidth={1.5} />
              Clear
            </Button>
            <Button
              variant="primary"
              isDisabled={sheets.length === 0}
              style={{ height: 36, borderRadius: 'var(--r-btn)' }}
              onPress={() => setScreen('task')}
            >
              Choose task
              <ArrowRight size={16} strokeWidth={1.5} />
            </Button>
          </>
        }
      />

      {/* Dropzone + PDF page picker */}
      <div className="flex items-stretch gap-4">
        <div className="min-w-0" style={{ flex: 1.35 }}>
          {/*
            The drop target is a plain div, not a role="button": the "Browse
            files" button inside it is the accessible control, and nesting one
            interactive element in another gives a duplicate tab stop. Clicking
            the card is a convenience — HeroUI's Button swallows the native
            click on itself (react-aria preventDefaults the pointer events), so
            this never double-fires.
          */}
          <div
            onClick={open}
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
            className="flex h-full cursor-pointer flex-col items-center justify-center gap-2"
            style={{
              minHeight: 196,
              boxSizing: 'border-box',
              borderRadius: 'var(--r-card)',
              background: dragging ? 'var(--accent-soft)' : 'var(--surface)',
              boxShadow: 'inset 0 0 0 1.5px var(--accent)',
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
              className="flex items-center justify-center"
              style={{
                width: 44,
                height: 44,
                borderRadius: 14,
                background: 'var(--accent-soft)',
                color: 'var(--accent-soft-fg)',
              }}
            >
              {isExtracting ? <FileUp size={22} strokeWidth={1.5} /> : <Upload size={22} strokeWidth={1.5} />}
            </div>
            <div style={{ fontSize: 15, fontWeight: 500 }}>
              {isExtracting ? 'Splitting PDF into sheets…' : 'Drop P&ID sheets, or browse'}
            </div>
            <div style={{ fontSize: 13, color: 'var(--muted)' }}>
              PDF, PNG, JPG · 50 MB per file · multi-page PDFs are split into one sheet per page
            </div>
            <div style={{ height: 4 }} />
            <Button
              variant="secondary"
              style={{ height: 32, borderRadius: 'var(--r-btn)' }}
              onPress={open}
            >
              Browse files
            </Button>
          </div>
        </div>

        <div className="min-w-0 flex-1">
          <Card className="flex h-full flex-col gap-3.5">
            {pdfName ? (
              <>
                <SectionHeader
                  title={`${pdfName} · ${pdfSheets.length} page${pdfSheets.length === 1 ? '' : 's'}`}
                  description="Every page became its own job."
                  actions={<Tag tone="accent">{pdfSheets.length} staged</Tag>}
                />
                <div className="flex flex-wrap gap-3">
                  {pdfSheets.map((s, i) => (
                    <div key={s.id} className="flex flex-col items-center gap-1.5">
                      <img
                        src={s.previewUrl}
                        alt=""
                        style={{
                          width: 64,
                          height: 46,
                          objectFit: 'cover',
                          borderRadius: 6,
                          background: '#f4f4f5',
                          boxShadow: 'inset 0 0 0 2px var(--accent)',
                        }}
                      />
                      <div className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>
                        p.{i + 1}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <SectionHeader
                title="No multi-page PDF staged"
                description="Drop a PDF and each page is split into its own sheet, with its own job and its own gate queue."
              />
            )}
          </Card>
        </div>
      </div>

      {intakeError && (
        <div
          style={{
            padding: '12px 16px',
            borderRadius: 'var(--r-field)',
            background: 'var(--danger-soft)',
            color: 'var(--danger-soft-fg)',
            fontSize: 13,
          }}
        >
          {intakeError}
        </div>
      )}

      {/* Staged sheets */}
      {sheets.length > 0 && (
        <Card className="flex flex-col gap-3.5">
          <SectionHeader
            title="Staged sheets"
            description="One sheet = one pipeline job. Sheets stay independent through all four gates; cross-sheet connectivity is resolved last, on the Merge screen."
            actions={
              <Button
                variant="ghost"
                style={{ height: 32, borderRadius: 'var(--r-btn)', fontSize: 13 }}
                onPress={() => setScreen('task')}
              >
                Assign task
              </Button>
            }
          />

          <div
            style={{
              padding: 4,
              background: 'var(--surface-secondary)',
              borderRadius: 'var(--r-table)',
            }}
          >
            <div className="flex items-center">
              {COLS.map((c) => (
                <div
                  key={c.key}
                  style={{
                    width: c.w,
                    flexShrink: 0,
                    padding: '10px 16px',
                    boxSizing: 'border-box',
                    fontSize: 12,
                    fontWeight: 500,
                    color: 'var(--muted)',
                  }}
                >
                  {c.label}
                </div>
              ))}
            </div>

            <div style={{ background: 'var(--surface)', borderRadius: 16, overflow: 'hidden' }}>
              {sheets.map((sheet, i) => (
                <div
                  key={sheet.id}
                  className="flex items-center"
                  style={{
                    borderBottom:
                      i === sheets.length - 1
                        ? undefined
                        : '1px solid color-mix(in oklab, var(--separator) 50%, transparent)',
                  }}
                >
                  <Cell width={300}>
                    <img
                      src={sheet.previewUrl}
                      alt=""
                      className="shrink-0"
                      style={{
                        width: 24,
                        height: 24,
                        objectFit: 'cover',
                        borderRadius: 5,
                        outline: '1px solid var(--border)',
                      }}
                    />
                    <span className="truncate">{sheet.label}</span>
                  </Cell>
                  <Cell width={130}>
                    <span className="mono" style={{ fontSize: 13 }}>
                      {sheet.size ? `${sheet.size.width} × ${sheet.size.height}` : '—'}
                    </span>
                  </Cell>
                  <Cell width={130}>
                    <Tag tone="accent">
                      {sheet.task === 'extraction' ? 'Extraction' : 'Detection'}
                    </Tag>
                  </Cell>
                  <Cell width={120}>
                    <span style={{ fontSize: 13, color: 'var(--muted)' }}>
                      {sheet.task === 'detection' ? '—' : sheet.ocrRoute}
                    </span>
                  </Cell>
                  <Cell width={300}>
                    <span className="mono truncate" style={{ fontSize: 12, color: 'var(--muted)' }}>
                      {config.weightFile || 'server default'}
                    </span>
                  </Cell>
                  <Cell width={110}>
                    <StatusCell sheet={sheet} />
                  </Cell>
                  <Cell width={108} align="end">
                    <Button
                      variant="ghost"
                      isIconOnly
                      aria-label={`Preview ${sheet.label}`}
                      style={{ width: 30, height: 30, borderRadius: 'var(--r-btn)' }}
                      onPress={() => window.open(sheet.previewUrl, '_blank')}
                    >
                      <Eye size={15} strokeWidth={1.5} />
                    </Button>
                    <Button
                      variant="ghost"
                      isIconOnly
                      aria-label={`Remove ${sheet.label}`}
                      isDisabled={Boolean(sheet.progress)}
                      style={{ width: 30, height: 30, borderRadius: 'var(--r-btn)' }}
                      onPress={() => removeSheet(sheet.id)}
                    >
                      <X size={15} strokeWidth={1.5} />
                    </Button>
                  </Cell>
                </div>
              ))}
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
