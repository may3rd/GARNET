import { useRef, useState } from 'react'
import { Button, Spinner } from '@heroui/react'
import { AlertTriangle, Check, Info, Upload, X } from 'lucide-react'
import { Card, SectionHeader, Tag } from '@/components/ui/primitives'
import { importAiObjects, type AiImportResult } from '@/lib/api'

/**
 * Imports externally produced equipment / line-number JSON into a job.
 *
 * Always previews first: the source raster is often framed differently from the
 * job's image, in which case the backend fits an offset from matching
 * line-number text. That fit has to be shown — with its residuals — before
 * anything is written, because a bad fit silently misplaces every box.
 */
export function AiImportDialog({
  jobId,
  onClose,
  onImported,
}: {
  jobId: string
  onClose: () => void
  onImported: () => void
}) {
  const [equipmentFile, setEquipmentFile] = useState<File | null>(null)
  const [lineNumberFile, setLineNumberFile] = useState<File | null>(null)
  const [align, setAlign] = useState(true)
  const [preview, setPreview] = useState<AiImportResult | null>(null)
  const [busy, setBusy] = useState<'preview' | 'apply' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const equipmentInput = useRef<HTMLInputElement>(null)
  const lineNumberInput = useRef<HTMLInputElement>(null)

  const run = async (mode: 'preview' | 'apply') => {
    if (!equipmentFile && !lineNumberFile) return
    setBusy(mode)
    setError(null)
    try {
      const result = await importAiObjects(
        jobId,
        { equipment: equipmentFile ?? undefined, lineNumbers: lineNumberFile ?? undefined },
        { mode, align }
      )
      if (mode === 'preview') setPreview(result)
      else onImported()
    } catch (err) {
      setError(err instanceof Error ? err.message : `Could not ${mode} the import`)
    } finally {
      setBusy(null)
    }
  }

  const skipped = preview?.report.filter((r) => r.status === 'skipped') ?? []
  const notes = preview?.report.filter((r) => r.status === 'info') ?? []
  const fit = preview?.fit

  return (
    <div
      className="fixed inset-0 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,.45)', zIndex: 60 }}
      onMouseDown={onClose}
    >
      <Card
        className="flex max-h-[80vh] w-[560px] max-w-[92vw] flex-col gap-3 overflow-hidden"
        padding={20}
        style={{ boxShadow: 'inset 0 0 0 1px var(--border), 0 20px 48px rgba(0,0,0,.4)' }}
      >
        <div onMouseDown={(e) => e.stopPropagation()} className="flex min-h-0 flex-col gap-3">
          <div className="flex items-start gap-3">
            <div className="min-w-0 flex-1">
              <SectionHeader
                title="Import extracted JSON"
                description="Equipment boxes with nozzle ports, and line numbers with their text."
              />
            </div>
            <button
              type="button"
              aria-label="Close"
              onClick={onClose}
              className="flex items-center justify-center"
              style={{
                width: 28, height: 28, border: 0, background: 'transparent',
                borderRadius: 'var(--r-btn)', color: 'var(--muted)', cursor: 'pointer',
              }}
            >
              <X size={16} strokeWidth={1.8} />
            </button>
          </div>

          <FilePicker
            label="Equipment bounding boxes"
            file={equipmentFile}
            inputRef={equipmentInput}
            onPick={(file) => {
              setEquipmentFile(file)
              setPreview(null)
            }}
          />
          <FilePicker
            label="Line number boxes"
            file={lineNumberFile}
            inputRef={lineNumberInput}
            onPick={(file) => {
              setLineNumberFile(file)
              setPreview(null)
            }}
          />

          <label className="flex items-center gap-2" style={{ fontSize: 12.5, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={align}
              onChange={(e) => {
                setAlign(e.target.checked)
                setPreview(null)
              }}
            />
            Align to this sheet if the source image was framed differently
          </label>

          {error && (
            <div
              className="flex items-start gap-2"
              style={{
                padding: '8px 10px', borderRadius: 'var(--r-btn)',
                background: 'var(--danger-soft)', color: 'var(--danger-soft-fg)', fontSize: 12.5,
              }}
            >
              <AlertTriangle size={14} strokeWidth={2} style={{ flexShrink: 0, marginTop: 1 }} />
              <span>{error}</span>
            </div>
          )}

          {preview && (
            <div className="flex min-h-0 flex-col gap-2.5 overflow-y-auto">
              <div className="flex flex-wrap items-center gap-1.5">
                <Tag tone="success">{preview.counts.equipment} equipment</Tag>
                <Tag tone="accent">{preview.counts.ports} ports</Tag>
                <Tag tone="accent">{preview.counts.line_numbers} line numbers</Tag>
                {preview.counts.skipped > 0 && (
                  <Tag tone="warning">{preview.counts.skipped} skipped</Tag>
                )}
              </div>

              {fit && (
                <div
                  style={{
                    padding: '8px 10px', borderRadius: 'var(--r-btn)',
                    background: 'var(--warning-soft)', color: 'var(--warning-soft-fg)', fontSize: 12.5,
                  }}
                >
                  <div style={{ fontWeight: 600 }}>Aligned to this sheet</div>
                  <div className="mono" style={{ marginTop: 2 }}>
                    offset ({preview.transform.dx.toFixed(1)}, {preview.transform.dy.toFixed(1)}) px
                    {' · '}scale {preview.transform.scale.toFixed(3)}
                  </div>
                  <div style={{ marginTop: 2 }}>
                    {fit.pairs_used} of {fit.pairs_matched} matched line numbers used · typical error{' '}
                    {fit.residual_median_x.toFixed(1)}/{fit.residual_median_y.toFixed(1)} px
                  </div>
                </div>
              )}

              {skipped.length > 0 && (
                <ReportList
                  title={`Skipped (${skipped.length})`}
                  rows={skipped.map((r) => ({ key: r.key, detail: r.reason }))}
                  tone="var(--warning)"
                />
              )}
              {notes.length > 0 && (
                <ReportList
                  title={`Flagged by the extraction (${notes.length})`}
                  rows={notes.map((r) => ({ key: r.key, detail: r.reason }))}
                  tone="var(--muted)"
                />
              )}
            </div>
          )}

          <div className="flex shrink-0 items-center justify-end gap-2">
            <Button variant="ghost" style={{ height: 32, borderRadius: 'var(--r-btn)' }} onPress={onClose}>
              Cancel
            </Button>
            <Button
              variant="secondary"
              isDisabled={(!equipmentFile && !lineNumberFile) || busy !== null}
              style={{ height: 32, borderRadius: 'var(--r-btn)' }}
              onPress={() => void run('preview')}
            >
              {busy === 'preview' ? <Spinner size="sm" /> : 'Preview'}
            </Button>
            <Button
              variant="primary"
              isDisabled={!preview || busy !== null}
              style={{ height: 32, borderRadius: 'var(--r-btn)' }}
              onPress={() => void run('apply')}
            >
              {busy === 'apply' ? <Spinner size="sm" /> : <Check size={14} strokeWidth={2} />}
              Apply
            </Button>
          </div>
        </div>
      </Card>
    </div>
  )
}

function FilePicker({
  label,
  file,
  inputRef,
  onPick,
}: {
  label: string
  file: File | null
  inputRef: React.RefObject<HTMLInputElement | null>
  onPick: (file: File | null) => void
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        ref={inputRef}
        type="file"
        accept="application/json,.json"
        style={{ display: 'none' }}
        onChange={(e) => onPick(e.target.files?.[0] ?? null)}
      />
      <Button
        variant="secondary"
        style={{ height: 30, borderRadius: 'var(--r-btn)', fontSize: 12.5 }}
        onPress={() => inputRef.current?.click()}
      >
        <Upload size={13} strokeWidth={2} />
        {label}
      </Button>
      <span className="mono truncate" style={{ flex: 1, fontSize: 11.5, color: 'var(--muted)' }}>
        {file ? file.name : 'none selected'}
      </span>
    </div>
  )
}

function ReportList({
  title,
  rows,
  tone,
}: {
  title: string
  rows: { key: string; detail: string }[]
  tone: string
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5" style={{ fontSize: 12, fontWeight: 500 }}>
        <Info size={12} strokeWidth={2} style={{ color: tone }} />
        {title}
      </div>
      <div style={{ marginTop: 4 }}>
        {rows.map((row, index) => (
          <div
            key={`${row.key}-${index}`}
            style={{
              padding: '4px 0',
              borderBottom: '1px solid color-mix(in oklab, var(--separator) 50%, transparent)',
              fontSize: 11.5,
            }}
          >
            <span className="mono">{row.key}</span>
            {row.detail && <span style={{ color: 'var(--muted)' }}> — {row.detail}</span>}
          </div>
        ))}
      </div>
    </div>
  )
}
