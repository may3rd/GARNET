import { useEffect, useMemo, useState } from 'react'
import { Button } from '@heroui/react'
import { ArrowLeft, Download, FileText, RefreshCw } from 'lucide-react'
import { Card, PageHeader, Tag } from '@/components/ui/primitives'
import { getMergedTestSystems, mergeUrl, type MergedLine, type MergedTestSystems } from '@/lib/api'
import { useRunStore } from '@/stores/runStore'

type Filter = 'all' | 'multi' | 'flagged'

const th: React.CSSProperties = { textAlign: 'left', padding: '8px 10px', fontWeight: 500, color: 'var(--muted)', position: 'sticky', top: 0, background: 'var(--surface)', whiteSpace: 'nowrap' }
const td: React.CSSProperties = { padding: '7px 10px', borderTop: '1px solid var(--separator)', verticalAlign: 'top' }
const chipBtn: React.CSSProperties = { border: 0, background: 'transparent', padding: 0, cursor: 'pointer' }

/**
 * Test systems across sheets. A line number links sheets only when its reviewed text is
 * IDENTICAL on each; near-matches are shown as "similar", never merged. Read-only: fix a line
 * on its own sheet (Gate 3) and refresh here.
 */
export function GarnetMerge() {
  const sheets = useRunStore((s) => s.sheets)
  const selectSheet = useRunStore((s) => s.selectSheet)
  const setScreen = useRunStore((s) => s.setScreen)

  // Sheets whose Stage 6 has run: completed, or parked at Gate 3.
  const ready = sheets.filter(
    (s) => s.task === 'garnet' && s.garnet && (s.garnet.status === 'completed' || (s.garnet.status === 'awaiting_gate' && s.garnet.gate === 3))
  )
  // Unticked sheets; everything else that is ready is in, including sheets that become ready later.
  const [excluded, setExcluded] = useState<Set<string>>(new Set())
  const jobIds = ready.map((s) => s.garnet!.id).filter((id) => !excluded.has(id))
  const sheetByJob = new Map(ready.map((s) => [s.garnet!.id, s]))

  const [data, setData] = useState<MergedTestSystems | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [filter, setFilter] = useState<Filter>('all')
  const [open, setOpen] = useState<string | null>(null)

  const load = async () => {
    if (jobIds.length === 0) {
      setData(null)
      return
    }
    setLoading(true)
    setError(null)
    try {
      setData(await getMergedTestSystems(jobIds))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }
  const key = jobIds.join(',')
  useEffect(() => {
    void load()
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps

  const flagged = (x: MergedLine) => x.continuation_missing || x.similar_to.length > 0 || x.conflict
  const rows = useMemo(
    () => (data?.lines ?? []).filter((x) => (filter === 'multi' ? x.sheet_count > 1 : filter === 'flagged' ? flagged(x) : true)),
    [data, filter]
  )

  const openSheet = (jobId: string) => {
    const sheet = sheetByJob.get(jobId)
    if (!sheet) return
    selectSheet(sheet.id)
    setScreen('garnetResults')
  }

  const linkStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    height: 34,
    padding: '0 12px',
    borderRadius: 'var(--r-btn)',
    fontSize: 13,
    fontWeight: 500,
    textDecoration: 'none',
  }

  return (
    <div className="flex h-full flex-col gap-4 overflow-hidden p-6">
      <PageHeader
        title="Multi-sheet test systems"
        subtitle={data ? `${data.sheets.length} sheets · ${data.summary.lines} line numbers · ${data.summary.multi_sheet} span more than one sheet` : 'Line numbers linked across sheets'}
        actions={
          <>
            {jobIds.length > 0 && (
              <>
                <a href={mergeUrl(jobIds, 'json')} download="garnet_multi_sheet_test_systems.json" style={{ ...linkStyle, background: 'var(--accent)', color: 'var(--accent-foreground)' }}>
                  <Download size={14} /> Export JSON
                </a>
                <a href={mergeUrl(jobIds, 'txt')} download="garnet_multi_sheet_test_systems.txt" style={{ ...linkStyle, color: 'inherit', boxShadow: 'inset 0 0 0 1px var(--border)' }}>
                  <FileText size={14} /> Export TXT
                </a>
              </>
            )}
            <Button variant="ghost" isIconOnly aria-label="Refresh" isDisabled={loading} onPress={() => void load()}>
              <RefreshCw size={16} />
            </Button>
            <Button variant="ghost" style={{ height: 36, borderRadius: 'var(--r-btn)' }} onPress={() => setScreen('garnet')}>
              <ArrowLeft size={16} strokeWidth={1.5} />
              Back to sheets
            </Button>
          </>
        }
      />

      <Card padding={14} className="flex flex-col gap-2">
        <div style={{ fontSize: 12, color: 'var(--muted)' }}>
          Sheets with Stage 6 results. A line number links sheets only when its reviewed text is identical; fix a line on its own sheet at Gate 3, then refresh.
        </div>
        {ready.length === 0 ? (
          <div style={{ fontSize: 13 }}>No sheet has reached Gate 3 yet.</div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {ready.map((s) => {
              const id = s.garnet!.id
              const on = !excluded.has(id)
              const final = s.garnet!.status === 'completed'
              return (
                <label key={id} className="inline-flex cursor-pointer items-center gap-2" style={{ fontSize: 13, padding: '4px 10px', borderRadius: 8, boxShadow: `inset 0 0 0 1px ${on ? 'var(--accent)' : 'var(--border)'}` }}>
                  <input
                    type="checkbox"
                    checked={on}
                    onChange={() =>
                      setExcluded((p) => {
                        const n = new Set(p)
                        if (n.has(id)) n.delete(id)
                        else n.add(id)
                        return n
                      })
                    }
                  />
                  {s.label}
                  {!final && <Tag tone="warning">Gate 3 open</Tag>}
                </label>
              )
            })}
          </div>
        )}
        {data && data.skipped_sheets.length > 0 && (
          <div style={{ fontSize: 12, color: 'hsl(0 70% 45%)' }}>
            {`Skipped: ${data.skipped_sheets.map((s) => `${s.sheet || s.job_id} (${s.reason})`).join(', ')}`}
          </div>
        )}
      </Card>

      {error && <Card padding={14} style={{ color: 'var(--danger)' }}>{error}</Card>}

      {data && (
        <div className="flex flex-wrap items-center gap-2">
          {(
            [
              ['all', `All · ${data.summary.lines}`],
              ['multi', `Multi-sheet · ${data.summary.multi_sheet}`],
              ['flagged', `Needs a look · ${data.lines.filter(flagged).length}`],
            ] as [Filter, string][]
          ).map(([k, label]) => (
            <button key={k} type="button" aria-pressed={filter === k} onClick={() => setFilter(k)} style={chipBtn}>
              <Tag tone={filter === k ? 'accent' : 'outline'}>{label}</Tag>
            </button>
          ))}
          <span style={{ fontSize: 12, color: 'var(--muted)', marginLeft: 8 }}>
            {`${data.summary.continuation_missing} continue off-sheet with no match · ${data.summary.similar} similar-text · ${data.summary.conflict} with a conflict`}
          </span>
        </div>
      )}

      {data && (
        <Card padding={0} className="min-h-0 flex-1 overflow-auto">
          <table className="w-full" style={{ borderCollapse: 'collapse', fontSize: 12.5 }}>
            <thead>
              <tr>
                <th style={th}>line number</th>
                <th style={th}>sheets</th>
                <th style={th}>sections</th>
                <th style={th}>equipment</th>
                <th style={th}>instr.</th>
                <th style={th}>inline</th>
                <th style={th}>flags</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((x) => {
                const isOpen = open === x.line_number
                return [
                  <tr
                    key={x.line_number}
                    onClick={() => setOpen(isOpen ? null : x.line_number)}
                    style={{ cursor: 'pointer', background: isOpen ? 'rgba(59,130,246,.08)' : undefined }}
                  >
                    <td style={td} className="mono">
                      <span style={x.no_text ? { color: 'var(--muted)' } : undefined}>{x.line_number}</span>
                    </td>
                    <td style={td}>
                      <span className="flex flex-wrap gap-1" onClick={(e) => e.stopPropagation()}>
                        {x.sheets.map((e) => (
                          <button key={e.job_id} type="button" title="Open this sheet's results" onClick={() => openSheet(e.job_id)} style={chipBtn}>
                            <Tag tone={x.sheet_count > 1 ? 'accent' : 'outline'}>{e.sheet}</Tag>
                          </button>
                        ))}
                      </span>
                    </td>
                    <td style={td} className="mono">{x.totals.pipe_sections}</td>
                    <td style={td} className="mono">{x.totals.equipment.join(', ') || '—'}</td>
                    <td style={td} className="mono">{x.totals.instruments}</td>
                    <td style={td} className="mono">{x.totals.inline_objects}</td>
                    <td style={td}>
                      <span className="flex flex-wrap gap-1">
                        {x.continuation_missing && (
                          <span title="Ends at an off-page connector, but no other selected sheet carries this line number">
                            <Tag tone="warning">continuation not found</Tag>
                          </span>
                        )}
                        {x.similar_to.length > 0 && (
                          <span title={`Similar text, NOT linked: ${x.similar_to.join(' | ')}`}>
                            <Tag tone="warning">similar text</Tag>
                          </span>
                        )}
                        {x.conflict && <Tag tone="warning">conflict</Tag>}
                      </span>
                    </td>
                  </tr>,
                  isOpen && (
                    <tr key={`${x.line_number}:detail`}>
                      <td colSpan={7} style={{ ...td, background: 'var(--surface-2, rgba(0,0,0,.02))' }}>
                        {x.similar_to.length > 0 && (
                          <div style={{ marginBottom: 8, color: 'hsl(30 90% 35%)' }}>
                            {`Similar text on other sheets, not linked: ${x.similar_to.join('  |  ')}. If it is the same line, correct the text at Gate 3 on that sheet.`}
                          </div>
                        )}
                        <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
                          {x.sheets.map((e) => (
                            <div key={e.job_id} className="flex flex-col gap-1" style={{ padding: 10, borderRadius: 8, boxShadow: 'inset 0 0 0 1px var(--border)' }}>
                              <div className="flex items-center gap-2" style={{ fontWeight: 600 }}>
                                {e.sheet}
                                {e.conflict && <Tag tone="warning">conflict on this sheet</Tag>}
                                <button type="button" onClick={() => openSheet(e.job_id)} style={{ ...chipBtn, marginLeft: 'auto' }}>
                                  <Tag>open →</Tag>
                                </button>
                              </div>
                              <Detail label="sections" value={e.trace_ids.join(', ')} />
                              <Detail label="equipment" value={e.equipment.join(', ')} />
                              <Detail label="instruments" value={e.instruments.join(', ')} />
                              <Detail label="inline" value={Object.entries(e.inline_counts).map(([c, n]) => `${c} ×${n}`).join(', ')} />
                              <Detail label="boundaries" value={Object.entries(e.boundaries).map(([c, n]) => `${c} ×${n}`).join(', ')} />
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ),
                ]
              })}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2" style={{ fontSize: 12 }}>
      <span style={{ width: 78, flexShrink: 0, color: 'var(--muted)' }}>{label}</span>
      <span className="mono" style={{ wordBreak: 'break-word' }}>{value || '—'}</span>
    </div>
  )
}
