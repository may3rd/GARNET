import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, CheckCircle2, Download, ExternalLink, Link2, Loader2, Save, X } from 'lucide-react'
import type { ConnectorOverride, ManualConnectorPair } from '@/types'
import { getPipelineSystemGraph, putPipelineSystemConnectorReview } from '@/lib/api'
import { useAppStore } from '@/stores/appStore'
import { Button } from '@/components/ui/button'

type GraphEdge = {
  id: string
  off_page_connector?: {
    connector_key?: string
    target_sheet_reference?: string
    reference_value?: string
    raw_reference_text?: string
  }
}

type SystemGraph = {
  sheets?: Array<{ sheet_id: string; graph_v1: { edges?: GraphEdge[] } }>
  cross_sheet_edges?: Array<Record<string, unknown>>
  merge_issues?: Array<{ issue_id: string; type: string; sheets_involved?: string[] }>
}

const statusLabel = (status: string) => status.replaceAll('_', ' ')

export function PipelineSystemView() {
  const system = useAppStore((state) => state.pipelineSystem)
  const refreshSystem = useAppStore((state) => state.refreshPipelineSystem)
  const openPage = useAppStore((state) => state.openPipelineSystemPage)
  const [graph, setGraph] = useState<SystemGraph | null>(null)
  const [overrides, setOverrides] = useState<Record<string, ConnectorOverride>>({})
  const [manualPairs, setManualPairs] = useState<ManualConnectorPair[]>([])
  const [leftConnector, setLeftConnector] = useState('')
  const [rightConnector, setRightConnector] = useState('')
  const [reviewer, setReviewer] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!system || ['awaiting_connector_review', 'completed', 'failed'].includes(system.status)) return
    const timer = window.setInterval(() => void refreshSystem(), 1500)
    return () => window.clearInterval(timer)
  }, [refreshSystem, system])

  useEffect(() => {
    if (!system?.graph_url) {
      setGraph(null)
      return
    }
    let active = true
    getPipelineSystemGraph(system.system_id)
      .then((payload) => {
        if (active) setGraph(payload as SystemGraph)
      })
      .catch((loadError) => {
        if (active) setError(loadError instanceof Error ? loadError.message : 'Failed to load system graph')
      })
    return () => {
      active = false
    }
  }, [system?.system_id, system?.graph_url, system?.connector_review?.revision])

  const connectors = useMemo(() => {
    return (graph?.sheets || []).flatMap((sheet) =>
      (sheet.graph_v1.edges || [])
        .filter((edge) => edge.off_page_connector)
        .map((edge) => ({
          connectorId: `${sheet.sheet_id}::${edge.id}`,
          sheetId: sheet.sheet_id,
          edgeId: edge.id,
          connectorKey: edge.off_page_connector?.connector_key || '',
          targetSheetId: edge.off_page_connector?.target_sheet_reference || edge.off_page_connector?.reference_value || '',
          rawReference: edge.off_page_connector?.raw_reference_text || '',
        }))
    )
  }, [graph])

  useEffect(() => {
    if (!system || !connectors.length) return
    const saved = new Map(system.connector_review.connector_overrides.map((item) => [item.connector_id, item]))
    setOverrides(Object.fromEntries(connectors.map((connector) => [
      connector.connectorId,
      saved.get(connector.connectorId) || {
        connector_id: connector.connectorId,
        target_sheet_id: connector.targetSheetId,
        connector_key: connector.connectorKey,
        review_state: 'accepted',
      },
    ])))
    setManualPairs(system.connector_review.manual_pairs || [])
    setReviewer(system.connector_review.reviewer || '')
  }, [system?.connector_review?.revision, connectors])

  if (!system) return null

  const nextReviewPage = system.pages.find((page) => page.status.startsWith('awaiting_'))
  const completedPages = system.pages.filter((page) => page.status === 'completed').length

  const updateOverride = (connectorId: string, patch: Partial<ConnectorOverride>) => {
    setOverrides((current) => ({
      ...current,
      [connectorId]: { ...current[connectorId], connector_id: connectorId, ...patch },
    }))
  }

  const addManualPair = () => {
    if (!leftConnector || !rightConnector || leftConnector === rightConnector) return
    setManualPairs((current) => [
      ...current.filter((pair) => ![pair.left_connector_id, pair.right_connector_id].some((id) => id === leftConnector || id === rightConnector)),
      { left_connector_id: leftConnector, right_connector_id: rightConnector },
    ])
    setLeftConnector('')
    setRightConnector('')
  }

  const saveReview = async () => {
    setSaving(true)
    setError(null)
    try {
      const updated = await putPipelineSystemConnectorReview(system.system_id, {
        connector_overrides: Object.values(overrides),
        manual_pairs: manualPairs,
        reviewer: reviewer || undefined,
      })
      useAppStore.setState({ pipelineSystem: updated })
      setGraph(await getPipelineSystemGraph(system.system_id) as SystemGraph)
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Failed to save connector review')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto bg-[var(--bg-canvas)] p-6">
      <div className="mx-auto max-w-6xl space-y-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold">Pipeline system</h1>
            <div className="mt-1 text-xs text-[var(--text-secondary)]">{system.system_id}</div>
          </div>
          <div className="flex items-center gap-2">
            {system.graph_url && (
              <Button variant="outline" asChild>
                <a href={system.graph_url} download>
                  <Download className="h-4 w-4" /> Graph v2
                </a>
              </Button>
            )}
            {nextReviewPage && (
              <Button variant="cta" onClick={() => openPage(nextReviewPage.job_id)}>
                Review next page <ExternalLink className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-4">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium capitalize">{statusLabel(system.status)}</span>
            <span className="text-[var(--text-secondary)]">{completedPages}/{system.pages.length} pages complete</span>
          </div>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-[var(--bg-primary)]">
            <div className="h-full bg-[var(--accent)]" style={{ width: `${Math.round((completedPages / system.pages.length) * 100)}%` }} />
          </div>
        </div>

        <section>
          <h2 className="text-sm font-semibold">Pages</h2>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            {system.pages.map((page) => (
              <div key={page.job_id} className="flex items-center gap-3 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-4">
                {page.status === 'completed' ? (
                  <CheckCircle2 className="h-5 w-5 text-[var(--success)]" />
                ) : page.status === 'processing' ? (
                  <Loader2 className="h-5 w-5 animate-spin text-[var(--accent)]" />
                ) : (
                  <AlertTriangle className="h-5 w-5 text-[var(--warning)]" />
                )}
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">{page.sheet_id}</div>
                  <div className="truncate text-xs text-[var(--text-secondary)]">{page.source_filename} · {statusLabel(page.status)}</div>
                </div>
                <Button variant="outline" size="sm" onClick={() => openPage(page.job_id)} disabled={page.status === 'processing'}>
                  Open
                </Button>
              </div>
            ))}
          </div>
        </section>

        {graph && (
          <section className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold">Cross-page connectors</h2>
              <div className="text-xs text-[var(--text-secondary)]">
                {graph.cross_sheet_edges?.length || 0} linked · {graph.merge_issues?.length || 0} unresolved
              </div>
            </div>
            <div className="space-y-3">
              {connectors.map((connector) => {
                const override = overrides[connector.connectorId]
                return (
                  <div key={connector.connectorId} className="grid gap-3 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-4 md:grid-cols-[1fr_1fr_1fr_auto]">
                    <div>
                      <div className="text-xs text-[var(--text-secondary)]">Connector</div>
                      <div className="mt-1 text-sm font-medium">{connector.sheetId} · {connector.edgeId}</div>
                      {connector.rawReference && <div className="mt-1 text-xs text-[var(--text-secondary)]">OCR: {connector.rawReference}</div>}
                    </div>
                    <label className="text-xs text-[var(--text-secondary)]">
                      Line / tag
                      <input
                        value={override?.connector_key || ''}
                        onChange={(event) => updateOverride(connector.connectorId, { connector_key: event.target.value })}
                        className="mt-1 w-full rounded-md border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm text-[var(--text-primary)]"
                      />
                    </label>
                    <label className="text-xs text-[var(--text-secondary)]">
                      Target sheet
                      <select
                        value={override?.target_sheet_id || ''}
                        onChange={(event) => updateOverride(connector.connectorId, { target_sheet_id: event.target.value })}
                        className="mt-1 w-full rounded-md border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm text-[var(--text-primary)]"
                      >
                        <option value="">Select sheet</option>
                        {system.pages.filter((page) => page.sheet_id !== connector.sheetId).map((page) => (
                          <option key={page.sheet_id} value={page.sheet_id}>{page.sheet_id}</option>
                        ))}
                      </select>
                    </label>
                    <label className="flex items-center gap-2 text-xs">
                      <input
                        type="checkbox"
                        checked={override?.review_state === 'rejected'}
                        onChange={(event) => updateOverride(connector.connectorId, { review_state: event.target.checked ? 'rejected' : 'accepted' })}
                      />
                      Reject
                    </label>
                  </div>
                )
              })}
            </div>

            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-4">
              <div className="flex items-center gap-2 text-sm font-medium"><Link2 className="h-4 w-4" /> Manual pair</div>
              <div className="mt-3 grid gap-2 md:grid-cols-[1fr_1fr_auto]">
                {[leftConnector, rightConnector].map((value, index) => (
                  <select
                    key={index}
                    value={value}
                    onChange={(event) => index === 0 ? setLeftConnector(event.target.value) : setRightConnector(event.target.value)}
                    className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm"
                  >
                    <option value="">Select connector</option>
                    {connectors.map((connector) => <option key={connector.connectorId} value={connector.connectorId}>{connector.connectorId}</option>)}
                  </select>
                ))}
                <Button variant="outline" onClick={addManualPair} disabled={!leftConnector || !rightConnector}>Add pair</Button>
              </div>
              <div className="mt-3 space-y-2">
                {manualPairs.map((pair) => (
                  <div key={`${pair.left_connector_id}:${pair.right_connector_id}`} className="flex items-center justify-between rounded-md bg-[var(--bg-primary)] px-3 py-2 text-xs">
                    <span>{pair.left_connector_id} ↔ {pair.right_connector_id}</span>
                    <Button variant="ghost" size="icon" onClick={() => setManualPairs((current) => current.filter((item) => item !== pair))} aria-label="Remove manual pair">
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>

            {(graph.merge_issues || []).length > 0 && (
              <div className="rounded-xl border border-[var(--warning)]/40 bg-[var(--warning)]/5 p-4">
                <div className="text-sm font-medium">Unresolved connectors</div>
                <ul className="mt-2 space-y-1 text-xs text-[var(--text-secondary)]">
                  {graph.merge_issues?.map((issue) => <li key={issue.issue_id}>{statusLabel(issue.type)} · {issue.sheets_involved?.join(', ') || issue.issue_id}</li>)}
                </ul>
              </div>
            )}

            <div className="flex flex-wrap items-end justify-between gap-3">
              <label className="text-xs text-[var(--text-secondary)]">
                Reviewer
                <input value={reviewer} onChange={(event) => setReviewer(event.target.value)} className="ml-2 rounded-md border border-[var(--border-muted)] bg-[var(--bg-secondary)] px-3 py-2 text-sm" />
              </label>
              <Button variant="cta" onClick={saveReview} disabled={saving}>
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                Save connector review
              </Button>
            </div>
          </section>
        )}

        {error && <div className="rounded-lg border border-[var(--danger)]/40 bg-[var(--danger)]/5 p-3 text-xs text-[var(--danger)]">{error}</div>}
      </div>
    </div>
  )
}
