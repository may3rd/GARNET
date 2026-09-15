import { useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  ExternalLink,
  Filter,
  LayoutGrid,
  Link2,
  Loader2,
  MapPin,
  Save,
  Trash2,
  X,
} from 'lucide-react'
import type { ConnectorOverride, ManualConnectorPair, PipelineArtifact } from '@/types'
import { getPipelineSystemGraph, putPipelineSystemConnectorReview } from '@/lib/api'
import { useAppStore } from '@/stores/appStore'
import { Button } from '@/components/ui/button'

type GraphEdge = {
  id: string
  src?: string
  dst?: string
  off_page_connector?: {
    connector_key?: string
    target_sheet_reference?: string
    reference_value?: string
    raw_reference_text?: string
  }
}

type GraphNode = {
  id: string
  bbox?: { x: number; y: number; w: number; h: number }
  geometry?: { center?: { x: number; y: number } }
}

type CrossSheetEdge = {
  id?: string
  connector_key?: string
  reference_value?: string
  reference_type?: string
  sheets?: string[]
  status?: string
  match_method?: string
  terminals?: Array<{ connector_id: string; sheet?: string }>
}

type MergeIssue = {
  issue_id: string
  type: string
  sheets_involved?: string[]
  connectors?: Array<{ connector_id?: string }>
}

type SystemGraph = {
  sheets?: Array<{
    sheet_id: string
    graph_v1: {
      edges?: GraphEdge[]
      nodes?: GraphNode[]
      tiling?: { tile?: { tile_width?: number; tile_height?: number } }
    }
  }>
  cross_sheet_edges?: CrossSheetEdge[]
  merge_issues?: MergeIssue[]
}

type Connector = {
  connectorId: string
  sheetId: string
  edgeId: string
  connectorKey: string
  targetSheetId: string
  rawReference: string
  position: { x: number; y: number } | null
  issues: string[]
  resolved: boolean
}

const statusLabel = (status: string) => status.replaceAll('_', ' ')

function pickBaseImageUrl(artifacts: PipelineArtifact[] | undefined): string {
  if (!artifacts) return ''
  for (const name of ['stage1_gray.png', 'stage1_gray_equalized.png']) {
    const match = artifacts.find((artifact) => artifact.name === name)
    if (match) return match.url
  }
  return artifacts[0]?.url ?? ''
}

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
  const [filter, setFilter] = useState<string>('all')
  const [unresolvedFirst, setUnresolvedFirst] = useState(false)
  const [selectedConnectorId, setSelectedConnectorId] = useState<string | null>(null)
  const [mapSheetId, setMapSheetId] = useState<string | null>(null)

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

  const connectors: Connector[] = useMemo(() => {
    const issuesByConnector = new Map<string, string[]>()
    for (const issue of graph?.merge_issues || []) {
      for (const item of issue.connectors || []) {
        if (!item.connector_id) continue
        const bucket = issuesByConnector.get(item.connector_id) || []
        if (!bucket.includes(issue.type)) bucket.push(issue.type)
        issuesByConnector.set(item.connector_id, bucket)
      }
    }
    const resolvedIds = new Set<string>()
    for (const edge of graph?.cross_sheet_edges || []) {
      for (const terminal of edge.terminals || []) {
        if (terminal.connector_id) resolvedIds.add(terminal.connector_id)
      }
    }
    return (graph?.sheets || []).flatMap((sheet) => {
      const nodesById = new Map((sheet.graph_v1.nodes || []).map((node) => [node.id, node]))
      const tile = sheet.graph_v1.tiling?.tile
      const width = tile?.tile_width
      const height = tile?.tile_height
      return (sheet.graph_v1.edges || [])
        .filter((edge) => edge.off_page_connector)
        .map((edge) => {
          const connectorNodeId = [edge.src, edge.dst].find((id) => String(id).startsWith('connection::'))
          const node = connectorNodeId ? nodesById.get(String(connectorNodeId)) : undefined
          const bbox = node?.bbox
          const center = node?.geometry?.center
          const x = bbox ? bbox.x + bbox.w / 2 : center?.x
          const y = bbox ? bbox.y + bbox.h / 2 : center?.y
          const connectorId = `${sheet.sheet_id}::${edge.id}`
          return {
            connectorId,
            sheetId: sheet.sheet_id,
            edgeId: edge.id,
            connectorKey: edge.off_page_connector?.connector_key || '',
            targetSheetId: edge.off_page_connector?.target_sheet_reference || edge.off_page_connector?.reference_value || '',
            rawReference: edge.off_page_connector?.raw_reference_text || '',
            position: x != null && y != null && width && height ? { x: x / width, y: y / height } : null,
            issues: issuesByConnector.get(connectorId) || [],
            resolved: resolvedIds.has(connectorId),
          }
        })
    })
  }, [graph])

  const issueTypes = useMemo(() => {
    const types = new Set<string>()
    for (const connector of connectors) for (const issue of connector.issues) types.add(issue)
    return Array.from(types).sort()
  }, [connectors])

  const visibleConnectors = useMemo(() => {
    const list = connectors.filter((connector) => {
      if (filter === 'all') return true
      if (filter === 'unresolved') return connector.issues.length > 0
      return connector.issues.includes(filter)
    })
    if (unresolvedFirst) {
      return [...list].sort((a, b) => Number(b.issues.length > 0) - Number(a.issues.length > 0) || a.connectorId.localeCompare(b.connectorId))
    }
    return [...list].sort((a, b) => a.connectorId.localeCompare(b.connectorId))
  }, [connectors, filter, unresolvedFirst])

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

  useEffect(() => {
    if (!selectedConnectorId) return
    const connector = connectors.find((item) => item.connectorId === selectedConnectorId)
    if (connector) setMapSheetId(connector.sheetId)
  }, [selectedConnectorId, connectors])

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

  const rejectUnresolved = () => {
    setOverrides((current) => {
      const next = { ...current }
      for (const connector of connectors) {
        if (connector.issues.length === 0) continue
        next[connector.connectorId] = { ...current[connector.connectorId], connector_id: connector.connectorId, review_state: 'rejected' }
      }
      return next
    })
  }

  const acceptAll = () => {
    setOverrides((current) => {
      const next = { ...current }
      for (const connector of connectors) {
        next[connector.connectorId] = { ...current[connector.connectorId], connector_id: connector.connectorId, review_state: 'accepted' }
      }
      return next
    })
  }

  const mapSheet = mapSheetId ? system.pages.find((page) => page.sheet_id === mapSheetId) : undefined
  const mapImageUrl = mapSheet ? pickBaseImageUrl(mapSheet.job.artifacts) : ''
  const mapTileWidth = (() => {
    const sheet = graph?.sheets?.find((item) => item.sheet_id === mapSheetId)
    return sheet?.graph_v1.tiling?.tile?.tile_width
  })()
  const mapTileHeight = (() => {
    const sheet = graph?.sheets?.find((item) => item.sheet_id === mapSheetId)
    return sheet?.graph_v1.tiling?.tile?.tile_height
  })()

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

            {graph.cross_sheet_edges && graph.cross_sheet_edges.length > 0 && (
              <div className="rounded-xl border border-[var(--success)]/30 bg-[var(--success)]/5 p-4">
                <div className="flex items-center gap-2 text-sm font-medium text-[var(--success)]"><Link2 className="h-4 w-4" /> Resolved connections</div>
                <div className="mt-3 space-y-2">
                  {graph.cross_sheet_edges.map((edge) => (
                    <div key={edge.id || edge.reference_value} className="flex flex-wrap items-center justify-between gap-2 rounded-md bg-[var(--bg-primary)] px-3 py-2 text-xs">
                      <div className="flex min-w-0 items-center gap-2">
                        <span className="font-mono font-medium text-[var(--text-primary)]">{edge.connector_key || edge.reference_value}</span>
                        <span className="text-[var(--text-secondary)]">
                          {edge.sheets?.join(' ↔ ')}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-[var(--text-secondary)]">
                        <span className="rounded bg-[var(--bg-secondary)] px-2 py-0.5">{edge.match_method || 'manual'}</span>
                        {edge.status && <span className="rounded bg-[var(--bg-secondary)] px-2 py-0.5">{edge.status}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
                  <Filter className="h-3.5 w-3.5" />
                  <select
                    value={filter}
                    onChange={(event) => setFilter(event.target.value)}
                    className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-primary)] px-2 py-1.5 text-sm"
                  >
                    <option value="all">All connectors</option>
                    <option value="unresolved">Unresolved</option>
                    {issueTypes.map((type) => <option key={type} value={type}>{statusLabel(type)}</option>)}
                  </select>
                </label>
                <label className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
                  <input
                    type="checkbox"
                    checked={unresolvedFirst}
                    onChange={(event) => setUnresolvedFirst(event.target.checked)}
                  />
                  Unresolved first
                </label>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button variant="outline" size="sm" onClick={acceptAll}>
                  <CheckCircle2 className="h-3.5 w-3.5" /> Accept all
                </Button>
                <Button variant="outline" size="sm" onClick={rejectUnresolved}>
                  <X className="h-3.5 w-3.5" /> Reject unresolved
                </Button>
                <Button variant="outline" size="sm" onClick={() => setManualPairs([])} disabled={manualPairs.length === 0}>
                  <Trash2 className="h-3.5 w-3.5" /> Clear pairs
                </Button>
              </div>
            </div>

            <div className="space-y-3">
              {visibleConnectors.length === 0 && (
                <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-4 text-xs text-[var(--text-secondary)]">No connectors match the current filter.</div>
              )}
              {visibleConnectors.map((connector) => {
                const override = overrides[connector.connectorId]
                return (
                  <div
                    key={connector.connectorId}
                    className={`grid gap-3 rounded-xl border bg-[var(--bg-secondary)] p-4 md:grid-cols-[1fr_1fr_1fr_auto] ${
                      selectedConnectorId === connector.connectorId ? 'border-[var(--accent)] ring-1 ring-[var(--accent)]' : 'border-[var(--border-muted)]'
                    }`}
                  >
                    <div>
                      <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
                        <span>Connector</span>
                        {connector.resolved && <span className="rounded bg-[var(--success)]/15 px-1.5 py-0.5 text-[10px] font-medium text-[var(--success)]">RESOLVED</span>}
                      </div>
                      <div className="mt-1 text-sm font-medium">{connector.sheetId} · {connector.edgeId}</div>
                      <div className="mt-1 flex flex-wrap items-center gap-1">
                        {connector.issues.length === 0 ? (
                          <span className="rounded bg-[var(--bg-primary)] px-1.5 py-0.5 text-[10px] text-[var(--text-secondary)]">no issues</span>
                        ) : (
                          connector.issues.map((issue) => (
                            <span key={issue} className="rounded bg-[var(--warning)]/15 px-1.5 py-0.5 text-[10px] font-medium text-[var(--warning)]">{statusLabel(issue)}</span>
                          ))
                        )}
                      </div>
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
                    <div className="flex flex-col items-start gap-2">
                      <label className="flex items-center gap-2 text-xs">
                        <input
                          type="checkbox"
                          checked={override?.review_state === 'rejected'}
                          onChange={(event) => updateOverride(connector.connectorId, { review_state: event.target.checked ? 'rejected' : 'accepted' })}
                        />
                        Reject
                      </label>
                      {connector.position && (
                        <Button variant="ghost" size="sm" onClick={() => setSelectedConnectorId(connector.connectorId)}>
                          <MapPin className="h-3.5 w-3.5" /> Show on sheet
                        </Button>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-4">
              <div className="flex items-center gap-2 text-sm font-medium"><LayoutGrid className="h-4 w-4" /> Connector map</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {system.pages.map((page) => (
                  <Button key={page.sheet_id} variant={mapSheetId === page.sheet_id ? 'cta' : 'outline'} size="sm" onClick={() => setMapSheetId(page.sheet_id)}>
                    {page.sheet_id}
                  </Button>
                ))}
              </div>
              {mapSheetId && mapImageUrl ? (
                <div className="relative mt-3 overflow-hidden rounded-lg border border-[var(--border-muted)]" style={{ aspectRatio: mapTileWidth && mapTileHeight ? `${mapTileWidth} / ${mapTileHeight}` : undefined }}>
                  <img src={mapImageUrl} alt={`Sheet ${mapSheetId}`} className="absolute inset-0 h-full w-full object-cover" draggable={false} />
                  {connectors
                    .filter((connector) => connector.sheetId === mapSheetId && connector.position)
                    .map((connector) => {
                      const position = connector.position as { x: number; y: number }
                      const selected = selectedConnectorId === connector.connectorId
                      return (
                        <button
                          key={connector.connectorId}
                          type="button"
                          onClick={() => setSelectedConnectorId(connector.connectorId)}
                          aria-label={`Select ${connector.connectorId}`}
                          title={connector.connectorId}
                          className="absolute z-10 -translate-x-1/2 -translate-y-1/2 rounded-full"
                          style={{ left: `${position.x * 100}%`, top: `${position.y * 100}%` }}
                        >
                          <span
                            className={`block h-3 w-3 rounded-full border-2 shadow ${
                              selected ? 'border-white bg-[var(--accent)]' : connector.resolved ? 'border-white bg-[var(--success)]' : 'border-white bg-[var(--warning)]'
                            }`}
                          />
                        </button>
                      )
                    })}
                </div>
              ) : mapSheetId ? (
                <div className="mt-3 text-xs text-[var(--text-secondary)]">No base image available for {mapSheetId}.</div>
              ) : null}
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
              {manualPairs.length > 0 && (
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
              )}
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
