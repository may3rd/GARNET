import { useEffect, useMemo, useState } from 'react'
import type { PipelineJob } from '@/types'

type JsonValue = string | number | boolean | null | JsonObject | JsonValue[]
type JsonObject = Record<string, JsonValue>

function SummaryCard({ title, entries }: { title: string; entries: Array<[string, JsonValue | undefined]> }) {
  const visibleEntries = entries.filter(([, value]) => value !== undefined)
  if (!visibleEntries.length) return null
  return (
    <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
      <div className="text-sm font-semibold">{title}</div>
      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {visibleEntries.map(([label, value]) => (
          <div key={label} className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
            <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">{label}</div>
            <div className="mt-1 text-sm font-semibold">{String(value)}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function PipelineResultsView({ job }: { job: PipelineJob }) {
  const [jsonSummaries, setJsonSummaries] = useState<Record<string, JsonObject>>({})
  const stages = job.manifest?.stages ?? []
  const imageArtifacts = job.artifacts.filter((artifact) => /\.(png|jpg|jpeg|webp)$/i.test(artifact.name))
  const jsonArtifacts = job.artifacts.filter((artifact) => artifact.name.endsWith('.json'))
  const route = job.manifest?.ocr_route ?? job.ocr_route
  const spotlightImageArtifacts = useMemo(
    () =>
      imageArtifacts.filter((artifact) =>
        [
          'stage4_line_number_overlay.png',
          'stage12_text_attachment_overlay.png',
        ].includes(artifact.name)
      ),
    [imageArtifacts]
  )
  const summaryArtifacts = useMemo(
    () =>
      jsonArtifacts.filter((artifact) =>
        [
          'stage2_ocr_summary.json',
          'stage4_objects_summary.json',
          'stage4_line_number_summary.json',
          'stage5_pipe_mask_summary.json',
          'stage6_pipe_mask_sealed_summary.json',
          'stage7_pipe_skeleton_summary.json',
          'stage8_node_summary.json',
          'stage9_node_cluster_summary.json',
          'stage10_pipe_edge_summary.json',
          'stage11_junction_review_summary.json',
          'stage12_equipment_attachment_summary.json',
          'stage12_text_attachment_summary.json',
          'stage12_graph_summary.json',
          'stage13_graph_qa_summary.json',
        ].includes(artifact.name)
      ),
    [jsonArtifacts]
  )

  useEffect(() => {
    let active = true
    const load = async () => {
      const results = await Promise.all(
        summaryArtifacts.map(async (artifact) => {
          try {
            const response = await fetch(artifact.url)
            if (!response.ok) return [artifact.name, null] as const
            const payload = (await response.json()) as JsonObject
            return [artifact.name, payload] as const
          } catch {
            return [artifact.name, null] as const
          }
        })
      )
      if (!active) return
      setJsonSummaries(
        Object.fromEntries(results.filter(([, payload]) => payload !== null)) as Record<string, JsonObject>
      )
    }
    void load()
    return () => {
      active = false
    }
  }, [summaryArtifacts])

  return (
    <div className="h-full overflow-auto bg-[var(--bg-canvas)]">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 py-6">
        <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
          <div className="text-lg font-semibold">Pipeline Review</div>
          <div className="mt-1 text-sm text-[var(--text-secondary)]">
            Full staged review through Stage 13: normalization, OCR, object detection, pipe mask, sealing, skeleton, nodes, clustering, edge tracing, junction review, graph assembly, and graph QA.
          </div>
          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
              <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Job</div>
              <div className="mt-1 font-mono text-xs">{job.job_id}</div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
              <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Status</div>
              <div className="mt-1 text-sm font-semibold capitalize">{job.status}</div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
              <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Current Stage</div>
              <div className="mt-1 text-sm font-semibold">{job.current_stage ?? 'Queued'}</div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
              <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">OCR Route</div>
              <div className="mt-1 text-sm font-semibold uppercase">{route}</div>
            </div>
          </div>
        </div>

        <SummaryCard
          title="Line Numbers"
          entries={[
            ['Detected Tags', jsonSummaries['stage4_line_number_summary.json']?.line_number_object_count],
            ['Matched OCR', jsonSummaries['stage4_line_number_summary.json']?.matched_line_number_count],
            ['Rejected Tags', jsonSummaries['stage4_line_number_summary.json']?.rejected_line_number_count],
            ['Attach Candidates', jsonSummaries['stage12_text_attachment_summary.json']?.candidate_count],
          ]}
        />

        <SummaryCard
          title="Attachments"
          entries={[
            ['Equipment Attached', jsonSummaries['stage12_equipment_attachment_summary.json']?.accepted_attachment_count],
            ['Equipment Rejected', jsonSummaries['stage12_equipment_attachment_summary.json']?.rejected_attachment_count],
            ['Text Attached', jsonSummaries['stage12_text_attachment_summary.json']?.accepted_attachment_count],
            ['Text Rejected', jsonSummaries['stage12_text_attachment_summary.json']?.rejected_attachment_count],
          ]}
        />

        <SummaryCard
          title="Graph Summary"
          entries={[
            ['Nodes', jsonSummaries['stage12_graph_summary.json']?.node_count],
            ['Edges', jsonSummaries['stage12_graph_summary.json']?.edge_count],
            ['Components', jsonSummaries['stage12_graph_summary.json']?.connected_component_count],
            ['Unresolved Junctions', jsonSummaries['stage12_graph_summary.json']?.unresolved_junction_count],
          ]}
        />

        <SummaryCard
          title="QA Summary"
          entries={[
            ['QA Components', jsonSummaries['stage13_graph_qa_summary.json']?.connected_component_count],
            ['Articulation Points', jsonSummaries['stage13_graph_qa_summary.json']?.articulation_point_count],
            ['Isolated Nodes', jsonSummaries['stage13_graph_qa_summary.json']?.isolated_node_count],
            ['Review Queue', jsonSummaries['stage13_graph_qa_summary.json']?.review_queue_count],
          ]}
        />

        <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
          <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
            <div className="text-sm font-semibold">Stages</div>
            <div className="mt-4 space-y-3">
              {stages.map((stage) => (
                <div key={stage.name} className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
                  <div className="text-xs uppercase tracking-wide text-[var(--text-secondary)]">Stage {stage.num}</div>
                  <div className="mt-1 text-sm font-semibold">{stage.name}</div>
                  <div className="mt-1 text-xs text-[var(--text-secondary)]">
                    {stage.status} {stage.duration_sec !== undefined ? `• ${stage.duration_sec.toFixed(3)}s` : ''}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-6">
            {spotlightImageArtifacts.length > 0 && (
              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
                <div className="text-sm font-semibold">Line Number Review</div>
                <div className="mt-4 grid gap-4 md:grid-cols-2">
                  {spotlightImageArtifacts.map((artifact) => (
                    <div key={artifact.name} className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
                      <div className="mb-2 text-xs font-semibold text-[var(--text-secondary)]">{artifact.name}</div>
                      <img src={artifact.url} alt={artifact.name} className="w-full rounded-lg border border-[var(--border-muted)]" />
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
              <div className="text-sm font-semibold">Image Artifacts</div>
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                {imageArtifacts.map((artifact) => (
                  <div key={artifact.name} className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] p-3">
                    <div className="mb-2 text-xs font-semibold text-[var(--text-secondary)]">{artifact.name}</div>
                    <img src={artifact.url} alt={artifact.name} className="w-full rounded-lg border border-[var(--border-muted)]" />
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
              <div className="text-sm font-semibold">JSON Artifacts</div>
              <div className="mt-4 space-y-2">
                {jsonArtifacts.map((artifact) => (
                  <a
                    key={artifact.name}
                    href={artifact.url}
                    target="_blank"
                    rel="noreferrer"
                    className="block rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-sm text-[var(--accent)]"
                  >
                    {artifact.name}
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
