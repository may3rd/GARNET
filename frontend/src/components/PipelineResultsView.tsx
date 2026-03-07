import type { PipelineJob } from '@/types'

export function PipelineResultsView({ job }: { job: PipelineJob }) {
  const stages = job.manifest?.stages ?? []
  const imageArtifacts = job.artifacts.filter((artifact) => /\.(png|jpg|jpeg|webp)$/i.test(artifact.name))
  const jsonArtifacts = job.artifacts.filter((artifact) => artifact.name.endsWith('.json'))

  return (
    <div className="h-full overflow-auto bg-[var(--bg-canvas)]">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 py-6">
        <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-5">
          <div className="text-lg font-semibold">Pipeline Review</div>
          <div className="mt-1 text-sm text-[var(--text-secondary)]">
            Slice 1 output for Stage 1 input normalization.
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
          </div>
        </div>

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
