import { Download } from 'lucide-react'
import { Card, SectionHeader, Tag } from '@/components/ui/primitives'
import type { PipelineArtifact } from '@/types'

/**
 * The job's real files on disk — same {name, url} the backend already
 * serves via GET /api/pipeline/jobs/{id} (`job.artifacts`), reused as-is
 * everywhere a screen has a `Sheet` in scope. No separate fetch.
 */
export function ArtifactsPanel({
  jobId,
  artifacts,
  title = 'Artifacts',
}: {
  jobId?: string | null
  artifacts: PipelineArtifact[]
  title?: string
}) {
  return (
    <Card className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-hidden" padding={16}>
      <SectionHeader
        title={title}
        description={jobId ? `job ${jobId.slice(0, 8)}…` : undefined}
        actions={<Tag tone="neutral">{artifacts.length}</Tag>}
      />
      <div className="min-h-0 flex-1 overflow-y-auto">
        {artifacts.length === 0 ? (
          <div style={{ fontSize: 12.5, color: 'var(--muted)' }}>Nothing written yet.</div>
        ) : (
          artifacts.map((a) => (
            <a
              key={a.name}
              href={a.url}
              target="_blank"
              rel="noreferrer"
              className="mono flex items-center gap-2.5"
              style={{
                padding: '7px 0',
                borderBottom: '1px solid color-mix(in oklab, var(--separator) 50%, transparent)',
                fontSize: 12,
                color: 'var(--foreground)',
                textDecoration: 'none',
              }}
            >
              <Download size={13} strokeWidth={1.8} style={{ color: 'var(--muted)', flexShrink: 0 }} />
              <span className="truncate" style={{ flex: 1 }}>
                {a.name}
              </span>
            </a>
          ))
        )}
      </div>
    </Card>
  )
}
