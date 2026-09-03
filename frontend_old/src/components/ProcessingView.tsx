import { CheckCircle2, Loader2, XCircle } from 'lucide-react'
import { useAppStore } from '@/stores/appStore'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

export function ProcessingView() {
  const processingMode = useAppStore((state) => state.processingMode)
  const pipelineJob = useAppStore((state) => state.pipelineJob)
  const progress = useAppStore((state) => state.progress)
  const cancelDetection = useAppStore((state) => state.cancelDetection)
  const percent = progress?.percent ?? 0
  const steps = processingMode === 'pipeline'
    ? (pipelineJob?.manifest?.stages.map((stage, index) => ({
      label: stage.name.replaceAll('_', ' '),
      threshold: Math.round(((index + 1) / Math.max(pipelineJob.manifest?.stages.length || 1, 1)) * 100),
    })) ?? [{ label: 'stage1 input normalization', threshold: 100 }])
    : [
      { label: 'Upload image', threshold: 15 },
      { label: 'Run detection', threshold: 55 },
      { label: 'Extract text (OCR)', threshold: 80 },
      { label: 'Finalize results', threshold: 100 },
    ]

  return (
    <div className="flex flex-col items-center justify-center h-full px-6">
      <div className="w-full max-w-xl bg-[var(--bg-secondary)] border border-[var(--border-muted)] rounded-2xl p-6">
        <div className="text-sm font-semibold">
          {processingMode === 'pipeline' ? 'Running pipeline...' : 'Analyzing P&amp;ID...'}
        </div>
        <div className="text-xs text-[var(--text-secondary)] mt-1">
          {progress?.step || (processingMode === 'pipeline' ? 'Preparing pipeline' : 'Preparing detection')}
        </div>
        <div className="mt-4 h-2 w-full bg-[var(--bg-primary)] rounded-full overflow-hidden">
          <div
            className="h-full bg-[var(--accent)] transition-all"
            style={{ width: `${percent}%` }}
          />
        </div>
        <div className="text-right text-xs text-[var(--text-secondary)] mt-2">
          {percent}%
        </div>

        <div className="mt-5 space-y-2 text-xs">
          {steps.map((step, index) => {
            const isDone = percent >= step.threshold
            const isActive = percent < step.threshold && (index === 0 || percent >= steps[index - 1].threshold)
            return (
              <div key={step.label} className="flex items-center gap-2">
                {isDone ? (
                  <CheckCircle2 className="h-4 w-4 text-[var(--success)]" />
                ) : isActive ? (
                  <Loader2 className="h-4 w-4 animate-spin text-[var(--accent)]" />
                ) : (
                  <div className="h-4 w-4 rounded-full border border-[var(--border-muted)]" />
                )}
                <span className={cn(
                  isDone ? 'text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'
                )}>
                  {step.label}
                </span>
              </div>
            )
          })}
        </div>

        <Button
          variant="outline"
          className="mt-6 w-full hover:border-[var(--danger)] hover:text-[var(--danger)]"
          onClick={cancelDetection}
        >
          <XCircle className="h-4 w-4 mr-2" />
          {processingMode === 'pipeline' ? 'Stop polling' : 'Cancel detection'}
        </Button>
      </div>
    </div>
  )
}
