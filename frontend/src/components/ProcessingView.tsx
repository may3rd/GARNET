import { useAppStore } from '@/stores/appStore'

export function ProcessingView() {
  const progress = useAppStore((state) => state.progress)

  return (
    <div className="flex flex-col items-center justify-center h-full px-6">
      <div className="w-full max-w-xl bg-[var(--bg-secondary)] border border-[var(--border-muted)] rounded-2xl p-6">
        <div className="text-sm font-semibold">Analyzing P&amp;ID...</div>
        <div className="text-xs text-[var(--text-secondary)] mt-1">
          {progress?.step || 'Preparing detection'}
        </div>
        <div className="mt-4 h-2 w-full bg-[var(--bg-primary)] rounded-full overflow-hidden">
          <div
            className="h-full bg-[var(--accent)] transition-all"
            style={{ width: `${progress?.percent ?? 0}%` }}
          />
        </div>
        <div className="text-right text-xs text-[var(--text-secondary)] mt-2">
          {progress?.percent ?? 0}%
        </div>
      </div>
    </div>
  )
}
