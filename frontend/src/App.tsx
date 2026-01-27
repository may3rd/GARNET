import { Header } from '@/components/Header'
import { UploadZone } from '@/components/UploadZone'
import { DetectionSetup } from '@/components/DetectionSetup'
import { ProcessingView } from '@/components/ProcessingView'
import { ResultsView } from '@/components/ResultsView'
import { BatchResultsView } from '@/components/BatchResultsView'
import { useAppStore } from '@/stores/appStore'
import { cn } from '@/lib/utils'

function PreviewPane() {
  const imageUrl = useAppStore((state) => state.imageUrl)
  const imageMeta = useAppStore((state) => state.imageMeta)

  if (!imageUrl) return null

  return (
    <div className="flex-1 flex items-center justify-center bg-[var(--bg-canvas)]">
      <div className="max-w-[85%] max-h-[85%] relative">
        <img
          src={imageUrl}
          alt="P&ID preview"
          className="rounded-xl border border-[var(--border-muted)]"
        />
        {imageMeta && (
          <div className={cn(
            'absolute bottom-3 right-3',
            'text-[11px] px-2 py-1 rounded-lg',
            'bg-black/60 text-white'
          )}>
            {imageMeta.width} × {imageMeta.height}
          </div>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const currentView = useAppStore((state) => state.currentView)

  return (
    <div className="flex flex-col h-full bg-[var(--bg-primary)]">
      <Header />
      <main className="flex-1 overflow-hidden">
        {currentView === 'empty' && <UploadZone />}

        {currentView === 'preview' && (
          <div className="flex h-full">
            <PreviewPane />
            <div className="w-[320px] border-l border-[var(--border-muted)] bg-[var(--bg-secondary)] overflow-y-auto">
              <DetectionSetup />
            </div>
          </div>
        )}

        {currentView === 'processing' && <ProcessingView />}

        {currentView === 'batch' && (
          <div className="flex h-full">
            <BatchResultsView />
            <div className="w-[320px] border-l border-[var(--border-muted)] bg-[var(--bg-secondary)] overflow-y-auto">
              <DetectionSetup />
            </div>
          </div>
        )}

        {currentView === 'results' && <ResultsView />}
      </main>
    </div>
  )
}
