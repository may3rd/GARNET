import { lazy, Suspense, useEffect } from 'react'
import { Header } from '@/components/Header'
import { UploadZone } from '@/components/UploadZone'
import { DetectionSetup } from '@/components/DetectionSetup'
import { ProcessingView } from '@/components/ProcessingView'
import { useAppStore } from '@/stores/appStore'
import { cn } from '@/lib/utils'

const ResultsView = lazy(() => import('@/components/ResultsView').then((module) => ({ default: module.ResultsView })))
const PipelineResultsView = lazy(() =>
  import('@/components/PipelineResultsView').then((module) => ({ default: module.PipelineResultsView }))
)
const BatchResultsView = lazy(() =>
  import('@/components/BatchResultsView').then((module) => ({ default: module.BatchResultsView }))
)
const PipelineSystemSetupView = lazy(() =>
  import('@/components/PipelineSystemSetupView').then((module) => ({ default: module.PipelineSystemSetupView }))
)
const PipelineSystemView = lazy(() =>
  import('@/components/PipelineSystemView').then((module) => ({ default: module.PipelineSystemView }))
)

function ViewLoading() {
  return (
    <div className="flex h-full items-center justify-center bg-[var(--bg-canvas)] text-sm text-[var(--text-secondary)]">
      Loading view...
    </div>
  )
}

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
  const processingMode = useAppStore((state) => state.processingMode)
  const pipelineJob = useAppStore((state) => state.pipelineJob)
  const restorePipelineSystem = useAppStore((state) => state.restorePipelineSystem)

  useEffect(() => {
    void restorePipelineSystem()
  }, [restorePipelineSystem])

  return (
    <div className="flex flex-col h-full bg-[var(--bg-primary)]">
      <Header />
      <main className="flex-1 overflow-hidden">
        {currentView === 'empty' && <UploadZone />}

        {currentView === 'preview' && (
          <div className="flex h-full flex-col lg:flex-row">
            <PreviewPane />
            <div className="w-full lg:w-[320px] border-t lg:border-t-0 lg:border-l border-[var(--border-muted)] bg-[var(--bg-secondary)] overflow-y-auto max-h-[45vh] lg:max-h-none">
              <DetectionSetup />
            </div>
          </div>
        )}

        {currentView === 'processing' && <ProcessingView />}

        {currentView === 'batch' && (
          <Suspense fallback={<ViewLoading />}>
            <div className="flex h-full flex-col lg:flex-row">
              {processingMode === 'pipeline' ? <PipelineSystemSetupView /> : <BatchResultsView />}
              <div className="w-full lg:w-[320px] border-t lg:border-t-0 lg:border-l border-[var(--border-muted)] bg-[var(--bg-secondary)] overflow-y-auto max-h-[45vh] lg:max-h-none">
                <DetectionSetup />
              </div>
            </div>
          </Suspense>
        )}

        {currentView === 'system' && (
          <Suspense fallback={<ViewLoading />}>
            <PipelineSystemView />
          </Suspense>
        )}

        {currentView === 'results' && (
          <Suspense fallback={<ViewLoading />}>
            {pipelineJob ? <PipelineResultsView job={pipelineJob} /> : <ResultsView />}
          </Suspense>
        )}
      </main>
    </div>
  )
}
