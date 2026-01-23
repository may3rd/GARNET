import { useMemo, useState } from 'react'
import { useAppStore } from '@/stores/appStore'
import { CanvasView } from '@/components/CanvasView'
import { ObjectSidebar } from '@/components/ObjectSidebar'

export function ResultsView() {
  const result = useAppStore((state) => state.result)
  const resultRunId = useAppStore((state) => state.resultRunId)
  const reviewStatus = useAppStore((state) => state.reviewStatus)
  const setReviewStatus = useAppStore((state) => state.setReviewStatus)
  const selectedObjectKey = useAppStore((state) => state.selectedObjectKey)
  const setSelectedObjectKey = useAppStore((state) => state.setSelectedObjectKey)
  const [hiddenClasses, setHiddenClasses] = useState<Set<string>>(new Set())
  const [confidenceFilter, setConfidenceFilter] = useState(0)

  if (!result) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-[var(--text-secondary)]">
        No results to display.
      </div>
    )
  }

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(result.objects, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'garnet-results.json'
    link.click()
    URL.revokeObjectURL(url)
  }

  const normalizeKey = (value: string) => value.toLowerCase().replace(/_/g, ' ').trim()

  const visibleObjects = useMemo(() => {
    return result.objects.filter((obj) => {
      const key = normalizeKey(obj.Object)
      if (hiddenClasses.has(key)) return false
      return obj.Score >= confidenceFilter
    })
  }, [result.objects, hiddenClasses, confidenceFilter])

  const toggleClass = (classKey: string) => {
    setHiddenClasses((prev) => {
      const next = new Set(prev)
      if (next.has(classKey)) {
        next.delete(classKey)
      } else {
        next.add(classKey)
      }
      return next
    })
  }

  return (
    <div className="flex h-full">
      <div className="flex-1 relative">
        <CanvasView
          key={resultRunId}
          imageUrl={result.image_url}
          objects={visibleObjects}
          selectedObjectKey={selectedObjectKey}
          fitKey={String(resultRunId)}
        />
      </div>
      <div className="w-[320px] shrink-0">
        <ObjectSidebar
          objects={result.objects}
          visibleObjects={visibleObjects}
          hiddenClasses={hiddenClasses}
          confidenceFilter={confidenceFilter}
          onToggleClass={toggleClass}
          onConfidenceChange={setConfidenceFilter}
          reviewStatus={reviewStatus}
          onSetReviewStatus={setReviewStatus}
          selectedObjectKey={selectedObjectKey}
          onSelectObject={setSelectedObjectKey}
          onExport={handleExport}
        />
      </div>
    </div>
  )
}
