import { useAppStore } from '@/stores/appStore'
import { CanvasView } from '@/components/CanvasView'
import { ObjectSidebar } from '@/components/ObjectSidebar'

export function ResultsView() {
  const result = useAppStore((state) => state.result)

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

  return (
    <div className="flex h-full">
      <div className="flex-1 relative">
        <CanvasView imageUrl={result.image_url} objects={result.objects} />
      </div>
      <div className="w-[320px] shrink-0">
        <ObjectSidebar objects={result.objects} onExport={handleExport} />
      </div>
    </div>
  )
}
