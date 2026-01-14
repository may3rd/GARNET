/**
 * GARNET - P&ID Object Detection Application
 */
import { Layout } from '@/components/layout/Layout'
import { DropZone } from '@/components/upload/DropZone'
import { FilePreview } from '@/components/upload/FilePreview'
import { ProgressPanel } from '@/components/detection/ProgressPanel'
import { Canvas } from '@/components/canvas/Canvas'
import { Sidebar } from '@/components/sidebar/Sidebar'
import { useUIStore } from '@/stores/uiStore'

function App() {
  const { currentView } = useUIStore()

  return (
    <Layout>
      {currentView === 'upload' && <DropZone />}

      {currentView === 'preview' && <FilePreview />}

      {currentView === 'processing' && <ProgressPanel />}

      {currentView === 'results' && (
        <div className="flex h-full">
          <div className="flex-1">
            <Canvas />
          </div>
          <Sidebar />
        </div>
      )}
    </Layout>
  )
}

export default App
