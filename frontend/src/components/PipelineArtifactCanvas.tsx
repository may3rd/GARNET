import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Image as KonvaImage, Layer, Stage } from 'react-konva'
import { ZoomControls } from '@/components/ZoomControls'

const MIN_ZOOM = 0.2
const MAX_ZOOM = 6
const PADDING = 24

type PipelineArtifactCanvasProps = {
  imageUrl: string
  title: string
}

export function PipelineArtifactCanvas({ imageUrl, title }: PipelineArtifactCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [stageSize, setStageSize] = useState({ width: 800, height: 520 })
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null)
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const [scale, setScale] = useState(1)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isLoading, setIsLoading] = useState(true)

  const fitToScreen = (nextImageSize = imageSize, nextStageSize = stageSize) => {
    if (!nextImageSize.width || !nextImageSize.height || !nextStageSize.width || !nextStageSize.height) return
    const fitScale = Math.min(
      (nextStageSize.width - PADDING * 2) / nextImageSize.width,
      (nextStageSize.height - PADDING * 2) / nextImageSize.height
    )
    const boundedScale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fitScale || 1))
    setScale(boundedScale)
    setPosition({
      x: (nextStageSize.width - nextImageSize.width * boundedScale) / 2,
      y: (nextStageSize.height - nextImageSize.height * boundedScale) / 2,
    })
  }

  useEffect(() => {
    setIsLoading(true)
    const img = new window.Image()
    img.onload = () => {
      setImageEl(img)
      const nextSize = { width: img.naturalWidth, height: img.naturalHeight }
      setImageSize(nextSize)
      setIsLoading(false)
      requestAnimationFrame(() => fitToScreen(nextSize))
    }
    img.onerror = () => {
      setImageEl(null)
      setImageSize({ width: 0, height: 0 })
      setIsLoading(false)
    }
    img.src = imageUrl
  }, [imageUrl])

  useLayoutEffect(() => {
    const node = containerRef.current
    if (!node) return
    const updateSize = () => {
      const nextSize = {
        width: Math.max(320, node.clientWidth),
        height: Math.max(360, node.clientHeight),
      }
      setStageSize(nextSize)
      if (imageSize.width && imageSize.height) {
        requestAnimationFrame(() => fitToScreen(imageSize, nextSize))
      }
    }
    updateSize()
    const observer = new ResizeObserver(updateSize)
    observer.observe(node)
    return () => observer.disconnect()
  }, [imageSize.width, imageSize.height])

  const zoomAt = (nextScale: number, centerX: number, centerY: number) => {
    const boundedScale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, nextScale))
    const worldX = (centerX - position.x) / scale
    const worldY = (centerY - position.y) / scale
    setScale(boundedScale)
    setPosition({
      x: centerX - worldX * boundedScale,
      y: centerY - worldY * boundedScale,
    })
  }

  return (
    <div className="relative h-[68vh] min-h-[420px] overflow-hidden rounded-xl border border-[var(--border-muted)] bg-[var(--bg-primary)]">
      <div className="absolute left-4 top-4 z-10 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)]/95 px-3 py-2 text-xs font-semibold text-[var(--text-secondary)] backdrop-blur">
        {title}
      </div>
      <div className="absolute right-4 top-4 z-10">
        <ZoomControls
          zoomPercent={Math.round(scale * 100)}
          onZoomIn={() => zoomAt(scale * 1.12, stageSize.width / 2, stageSize.height / 2)}
          onZoomOut={() => zoomAt(scale / 1.12, stageSize.width / 2, stageSize.height / 2)}
          onReset={() => {
            setScale(1)
            setPosition({ x: 0, y: 0 })
          }}
          onFit={() => fitToScreen()}
          onZoomTo={(percent) => zoomAt(percent / 100, stageSize.width / 2, stageSize.height / 2)}
        />
      </div>
      <div ref={containerRef} className="h-full w-full">
        <Stage
          width={stageSize.width}
          height={stageSize.height}
          draggable
          x={position.x}
          y={position.y}
          scaleX={scale}
          scaleY={scale}
          onDragEnd={(event) => setPosition({ x: event.target.x(), y: event.target.y() })}
          onWheel={(event) => {
            event.evt.preventDefault()
            const pointer = event.target.getStage()?.getPointerPosition()
            if (!pointer) return
            const direction = event.evt.deltaY > 0 ? 0.92 : 1.08
            zoomAt(scale * direction, pointer.x, pointer.y)
          }}
        >
          <Layer>{imageEl ? <KonvaImage image={imageEl} width={imageSize.width} height={imageSize.height} /> : null}</Layer>
        </Stage>
      </div>
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-[var(--bg-primary)]/65 text-sm text-[var(--text-secondary)] backdrop-blur-sm">
          Loading artifact...
        </div>
      )}
    </div>
  )
}
