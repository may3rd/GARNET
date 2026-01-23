import { useEffect, useRef, useState } from 'react'
import type { DetectedObject } from '@/types'
import { getCategoryColor } from '@/lib/categoryColors'
import { objectKey } from '@/lib/objectKey'
import { ZoomControls } from '@/components/ZoomControls'
import { cn } from '@/lib/utils'

type CanvasViewProps = {
  imageUrl: string
  objects: DetectedObject[]
  selectedObjectKey: string | null
  fitKey?: string
}

const MIN_ZOOM = 0.2
const MAX_ZOOM = 4
const MINI_MAX_WIDTH = 180
const MINI_MAX_HEIGHT = 120

export function CanvasView({ imageUrl, objects, selectedObjectKey, fitKey }: CanvasViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const minimapRef = useRef<HTMLCanvasElement>(null)
  const minimapMetaRef = useRef({ scale: 1, offsetX: 0, offsetY: 0, width: 0, height: 0 })
  const [imageSize, setImageSize] = useState({ width: 1, height: 1 })
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [activePointerId, setActivePointerId] = useState<number | null>(null)
  const [isMinimapDragging, setIsMinimapDragging] = useState(false)
  const [minimapPointerId, setMinimapPointerId] = useState<number | null>(null)
  const [hasAutoFit, setHasAutoFit] = useState(false)
  const autoFitAttemptsRef = useRef(0)
  const lastFitKeyRef = useRef<string | undefined>(undefined)

  useEffect(() => {
    setHasAutoFit(false)
    autoFitAttemptsRef.current = 0
    const img = imageRef.current
    if (!img) return
    const handleLoad = () => {
      const nextSize = { width: img.naturalWidth, height: img.naturalHeight }
      if (nextSize.width && nextSize.height) {
        setImageSize(nextSize)
        requestAnimationFrame(() => {
          if (fitToScreen(nextSize)) {
            setHasAutoFit(true)
          }
        })
      }
    }
    if (img.complete && img.naturalWidth) {
      handleLoad()
      return
    }
    img.addEventListener('load', handleLoad)
    return () => img.removeEventListener('load', handleLoad)
  }, [imageUrl])

  useEffect(() => {
    const handleResize = () => {
      if (!hasAutoFit) {
        if (fitToScreen(imageSize)) {
          setHasAutoFit(true)
        }
        return
      }
      setOffset(clampOffset({ ...offset }, scale))
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [imageSize, offset, scale, hasAutoFit])

  useEffect(() => {
    if (hasAutoFit) return
    const tryAutoFit = () => {
      if (hasAutoFit) return
      autoFitAttemptsRef.current += 1
      if (fitToScreen(imageSize)) {
        setHasAutoFit(true)
        return
      }
      if (autoFitAttemptsRef.current < 10) {
        requestAnimationFrame(tryAutoFit)
      }
    }
    requestAnimationFrame(tryAutoFit)
  }, [hasAutoFit, imageSize])

  useEffect(() => {
    if (!fitKey || fitKey === lastFitKeyRef.current) return
    lastFitKeyRef.current = fitKey
    requestAnimationFrame(() => {
      fitToScreen(imageSize)
      setHasAutoFit(true)
    })
  }, [fitKey, imageSize])

  useEffect(() => {
    if (hasAutoFit) return
    const container = containerRef.current
    if (!container) return
    const observer = new ResizeObserver(() => {
      if (!hasAutoFit && fitToScreen(imageSize)) {
        setHasAutoFit(true)
      }
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [hasAutoFit, imageSize])

  useEffect(() => {
    const canvas = minimapRef.current
    const img = imageRef.current
    if (!canvas || !img || !imageSize.width || !imageSize.height) return

    const drawMinimap = () => {
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const scaleX = MINI_MAX_WIDTH / imageSize.width
      const scaleY = MINI_MAX_HEIGHT / imageSize.height
      const miniScale = Math.min(scaleX, scaleY)
      const drawWidth = Math.max(1, Math.round(imageSize.width * miniScale))
      const drawHeight = Math.max(1, Math.round(imageSize.height * miniScale))
      canvas.width = drawWidth
      canvas.height = drawHeight
      ctx.clearRect(0, 0, drawWidth, drawHeight)
      ctx.drawImage(img, 0, 0, drawWidth, drawHeight)
      minimapMetaRef.current = {
        scale: miniScale,
        offsetX: 0,
        offsetY: 0,
        width: drawWidth,
        height: drawHeight,
      }
    }

    drawMinimap()
  }, [imageUrl, imageSize])

  const fitToScreen = (size = imageSize) => {
    const container = containerRef.current
    if (!container) return false
    const { clientWidth, clientHeight } = container
    const resolvedSize = size.width && size.height
      ? size
      : imageRef.current
        ? { width: imageRef.current.naturalWidth, height: imageRef.current.naturalHeight }
        : size
    if (!resolvedSize.width || !resolvedSize.height || !clientWidth || !clientHeight) {
      return false
    }
    const scaleX = clientWidth / resolvedSize.width
    const scaleY = clientHeight / resolvedSize.height
    const nextScale = Math.min(scaleX, scaleY, 1)
    setScale(nextScale)
    const nextOffset = {
      x: (clientWidth - resolvedSize.width * nextScale) / 2,
      y: (clientHeight - resolvedSize.height * nextScale) / 2,
    }
    setOffset(clampOffset(nextOffset, nextScale))
    return true
  }

  const resetZoom = () => {
    const container = containerRef.current
    if (!container) return
    const nextScale = 1
    const nextOffset = {
      x: (container.clientWidth - imageSize.width * nextScale) / 2,
      y: (container.clientHeight - imageSize.height * nextScale) / 2,
    }
    setScale(nextScale)
    setOffset(clampOffset(nextOffset, nextScale))
  }

  const clampScale = (value: number) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, value))

  const clampOffset = (nextOffset: { x: number; y: number }, nextScale = scale) => {
    const container = containerRef.current
    if (!container) return nextOffset
    const imageWidth = imageSize.width * nextScale
    const imageHeight = imageSize.height * nextScale
    const { clientWidth, clientHeight } = container

    if (imageWidth <= clientWidth) {
      nextOffset.x = (clientWidth - imageWidth) / 2
    } else {
      const minX = clientWidth - imageWidth
      nextOffset.x = Math.min(0, Math.max(minX, nextOffset.x))
    }

    if (imageHeight <= clientHeight) {
      nextOffset.y = (clientHeight - imageHeight) / 2
    } else {
      const minY = clientHeight - imageHeight
      nextOffset.y = Math.min(0, Math.max(minY, nextOffset.y))
    }

    return nextOffset
  }

  const handleWheel = (event: React.WheelEvent) => {
    event.preventDefault()
    const container = containerRef.current
    if (!container) return
    const rect = container.getBoundingClientRect()
    const cursorX = event.clientX - rect.left
    const cursorY = event.clientY - rect.top
    const zoomDirection = event.deltaY < 0 ? 1.08 : 0.92
    const nextScale = clampScale(scale * zoomDirection)
    const scaleRatio = nextScale / scale

    const nextOffset = {
      x: cursorX - (cursorX - offset.x) * scaleRatio,
      y: cursorY - (cursorY - offset.y) * scaleRatio,
    }
    setScale(nextScale)
    setOffset(clampOffset(nextOffset, nextScale))
  }

  const handlePointerDown = (event: React.PointerEvent) => {
    if (event.button !== 0) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    setActivePointerId(event.pointerId)
    setIsDragging(true)
    setDragStart({ x: event.clientX - offset.x, y: event.clientY - offset.y })
  }

  const handlePointerMove = (event: React.PointerEvent) => {
    if (!isDragging || (activePointerId !== null && event.pointerId !== activePointerId)) return
    const nextOffset = {
      x: event.clientX - dragStart.x,
      y: event.clientY - dragStart.y,
    }
    setOffset(clampOffset(nextOffset))
  }

  const handlePointerUp = (event: React.PointerEvent) => {
    if (activePointerId !== null) {
      event.currentTarget.releasePointerCapture(activePointerId)
    }
    setIsDragging(false)
    setActivePointerId(null)
  }

  const zoomPercent = Number.isFinite(scale) ? Math.round(scale * 100) : 100

  const centerToImagePoint = (imageX: number, imageY: number) => {
    const container = containerRef.current
    if (!container) return
    const nextOffset = {
      x: container.clientWidth / 2 - imageX * scale,
      y: container.clientHeight / 2 - imageY * scale,
    }
    setOffset(clampOffset(nextOffset))
  }

  const handleMinimapClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = minimapRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const clickX = event.clientX - rect.left
    const clickY = event.clientY - rect.top
    const meta = minimapMetaRef.current
    if (!meta.scale) return
    const imageX = (clickX - meta.offsetX) / meta.scale
    const imageY = (clickY - meta.offsetY) / meta.scale
    centerToImagePoint(imageX, imageY)
  }

  const renderMinimapViewport = () => {
    const canvas = minimapRef.current
    const container = containerRef.current
    if (!canvas || !container) return null
    const meta = minimapMetaRef.current
    const imageWidth = imageSize.width
    const imageHeight = imageSize.height
    if (!imageWidth || !imageHeight) return null
    const viewWidth = container.clientWidth / scale
    const viewHeight = container.clientHeight / scale
    const maxViewX = Math.max(0, imageWidth - viewWidth)
    const maxViewY = Math.max(0, imageHeight - viewHeight)
    const viewX = Math.min(maxViewX, Math.max(0, -offset.x / scale))
    const viewY = Math.min(maxViewY, Math.max(0, -offset.y / scale))
    const rectX = viewX * meta.scale
    const rectY = viewY * meta.scale
    const rectWidth = Math.min(imageWidth, viewWidth) * meta.scale
    const rectHeight = Math.min(imageHeight, viewHeight) * meta.scale
    return (
      <div
        className="absolute border-2 border-blue-400/90 shadow-[0_0_0_1px_rgba(0,0,0,0.35)] bg-blue-400/10 pointer-events-none"
        style={{
          left: rectX,
          top: rectY,
          width: rectWidth,
          height: rectHeight,
        }}
      />
    )
  }

  const handleMinimapPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    setIsMinimapDragging(true)
    setMinimapPointerId(event.pointerId)
    handleMinimapClick(event as unknown as React.MouseEvent<HTMLCanvasElement>)
  }

  const handleMinimapPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isMinimapDragging || (minimapPointerId !== null && event.pointerId !== minimapPointerId)) {
      return
    }
    const canvas = minimapRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const moveX = event.clientX - rect.left
    const moveY = event.clientY - rect.top
    const meta = minimapMetaRef.current
    if (!meta.scale) return
    const imageX = (moveX - meta.offsetX) / meta.scale
    const imageY = (moveY - meta.offsetY) / meta.scale
    centerToImagePoint(imageX, imageY)
  }

  const handleMinimapPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (minimapPointerId !== null) {
      event.currentTarget.releasePointerCapture(minimapPointerId)
    }
    setIsMinimapDragging(false)
    setMinimapPointerId(null)
  }

  return (
    <div className="relative h-full w-full bg-[var(--bg-canvas)]">
      <div
        ref={containerRef}
        className={cn(
          'absolute inset-0 overflow-hidden',
          isDragging ? 'cursor-grabbing' : 'cursor-grab'
        )}
        style={{ touchAction: 'none' }}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      >
        <div
          className="absolute top-0 left-0 origin-top-left"
          style={{
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
            width: imageSize.width,
            height: imageSize.height,
          }}
        >
          <img
            ref={imageRef}
            src={imageUrl}
            alt="P&ID results"
            className="block pointer-events-none select-none"
            onLoad={(event) => {
              const target = event.currentTarget
              const nextSize = { width: target.naturalWidth, height: target.naturalHeight }
              if (nextSize.width && nextSize.height) {
                setImageSize(nextSize)
                requestAnimationFrame(() => {
                  if (fitToScreen(nextSize)) {
                    setHasAutoFit(true)
                  }
                })
              }
            }}
          />
          {objects.map((obj) => (
            <div
              key={`${obj.CategoryID}-${obj.ObjectID}-${obj.Index}`}
              className="absolute border-2 border-[var(--accent)] pointer-events-none"
              style={{
                left: obj.Left,
                top: obj.Top,
                width: obj.Width,
                height: obj.Height,
                borderColor: getCategoryColor(obj.Object),
                opacity: Math.max(obj.Score, 0.25),
                boxShadow: selectedObjectKey === objectKey(obj)
                  ? '0 0 0 2px rgba(59, 130, 246, 0.6)'
                  : undefined,
              }}
              title={`${obj.Object} (${Math.round(obj.Score * 100)}%)`}
            />
          ))}
        </div>
      </div>

      <div className="absolute bottom-5 left-1/2 -translate-x-1/2">
        <ZoomControls
          zoomPercent={zoomPercent}
          onZoomIn={() => {
            const container = containerRef.current
            if (!container) return
            const centerX = container.clientWidth / 2
            const centerY = container.clientHeight / 2
            const nextScale = clampScale(scale * 1.1)
            const scaleRatio = nextScale / scale
            const nextOffset = {
              x: centerX - (centerX - offset.x) * scaleRatio,
              y: centerY - (centerY - offset.y) * scaleRatio,
            }
            setScale(nextScale)
            setOffset(clampOffset(nextOffset, nextScale))
          }}
          onZoomOut={() => {
            const container = containerRef.current
            if (!container) return
            const centerX = container.clientWidth / 2
            const centerY = container.clientHeight / 2
            const nextScale = clampScale(scale / 1.1)
            const scaleRatio = nextScale / scale
            const nextOffset = {
              x: centerX - (centerX - offset.x) * scaleRatio,
              y: centerY - (centerY - offset.y) * scaleRatio,
            }
            setScale(nextScale)
            setOffset(clampOffset(nextOffset, nextScale))
          }}
          onReset={resetZoom}
          onFit={fitToScreen}
        />
      </div>

      <div className="absolute bottom-5 right-5 bg-black/40 border border-white/10 rounded-lg p-2 backdrop-blur">
        <div className="relative">
          <canvas
            ref={minimapRef}
            onClick={handleMinimapClick}
            onPointerDown={handleMinimapPointerDown}
            onPointerMove={handleMinimapPointerMove}
            onPointerUp={handleMinimapPointerUp}
            onPointerLeave={handleMinimapPointerUp}
            className="block rounded-md cursor-pointer"
            aria-label="Minimap"
          />
          {renderMinimapViewport()}
        </div>
      </div>
    </div>
  )
}
