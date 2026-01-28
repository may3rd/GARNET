import { forwardRef, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { ForwardedRef } from 'react'
import type { DetectedObject } from '@/types'
import { getCategoryColor } from '@/lib/categoryColors'
import { objectKey } from '@/lib/objectKey'
import { ZoomControls } from '@/components/ZoomControls'
import { cn } from '@/lib/utils'
import { CheckCircle2, ChevronLeft, ChevronRight, Loader2, Trash2, XCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

type CanvasViewProps = {
  imageUrl: string
  objects: DetectedObject[]
  selectedObjectKey: string | null
  selectedObject: DetectedObject | null
  reviewStatus: Record<string, 'accepted' | 'rejected'>
  onSelectObject: (key: string | null) => void
  onSetReviewStatus: (key: string, status: 'accepted' | 'rejected' | null) => void
  isEditing: boolean
  editDraft: {
    Object: string
    Left: number
    Top: number
    Width: number
    Height: number
    Text: string
  } | null
  isCreating: boolean
  createDraft: {
    Object: string
    Left: number
    Top: number
    Width: number
    Height: number
    Text: string
  } | null
  onCreateDraftChange: (draft: NonNullable<CanvasViewProps['createDraft']> | null) => void
  onStartEdit: (obj: DetectedObject) => void
  onCancelEdit: () => void
  onChangeEdit: (field: keyof NonNullable<CanvasViewProps['editDraft']>, value: string) => void
  onReplaceEditDraft: (draft: NonNullable<CanvasViewProps['editDraft']>) => void
  onSaveEdit: () => void
  onDeleteSelected: () => void
  onNavigatePrevious: () => void
  onNavigateNext: () => void
  fitKey?: string
}

export type CanvasViewHandle = {
  zoomIn: () => void
  zoomOut: () => void
  resetZoom: () => void
  fitToScreen: () => void
  centerOnObject: (obj: DetectedObject) => void
}

const MIN_ZOOM = 0.2
const MAX_ZOOM = 4
const MINI_MAX_WIDTH = 180
const MINI_MAX_HEIGHT = 120
const RESIZE_HANDLE_THRESHOLD = 30
const MIN_OBJECT_SIZE = 4
const DRAG_DETECTION_THRESHOLD = 3
const AUTO_FIT_MAX_ATTEMPTS = 10

export const CanvasView = forwardRef(function CanvasView(
  {
    imageUrl,
    objects,
    selectedObjectKey,
    selectedObject,
    reviewStatus,
    onSelectObject,
    onSetReviewStatus,
    isEditing,
    editDraft,
    isCreating,
    createDraft,
    onCreateDraftChange,
    onStartEdit,
    onCancelEdit,
    onChangeEdit,
    onReplaceEditDraft,
    onSaveEdit,
    onDeleteSelected,
    onNavigatePrevious,
    onNavigateNext,
    fitKey,
  }: CanvasViewProps,
  ref: ForwardedRef<CanvasViewHandle>
) {
  const containerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const minimapRef = useRef<HTMLCanvasElement>(null)
  const minimapMetaRef = useRef({ scale: 1, offsetX: 0, offsetY: 0, width: 0, height: 0 })
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [activePointerId, setActivePointerId] = useState<number | null>(null)
  const [isMinimapDragging, setIsMinimapDragging] = useState(false)
  const [minimapPointerId, setMinimapPointerId] = useState<number | null>(null)
  const [hasAutoFit, setHasAutoFit] = useState(false)
  const [isImageLoading, setIsImageLoading] = useState(true)
  const [cardSize, setCardSize] = useState({ width: 240, height: 150 })
  const cardRef = useRef<HTMLDivElement>(null)
  const autoFitAttemptsRef = useRef(0)
  const lastFitKeyRef = useRef<string | undefined>(undefined)
  const dragMovedRef = useRef(false)
  const pointerStartRef = useRef({ x: 0, y: 0 })
  const resizePointerIdRef = useRef<number | null>(null)
  const createStartRef = useRef<{ x: number; y: number } | null>(null)
  const createPointerIdRef = useRef<number | null>(null)
  const lastContainerSizeRef = useRef<{ width: number; height: number } | null>(null)
  const resizeRafRef = useRef<number | null>(null)
  const resizeTimeoutRef = useRef<number | null>(null)
  const [isLayoutResizing, setIsLayoutResizing] = useState(false)
  const scaleRef = useRef(scale)
  const offsetRef = useRef(offset)
  const imageSizeRef = useRef(imageSize)
  const hasAutoFitRef = useRef(hasAutoFit)
  const interactionRef = useRef({
    isDragging: false,
    isMinimapDragging: false,
    isCreating: false,
    isResizingEdit: false,
  })
  const [resizeState, setResizeState] = useState<{
    handle: 'n' | 's' | 'e' | 'w' | 'nw' | 'ne' | 'sw' | 'se' | 'move'
    startX: number
    startY: number
    start: NonNullable<CanvasViewProps['editDraft']>
  } | null>(null)
  const [editCursor, setEditCursor] = useState<string>('grab')

  useEffect(() => {
    scaleRef.current = scale
  }, [scale])

  useEffect(() => {
    offsetRef.current = offset
  }, [offset])

  useEffect(() => {
    imageSizeRef.current = imageSize
  }, [imageSize])

  useEffect(() => {
    hasAutoFitRef.current = hasAutoFit
  }, [hasAutoFit])

  useEffect(() => {
    interactionRef.current = {
      isDragging,
      isMinimapDragging,
      isCreating,
      isResizingEdit: Boolean(resizeState),
    }
  }, [isDragging, isMinimapDragging, isCreating, resizeState])

  useEffect(() => {
    setHasAutoFit(false)
    setIsImageLoading(true)
    autoFitAttemptsRef.current = 0
    const img = imageRef.current
    if (!img) return
    const handleLoad = () => {
      setIsImageLoading(false)
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
    if (hasAutoFit) return
    const tryAutoFit = () => {
      if (hasAutoFit) return
      autoFitAttemptsRef.current += 1
      if (fitToScreen(imageSize)) {
        setHasAutoFit(true)
        return
      }
      if (autoFitAttemptsRef.current < AUTO_FIT_MAX_ATTEMPTS) {
        requestAnimationFrame(tryAutoFit)
      }
    }
    requestAnimationFrame(tryAutoFit)
  }, [hasAutoFit, imageSize])

  useEffect(() => {
    if (!fitKey || fitKey === lastFitKeyRef.current) return
    if (imageSize.width === 0 || imageSize.height === 0) return

    lastFitKeyRef.current = fitKey
    requestAnimationFrame(() => {
      if (fitToScreen(imageSize)) {
        setHasAutoFit(true)
      }
    })
  }, [fitKey, imageSize])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const observer = new ResizeObserver(() => {
      if (resizeRafRef.current !== null) return
      resizeRafRef.current = window.requestAnimationFrame(() => {
        resizeRafRef.current = null
        const resolvedContainer = containerRef.current
        if (!resolvedContainer) return

        const width = resolvedContainer.clientWidth
        const height = resolvedContainer.clientHeight
        if (!width || !height) return

        const prev = lastContainerSizeRef.current
        lastContainerSizeRef.current = { width, height }
        if (!prev) return

        const { width: imageWidth, height: imageHeight } = imageSizeRef.current
        if (!imageWidth || !imageHeight) return

        if (!hasAutoFitRef.current) {
          if (fitToScreen(imageSizeRef.current)) {
            setHasAutoFit(true)
          }
          return
        }

        const interacting = interactionRef.current.isDragging
          || interactionRef.current.isMinimapDragging
          || interactionRef.current.isCreating
          || interactionRef.current.isResizingEdit
        if (interacting) {
          setOffset(clampOffset({ ...offsetRef.current }, scaleRef.current))
          return
        }

        const currentScale = scaleRef.current
        const currentOffset = offsetRef.current

        const centerImgX = (prev.width / 2 - currentOffset.x) / currentScale
        const centerImgY = (prev.height / 2 - currentOffset.y) / currentScale
        const nextOffset = {
          x: width / 2 - centerImgX * currentScale,
          y: height / 2 - centerImgY * currentScale,
        }

        setIsLayoutResizing(true)
        if (resizeTimeoutRef.current) {
          window.clearTimeout(resizeTimeoutRef.current)
        }
        resizeTimeoutRef.current = window.setTimeout(() => setIsLayoutResizing(false), 220)

        setOffset(clampOffset(nextOffset, currentScale))
      })
    })
    observer.observe(container)
    return () => {
      observer.disconnect()
      if (resizeRafRef.current !== null) {
        window.cancelAnimationFrame(resizeRafRef.current)
        resizeRafRef.current = null
      }
      if (resizeTimeoutRef.current) {
        window.clearTimeout(resizeTimeoutRef.current)
        resizeTimeoutRef.current = null
      }
    }
  }, [])

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

  useLayoutEffect(() => {
    if (!cardRef.current) return
    const rect = cardRef.current.getBoundingClientRect()
    if (rect.width && rect.height) {
      setCardSize({ width: rect.width, height: rect.height })
    }
  }, [selectedObjectKey, selectedObject])

  const getImagePoint = (clientX: number, clientY: number) => {
    const container = containerRef.current
    if (!container) return null
    const rect = container.getBoundingClientRect()
    return {
      x: (clientX - rect.left - offset.x) / scale,
      y: (clientY - rect.top - offset.y) / scale,
    }
  }

  const buildCreateDraft = (start: { x: number; y: number }, current: { x: number; y: number }) => {
    const left = Math.min(start.x, current.x)
    const top = Math.min(start.y, current.y)
    const width = Math.max(MIN_OBJECT_SIZE, Math.abs(current.x - start.x))
    const height = Math.max(MIN_OBJECT_SIZE, Math.abs(current.y - start.y))
    return {
      Object: createDraft?.Object || 'custom',
      Text: createDraft?.Text || '',
      Left: left,
      Top: top,
      Width: width,
      Height: height,
    }
  }

  const applyResize = (
    start: NonNullable<CanvasViewProps['editDraft']>,
    handle: 'n' | 's' | 'e' | 'w' | 'nw' | 'ne' | 'sw' | 'se' | 'move',
    dx: number,
    dy: number
  ) => {
    let left = start.Left
    let top = start.Top
    let width = start.Width
    let height = start.Height

    const applyWest = () => {
      const nextWidth = Math.max(MIN_OBJECT_SIZE, width - dx)
      left += width - nextWidth
      width = nextWidth
    }
    const applyEast = () => {
      width = Math.max(MIN_OBJECT_SIZE, width + dx)
    }
    const applyNorth = () => {
      const nextHeight = Math.max(MIN_OBJECT_SIZE, height - dy)
      top += height - nextHeight
      height = nextHeight
    }
    const applySouth = () => {
      height = Math.max(MIN_OBJECT_SIZE, height + dy)
    }

    switch (handle) {
      case 'move':
        left += dx
        top += dy
        break
      case 'n':
        applyNorth()
        break
      case 's':
        applySouth()
        break
      case 'w':
        applyWest()
        break
      case 'e':
        applyEast()
        break
      case 'nw':
        applyNorth()
        applyWest()
        break
      case 'ne':
        applyNorth()
        applyEast()
        break
      case 'sw':
        applySouth()
        applyWest()
        break
      case 'se':
        applySouth()
        applyEast()
        break
      default:
        break
    }

    return { ...start, Left: left, Top: top, Width: width, Height: height }
  }

  useEffect(() => {
    if (!resizeState) return

    const handlePointerMove = (event: PointerEvent) => {
      if (resizePointerIdRef.current !== null && event.pointerId !== resizePointerIdRef.current) {
        return
      }
      const point = getImagePoint(event.clientX, event.clientY)
      if (!point) return
      const dx = point.x - resizeState.startX
      const dy = point.y - resizeState.startY
      const nextDraft = applyResize(resizeState.start, resizeState.handle, dx, dy)
      onReplaceEditDraft(nextDraft)
    }

    const handlePointerUp = (event: PointerEvent) => {
      if (resizePointerIdRef.current !== null && event.pointerId !== resizePointerIdRef.current) {
        return
      }
      resizePointerIdRef.current = null
      setResizeState(null)
      document.body.style.cursor = ''
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp)
    return () => {
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
    }
  }, [resizeState, scale, offset, onReplaceEditDraft])

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
    const size = imageSizeRef.current
    const imageWidth = size.width * nextScale
    const imageHeight = size.height * nextScale
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

  const zoomAt = (nextScale: number, centerX: number, centerY: number) => {
    const clamped = clampScale(nextScale)
    const scaleRatio = clamped / scale
    const nextOffset = {
      x: centerX - (centerX - offset.x) * scaleRatio,
      y: centerY - (centerY - offset.y) * scaleRatio,
    }
    setScale(clamped)
    setOffset(clampOffset(nextOffset, clamped))
  }

  const handleWheel = (event: WheelEvent) => {
    event.preventDefault()
    const container = containerRef.current
    if (!container) return
    const rect = container.getBoundingClientRect()
    const cursorX = event.clientX - rect.left
    const cursorY = event.clientY - rect.top
    const zoomDirection = event.deltaY < 0 ? 1.08 : 0.92
    zoomAt(scale * zoomDirection, cursorX, cursorY)
  }

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const wheelListener = (event: WheelEvent) => handleWheel(event)
    container.addEventListener('wheel', wheelListener, { passive: false })
    return () => {
      container.removeEventListener('wheel', wheelListener)
    }
  }, [scale, offset, resizeState])

  const handlePointerDown = (event: React.PointerEvent) => {
    if (event.button !== 0) return
    if (resizeState) return
    if (isCreating) {
      event.preventDefault()
      event.stopPropagation()
      const point = getImagePoint(event.clientX, event.clientY)
      if (!point) return
      createPointerIdRef.current = event.pointerId
      createStartRef.current = point
      event.currentTarget.setPointerCapture(event.pointerId)
      onCreateDraftChange(buildCreateDraft(point, point))
      return
    }
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    setActivePointerId(event.pointerId)
    setIsDragging(true)
    dragMovedRef.current = false
    pointerStartRef.current = { x: event.clientX, y: event.clientY }
    setDragStart({ x: event.clientX - offset.x, y: event.clientY - offset.y })
  }

  const handlePointerMove = (event: React.PointerEvent) => {
    if (resizeState) return
    if (isCreating && createStartRef.current) {
      if (createPointerIdRef.current !== null && event.pointerId !== createPointerIdRef.current) {
        return
      }
      const point = getImagePoint(event.clientX, event.clientY)
      if (!point) return
      onCreateDraftChange(buildCreateDraft(createStartRef.current, point))
      return
    }
    if (!isDragging || (activePointerId !== null && event.pointerId !== activePointerId)) return
    if (!dragMovedRef.current) {
      const deltaX = Math.abs(event.clientX - pointerStartRef.current.x)
      const deltaY = Math.abs(event.clientY - pointerStartRef.current.y)
      if (deltaX > DRAG_DETECTION_THRESHOLD || deltaY > DRAG_DETECTION_THRESHOLD) {
        dragMovedRef.current = true
      }
    }
    const nextOffset = {
      x: event.clientX - dragStart.x,
      y: event.clientY - dragStart.y,
    }
    setOffset(clampOffset(nextOffset))
  }

  const handlePointerUp = (event: React.PointerEvent) => {
    if (resizeState) return
    if (isCreating) {
      if (createPointerIdRef.current !== null) {
        event.currentTarget.releasePointerCapture(createPointerIdRef.current)
      }
      createPointerIdRef.current = null
      createStartRef.current = null
      return
    }
    if (activePointerId !== null) {
      event.currentTarget.releasePointerCapture(activePointerId)
    }
    setIsDragging(false)
    setActivePointerId(null)
    if (!dragMovedRef.current) {
      onSelectObject(null)
    }
  }

  const handlePointerLeave = (event: React.PointerEvent) => {
    if (resizeState) return
    if (isCreating) {
      if (createPointerIdRef.current !== null) {
        event.currentTarget.releasePointerCapture(createPointerIdRef.current)
      }
      createPointerIdRef.current = null
      createStartRef.current = null
      return
    }
    if (!isDragging) return
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

  const centerOnObject = (obj: DetectedObject) => {
    centerToImagePoint(obj.Left + obj.Width / 2, obj.Top + obj.Height / 2)
  }

  const zoomIn = () => {
    const container = containerRef.current
    if (!container) return
    zoomAt(scale * 1.1, container.clientWidth / 2, container.clientHeight / 2)
  }

  const zoomOut = () => {
    const container = containerRef.current
    if (!container) return
    zoomAt(scale / 1.1, container.clientWidth / 2, container.clientHeight / 2)
  }

  const zoomToPercent = (percent: number) => {
    const container = containerRef.current
    if (!container) return
    const nextScale = clampScale(percent / 100)
    zoomAt(nextScale, container.clientWidth / 2, container.clientHeight / 2)
  }

  useImperativeHandle(ref, () => ({
    zoomIn,
    zoomOut,
    resetZoom,
    fitToScreen: () => {
      fitToScreen(imageSize)
    },
    centerOnObject,
  }), [scale, imageSize, resetZoom, fitToScreen])

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

  const selectionCardStyle = (() => {
    if (!selectedObject || !containerRef.current) return null
    const container = containerRef.current
    const baseX = offset.x + (selectedObject.Left + selectedObject.Width) * scale + 12
    const baseY = offset.y + selectedObject.Top * scale
    const clampedX = Math.min(
      container.clientWidth - cardSize.width - 12,
      Math.max(12, baseX)
    )
    const clampedY = Math.min(
      container.clientHeight - cardSize.height - 12,
      Math.max(12, baseY)
    )
    return { left: clampedX, top: clampedY }
  })()

  return (
    <div className="relative h-full w-full bg-[var(--bg-canvas)]">
      {isImageLoading && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-3 text-[var(--text-secondary)]">
            <Loader2 className="h-8 w-8 animate-spin" />
            <span className="text-sm">Loading image...</span>
          </div>
        </div>
      )}
      <div
        ref={containerRef}
        className={cn(
          'absolute inset-0 overflow-hidden',
          isCreating ? 'cursor-crosshair' : isDragging ? 'cursor-grabbing' : 'cursor-grab'
        )}
        style={{ touchAction: 'none' }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerLeave}
      >
        <div
          className={cn(
            'absolute top-0 left-0 origin-top-left',
            isLayoutResizing && !isDragging && !isMinimapDragging && !isCreating && !resizeState && 'transition-transform duration-200 ease-out'
          )}
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
          {useMemo(() => objects.map((obj) => {
            const key = objectKey(obj)
            const status = reviewStatus[key]
            const opacity = status === 'accepted'
              ? 1
              : status === 'rejected'
                ? 0.15
                : 0.5
            const adjusted = isEditing && editDraft && selectedObjectKey === key
              ? { ...obj, ...editDraft }
              : obj
            return (
              <div
                key={key}
                className={cn(
                  'absolute border-2',
                  isEditing || isCreating ? 'pointer-events-none' : 'pointer-events-auto',
                  'transition-shadow',
                  selectedObjectKey === key && 'ring-2 ring-white/80',
                  status === 'accepted' && 'shadow-[0_0_0_1px_rgba(22,163,74,0.6)]'
                )}
                style={{
                  left: adjusted.Left,
                  top: adjusted.Top,
                  width: adjusted.Width,
                  height: adjusted.Height,
                  borderColor: getCategoryColor(adjusted.Object),
                  opacity,
                  borderWidth: selectedObjectKey === key ? 3 : 2,
                  backgroundColor: selectedObjectKey === key
                    ? 'rgba(59, 130, 246, 0.08)'
                    : status === 'rejected'
                      ? 'rgba(239, 68, 68, 0.06)'
                      : status === 'accepted'
                        ? 'rgba(34, 197, 94, 0.0)'
                        : 'rgba(59, 130, 246, 0.5)',
                  boxShadow: selectedObjectKey === key
                    ? '0 0 0 3px rgba(59, 130, 246, 0.8), 0 0 12px rgba(59, 130, 246, 0.6)'
                    : undefined,
                  borderStyle: status === 'rejected' ? 'dashed' : 'solid',
                }}
                title={`${adjusted.Object} (${Math.round(obj.Score * 100)}%)`}
                onPointerDown={isEditing ? undefined : (event) => event.stopPropagation()}
                onPointerUp={isEditing ? undefined : (event) => event.stopPropagation()}
                onClick={isEditing ? undefined : (event) => {
                  event.stopPropagation()
                  onSelectObject(key)
                }}
              />
            )
          }), [objects, reviewStatus, isEditing, editDraft, selectedObjectKey, isCreating, onSelectObject])}
          {isCreating && createDraft && (
            <div
              className="absolute border-2 border-dashed border-blue-500/80 bg-blue-500/10 pointer-events-none"
              style={{
                left: createDraft.Left,
                top: createDraft.Top,
                width: createDraft.Width,
                height: createDraft.Height,
              }}
            />
          )}
        </div>
        {isEditing && editDraft && selectedObjectKey && (
          <div
            className="absolute z-30"
            style={{
              left: offset.x + editDraft.Left * scale,
              top: offset.y + editDraft.Top * scale,
              width: editDraft.Width * scale,
              height: editDraft.Height * scale,
            }}
          >
            <div
              className="absolute inset-0 border-2 border-blue-500/90 bg-blue-500/10"
              style={{ cursor: editCursor }}
              onPointerDown={(event) => {
                event.preventDefault()
                event.stopPropagation()
                const point = getImagePoint(event.clientX, event.clientY)
                if (!point) return
                resizePointerIdRef.current = event.pointerId
                setResizeState({
                  handle: 'move',
                  startX: point.x,
                  startY: point.y,
                  start: editDraft,
                })
                document.body.style.cursor = 'grabbing'
              }}
              onMouseMove={(event) => {
                const rect = event.currentTarget.getBoundingClientRect()
                const x = event.clientX - rect.left
                const y = event.clientY - rect.top
                const width = rect.width
                const height = rect.height
                const threshold = RESIZE_HANDLE_THRESHOLD
                let cursor = 'grab'
                if (x < threshold && y < threshold) cursor = 'nw-resize'
                else if (x > width - threshold && y < threshold) cursor = 'ne-resize'
                else if (x < threshold && y > height - threshold) cursor = 'sw-resize'
                else if (x > width - threshold && y > height - threshold) cursor = 'se-resize'
                else if (x < threshold) cursor = 'w-resize'
                else if (x > width - threshold) cursor = 'e-resize'
                else if (y < threshold) cursor = 'n-resize'
                else if (y > height - threshold) cursor = 's-resize'
                setEditCursor(cursor)
              }}
              onMouseLeave={() => setEditCursor('grab')}
            />
            {([
              { key: 'nw', left: -5, top: -5, cursorClass: 'cursor-nwse-resize' },
              { key: 'n', left: '50%', top: -5, cursorClass: 'cursor-ns-resize', transform: 'translateX(-50%)' },
              { key: 'ne', right: -5, top: -5, cursorClass: 'cursor-nesw-resize' },
              { key: 'w', left: -5, top: '50%', cursorClass: 'cursor-ew-resize', transform: 'translateY(-50%)' },
              { key: 'e', right: -5, top: '50%', cursorClass: 'cursor-ew-resize', transform: 'translateY(-50%)' },
              { key: 'sw', left: -5, bottom: -5, cursorClass: 'cursor-nesw-resize' },
              { key: 's', left: '50%', bottom: -5, cursorClass: 'cursor-ns-resize', transform: 'translateX(-50%)' },
              { key: 'se', right: -5, bottom: -5, cursorClass: 'cursor-nwse-resize' },
            ] as const).map((handle) => (
              <div
                key={handle.key}
                className={cn(
                  'absolute h-2.5 w-2.5 rounded-sm border border-blue-600 bg-white z-[60]',
                  handle.cursorClass
                )}
                style={{
                  ...('left' in handle ? { left: handle.left } : {}),
                  ...('top' in handle ? { top: handle.top } : {}),
                  ...('right' in handle ? { right: handle.right } : {}),
                  ...('bottom' in handle ? { bottom: handle.bottom } : {}),
                  ...('transform' in handle ? { transform: handle.transform } : {}),
                }}
                onPointerDown={(event) => {
                  event.preventDefault()
                  event.stopPropagation()
                  const point = getImagePoint(event.clientX, event.clientY)
                  if (!point) return
                  resizePointerIdRef.current = event.pointerId
                  setResizeState({
                    handle: handle.key,
                    startX: point.x,
                    startY: point.y,
                    start: editDraft,
                  })
                  document.body.style.cursor = handle.cursorClass.replace('cursor-', '')
                }}
              />
            ))}
          </div>
        )}
      </div>

      {selectedObject && selectionCardStyle && (
        <div
          ref={cardRef}
          className="absolute z-50 w-60 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-secondary)] p-3 shadow-lg"
          style={selectionCardStyle}
        >
          <div className="text-xs font-semibold uppercase tracking-wide text-[var(--text-secondary)]">
            Selected object
          </div>
          {!isEditing ? (
            <>
              <div className="mt-1 flex items-center gap-2 flex-wrap">
                <span className="text-sm font-semibold leading-tight">{selectedObject.Object}</span>
                {reviewStatus[objectKey(selectedObject)] === 'accepted' && (
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-[var(--success)]/10 text-[var(--success)] text-[10px] font-bold uppercase tracking-tight">
                    <CheckCircle2 className="h-3 w-3" />
                    Accepted
                  </span>
                )}
                {reviewStatus[objectKey(selectedObject)] === 'rejected' && (
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-[var(--danger)]/10 text-[var(--danger)] text-[10px] font-bold uppercase tracking-tight">
                    <XCircle className="h-3 w-3" />
                    Rejected
                  </span>
                )}
              </div>
              <div className="text-xs text-[var(--text-secondary)]">
                Confidence {Math.round(selectedObject.Score * 100)}%
              </div>
              {selectedObject.Text && (
                <div className="mt-2 text-xs text-[var(--text-secondary)]">
                  OCR: <span className="text-[var(--text-primary)]">{selectedObject.Text}</span>
                </div>
              )}
              <div className="mt-3 flex items-center justify-between">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onNavigatePrevious}
                  title="Previous (←)"
                  aria-label="Previous object"
                  className="h-11 w-11"
                >
                  <ChevronLeft className="h-5 w-5" />
                </Button>
                <span className="text-xs text-[var(--text-secondary)]">Navigate</span>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onNavigateNext}
                  title="Next (→)"
                  aria-label="Next object"
                  className="h-11 w-11"
                >
                  <ChevronRight className="h-5 w-5" />
                </Button>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => {
                    onSetReviewStatus(objectKey(selectedObject), 'accepted')
                  }}
                  className={cn(
                    'w-full px-2.5 py-1.5 rounded-md text-xs font-semibold transition-all',
                    reviewStatus[objectKey(selectedObject)] === 'accepted'
                      ? 'bg-[var(--success)] text-white hover:brightness-95'
                      : 'bg-[var(--bg-primary)] border border-[var(--border-muted)] text-[var(--text-secondary)] hover:border-[var(--success)] hover:text-[var(--success)] hover:bg-[var(--success)]/5'
                  )}
                >
                  Accept
                </button>
                <button
                  type="button"
                  onClick={() => {
                    onSetReviewStatus(objectKey(selectedObject), 'rejected')
                  }}
                  className={cn(
                    'w-full px-2.5 py-1.5 rounded-md text-xs font-semibold transition-all',
                    reviewStatus[objectKey(selectedObject)] === 'rejected'
                      ? 'bg-[var(--danger)] text-white hover:brightness-95'
                      : 'bg-[var(--bg-primary)] border border-[var(--border-muted)] text-[var(--text-secondary)] hover:border-[var(--danger)] hover:text-[var(--danger)] hover:bg-[var(--danger)]/5'
                  )}
                >
                  Reject
                </button>
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full"
                  onClick={() => onStartEdit(selectedObject)}
                >
                  Edit
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full hover:border-[var(--danger)] hover:text-[var(--danger)]"
                  onClick={onDeleteSelected}
                  title="Delete object"
                >
                  <Trash2 className="h-3.5 w-3.5 mr-1" />
                  Delete
                </Button>
              </div>
            </>
          ) : (
            <>
              <div className="mt-2 space-y-2 text-xs">
                <label className="block">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--text-secondary)]">Label</span>
                  <Input
                    value={editDraft?.Object ?? ''}
                    onChange={(event) => onChangeEdit('Object', event.target.value)}
                    className="mt-1 h-7 text-xs"
                  />
                </label>
                <label className="block">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--text-secondary)]">OCR text</span>
                  <Input
                    value={editDraft?.Text ?? ''}
                    onChange={(event) => onChangeEdit('Text', event.target.value)}
                    className="mt-1 h-7 text-xs"
                  />
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {(['Left', 'Top', 'Width', 'Height'] as const).map((field) => (
                    <label key={field} className="block">
                      <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--text-secondary)]">
                        {field}
                      </span>
                      <Input
                        value={editDraft ? String(editDraft[field]) : ''}
                        onChange={(event) => onChangeEdit(field, event.target.value)}
                        className="mt-1 h-7 text-xs"
                      />
                    </label>
                  ))}
                </div>
              </div>
              <div className="mt-3 flex items-center gap-2">
                <Button size="sm" onClick={onSaveEdit}>
                  Save
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onCancelEdit}
                  className="hover:border-[var(--danger)] hover:text-[var(--danger)]"
                >
                  Cancel
                </Button>
              </div>
            </>
          )}
        </div>
      )}

      <div className="absolute bottom-5 left-1/2 -translate-x-1/2">
        <ZoomControls
          zoomPercent={zoomPercent}
          onZoomIn={zoomIn}
          onZoomOut={zoomOut}
          onReset={resetZoom}
          onFit={fitToScreen}
          onZoomTo={zoomToPercent}
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
})
