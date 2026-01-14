/**
 * Canvas component - main P&ID viewer with bounding boxes
 * Matches script.js implementation logic for coordinate alignment
 */
import { useEffect, useRef, useCallback } from 'react'
import { Canvas as FabricCanvas, FabricImage, Rect, FabricText, Group } from 'fabric'
import { useDetectionStore } from '@/stores/detectionStore'
import { useCanvasStore } from '@/stores/canvasStore'
import { useUIStore } from '@/stores/uiStore'
import { CATEGORY_COLORS } from '@/lib/constants'
import { ZoomControls } from './ZoomControls'

// Store image dimensions globally
let imageWidth = 100
let imageHeight = 100

export function Canvas() {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const fabricRef = useRef<FabricCanvas | null>(null)

    const { objects, imageUrl, selectedIds, selectObject } = useDetectionStore()
    const { zoom, setZoom, setPan } = useCanvasStore()
    const { confidenceFilter, hiddenCategories } = useUIStore()

    // Calculate minimum zoom to fit image in container
    const getMinZoom = useCallback(() => {
        const container = containerRef.current
        if (!container) return 0.1
        return Math.min(
            container.clientWidth / imageWidth,
            container.clientHeight / imageHeight
        )
    }, [])

    // Validate image position to keep it within canvas bounds
    const validateImagePosition = useCallback((x?: number, y?: number) => {
        const canvas = fabricRef.current
        const container = containerRef.current
        if (!canvas || !container) return

        const vpt = canvas.viewportTransform
        if (!vpt) return

        const currentZoom = canvas.getZoom()
        const offsetWidth = container.clientWidth - imageWidth * currentZoom
        const offsetHeight = container.clientHeight - imageHeight * currentZoom

        const minX = offsetWidth > 0 ? offsetWidth / 2 : offsetWidth
        const maxX = offsetWidth > 0 ? offsetWidth / 2 : 0
        const minY = offsetHeight > 0 ? offsetHeight / 2 : offsetHeight
        const maxY = offsetHeight > 0 ? offsetHeight / 2 : 0

        const newX = x ?? vpt[4]
        const newY = y ?? vpt[5]

        vpt[4] = Math.max(Math.min(newX, maxX), minX)
        vpt[5] = Math.max(Math.min(newY, maxY), minY)

        canvas.setViewportTransform(vpt)
        canvas.renderAll()
    }, [])

    const zoomTo = useCallback((newZoom: number) => {
        const canvas = fabricRef.current
        const container = containerRef.current
        if (!canvas || !container) return

        const minZoom = getMinZoom()
        const maxZoom = 5.0

        newZoom = Math.min(newZoom, maxZoom)
        newZoom = Math.max(newZoom, minZoom)

        const centerX = container.clientWidth / 2
        const centerY = container.clientHeight / 2
        canvas.zoomToPoint({ x: centerX, y: centerY }, newZoom)

        validateImagePosition()
        setZoom(newZoom)
    }, [getMinZoom, validateImagePosition, setZoom])

    // Initialize canvas
    useEffect(() => {
        if (!canvasRef.current || !containerRef.current) return

        const container = containerRef.current
        const canvas = new FabricCanvas(canvasRef.current, {
            backgroundColor: '#1e293b',
            selection: false,
            preserveObjectStacking: true,
        })

        canvas.setDimensions({
            width: container.clientWidth,
            height: container.clientHeight,
        })

        fabricRef.current = canvas

        const resizeObserver = new ResizeObserver(() => {
            canvas.setDimensions({
                width: container.clientWidth,
                height: container.clientHeight,
            })
            canvas.renderAll()
        })
        resizeObserver.observe(container)

        return () => {
            resizeObserver.disconnect()
            canvas.dispose()
        }
    }, [])

    // Load image and add bounding boxes
    useEffect(() => {
        const canvas = fabricRef.current
        const container = containerRef.current
        if (!canvas || !imageUrl || !container) return

        // Clear canvas
        canvas.clear()
        canvas.backgroundColor = '#1e293b'
        canvas.setViewportTransform([1, 0, 0, 1, 0, 0])

        FabricImage.fromURL(imageUrl, { crossOrigin: 'anonymous' }).then((img) => {
            imageWidth = img.width || 100
            imageHeight = img.height || 100
            console.log('Canvas: Loaded image dimensions:', imageWidth, imageHeight)

            // Add image at Standard Origin (Top-Left) at (0,0)
            // This matches generic Fabric.js behavior and script.js logic
            img.set({
                originX: 'left',
                originY: 'top',
                left: 0,
                top: 0,
                selectable: false,
                evented: false,
            })
            canvas.add(img)

            const visibleObjects = objects.filter(
                (obj) =>
                    obj.confidence >= confidenceFilter &&
                    !hiddenCategories.has(obj.categoryId)
            )

            const boxes: Rect[] = []
            visibleObjects.forEach((obj) => {
                const isSelected = selectedIds.has(obj.id)
                const color = CATEGORY_COLORS[obj.categoryId] || '#ffffff'
                console.log('Canvas: Adding bbox for object', obj.id, ':', obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height)

                const rect = new Rect({
                    left: obj.bbox.x,
                    top: obj.bbox.y,
                    width: obj.bbox.width,
                    height: obj.bbox.height,
                    fill: 'rgba(0, 76, 255, 0.15)',
                    stroke: color,
                    strokeWidth: 3,
                    selectable: false,
                    opacity: 1.0,
                })
                    ; (rect as any).objectId = obj.id
                boxes.push(rect)

                // Labels added separately (not in group) to ensure visibility
                if (isSelected && obj.ocrText) {
                    const label = new FabricText(obj.ocrText, {
                        left: obj.bbox.x,
                        top: obj.bbox.y - 20,
                        fontSize: 14,
                        fill: color,
                        backgroundColor: 'rgba(0, 0, 0, 0.7)',
                        selectable: false,
                        evented: false,
                    })
                    canvas.add(label)
                }
            })

            // Add boxes as a Group to match script.js coordinate handling
            if (boxes.length > 0) {
                const group = new Group(boxes, {
                    selectable: false,
                    evented: true,
                    subTargetCheck: true,
                    interactive: true
                })
                canvas.add(group)
            }

            // Calculations for zoom
            const minZoom = Math.min(
                container.clientWidth / imageWidth,
                container.clientHeight / imageHeight
            )

            // Initial zoom
            zoomTo(minZoom)
            console.log('Canvas: Initial zoom set to:', minZoom, 'canvas zoom:', canvas.getZoom(), 'viewport transform:', canvas.viewportTransform)

            canvas.renderAll()
        }).catch((error) => {
            console.error('Failed to load image:', error)
        })
    }, [imageUrl, objects, selectedIds, confidenceFilter, hiddenCategories, zoomTo])

    // Event listeners
    useEffect(() => {
        const canvas = fabricRef.current
        if (!canvas) return

        const handleWheel = (opt: { e: WheelEvent }) => {
            const delta = opt.e.deltaY
            let newZoom = canvas.getZoom()
            newZoom -= (delta / 100) * 0.1

            const minZoom = getMinZoom()
            const maxZoom = 5.0
            newZoom = Math.min(newZoom, maxZoom)
            newZoom = Math.max(newZoom, minZoom)

            canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, newZoom)
            validateImagePosition()
            setZoom(newZoom)
            opt.e.preventDefault()
            opt.e.stopPropagation()
        }

        canvas.on('mouse:wheel', handleWheel)
        return () => { canvas.off('mouse:wheel', handleWheel) }
    }, [getMinZoom, validateImagePosition, setZoom])

    useEffect(() => {
        const canvas = fabricRef.current
        if (!canvas) return

        let isDragging = false
        let lastPosX = 0
        let lastPosY = 0

        const handleMouseDown = (opt: { e: MouseEvent }) => {
            isDragging = true
            lastPosX = opt.e.clientX
            lastPosY = opt.e.clientY
        }

        const handleMouseMove = (opt: { e: MouseEvent }) => {
            if (!isDragging) return
            const vpt = canvas.viewportTransform
            if (!vpt) return
            const newX = vpt[4] + opt.e.clientX - lastPosX
            const newY = vpt[5] + opt.e.clientY - lastPosY
            validateImagePosition(newX, newY)
            lastPosX = opt.e.clientX
            lastPosY = opt.e.clientY
        }

        const handleMouseUp = () => {
            isDragging = false
            if (canvas.viewportTransform) {
                canvas.setViewportTransform(canvas.viewportTransform)
            }
        }

        canvas.on('mouse:down', handleMouseDown)
        canvas.on('mouse:move', handleMouseMove)
        canvas.on('mouse:up', handleMouseUp)

        return () => {
            canvas.off('mouse:down', handleMouseDown)
            canvas.off('mouse:move', handleMouseMove)
            canvas.off('mouse:up', handleMouseUp)
        }
    }, [validateImagePosition])

    useEffect(() => {
        const canvas = fabricRef.current
        if (!canvas) return
        const handleClick = (opt: { target?: unknown }) => {
            const target = opt.target as { objectId?: number } | undefined
            if (target && 'objectId' in target && target.objectId !== undefined) {
                selectObject(target.objectId)
            }
        }
        canvas.on('mouse:dblclick', handleClick)
        return () => { canvas.off('mouse:dblclick', handleClick) }
    }, [selectObject])

    return (
        <div ref={containerRef} className="relative h-full w-full overflow-hidden bg-slate-800">
            <canvas ref={canvasRef} />
            <ZoomControls />
        </div>
    )
}
