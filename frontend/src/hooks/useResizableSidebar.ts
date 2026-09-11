import { useRef, useState } from 'react'

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

/**
 * Drag-to-resize + collapse state for a right-docked sidebar. The sidebar
 * sits to the right of a canvas, so dragging its left-edge handle left
 * (negative clientX delta) grows it and dragging right shrinks it.
 */
export function useResizableSidebar(defaultWidth: number, opts?: { min?: number; max?: number }) {
  const { min = 220, max = 520 } = opts ?? {}
  const [width, setWidth] = useState(defaultWidth)
  const [collapsed, setCollapsed] = useState(false)
  const widthRef = useRef(width)
  widthRef.current = width

  const startResize = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    const originX = e.clientX
    const originWidth = widthRef.current
    const move = (ev: MouseEvent) => {
      const next = clamp(originWidth - (ev.clientX - originX), min, max)
      widthRef.current = next
      setWidth(next)
    }
    const up = () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  const toggleCollapsed = () => setCollapsed((v) => !v)

  return { width, collapsed, toggleCollapsed, startResize }
}
