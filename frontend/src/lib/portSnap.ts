/**
 * Equipment ports sit on the box border. Pure, so `bun run src/lib/portSnap.test.ts` checks it.
 */

export type BBox = { x_min: number; y_min: number; x_max: number; y_max: number }
/** The spellings the tracer's port loader accepts (garnet/stage5b/host.py `_SIDE_TO_DIRECTION`). */
export type PortSide = 'top' | 'bottom' | 'left' | 'right'

/** Snap (x, y) onto the nearest edge of `b`, inside or outside the box; returns that edge as `side`. */
export function snapToEdge(b: BBox, x: number, y: number): { side: PortSide; point_px: [number, number] } {
  const cx = Math.round(Math.min(Math.max(x, b.x_min), b.x_max))
  const cy = Math.round(Math.min(Math.max(y, b.y_min), b.y_max))
  const edges: [PortSide, number, [number, number]][] = [
    ['top', Math.hypot(x - cx, y - b.y_min), [cx, b.y_min]],
    ['bottom', Math.hypot(x - cx, y - b.y_max), [cx, b.y_max]],
    ['left', Math.hypot(x - b.x_min, y - cy), [b.x_min, cy]],
    ['right', Math.hypot(x - b.x_max, y - cy), [b.x_max, cy]],
  ]
  const [side, , point_px] = edges.reduce((best, e) => (e[1] < best[1] ? e : best))
  return { side, point_px }
}
