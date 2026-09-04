export type Pan = { x: number; y: number }

/**
 * Keep the raster filling the view: no empty gutters.
 *
 * On an axis where the scaled sheet is larger than the view, the pan is
 * clamped so neither edge can be dragged inside the frame. On an axis where it
 * is smaller (which is always one of them at fit scale, since the sheet's
 * aspect rarely matches the frame's) it is centred instead — otherwise "no
 * gap" would be impossible and the sheet would sit against a corner.
 */
export function clampPan(
  pan: Pan,
  scale: number,
  imgW: number,
  imgH: number,
  viewW: number,
  viewH: number
): Pan {
  const axis = (p: number, content: number, view: number) => {
    if (content <= view) return (view - content) / 2
    return Math.min(0, Math.max(view - content, p))
  }
  return {
    x: axis(pan.x, imgW * scale, viewW),
    y: axis(pan.y, imgH * scale, viewH),
  }
}

/** Scale at which the whole sheet fits the view. */
export function fitScale(imgW: number, imgH: number, viewW: number, viewH: number): number {
  if (!imgW || !imgH || !viewW || !viewH) return 1
  return Math.min(viewW / imgW, viewH / imgH)
}

/* ---------------------------------------------------------------------------
   Box editing. Coordinates are image pixels, matching DetectedObject.
   --------------------------------------------------------------------------- */

export type Box = { Left: number; Top: number; Width: number; Height: number }

/** The eight anchors, named by compass point. */
export type Handle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w'

export const HANDLES: Handle[] = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']

/** Which cursor each anchor should show. */
export const HANDLE_CURSOR: Record<Handle, string> = {
  nw: 'nwse-resize',
  n: 'ns-resize',
  ne: 'nesw-resize',
  e: 'ew-resize',
  se: 'nwse-resize',
  s: 'ns-resize',
  sw: 'nesw-resize',
  w: 'ew-resize',
}

/** Anchor position in image coordinates, for drawing the handle. */
export function handlePoint(box: Box, handle: Handle): { x: number; y: number } {
  const midX = box.Left + box.Width / 2
  const midY = box.Top + box.Height / 2
  const right = box.Left + box.Width
  const bottom = box.Top + box.Height
  switch (handle) {
    case 'nw': return { x: box.Left, y: box.Top }
    case 'n': return { x: midX, y: box.Top }
    case 'ne': return { x: right, y: box.Top }
    case 'e': return { x: right, y: midY }
    case 'se': return { x: right, y: bottom }
    case 's': return { x: midX, y: bottom }
    case 'sw': return { x: box.Left, y: bottom }
    case 'w': return { x: box.Left, y: midY }
  }
}

const round = (b: Box): Box => ({
  Left: Math.round(b.Left),
  Top: Math.round(b.Top),
  Width: Math.round(b.Width),
  Height: Math.round(b.Height),
})

/**
 * Drag one anchor to (x, y).
 *
 * Dragging an edge past its opposite flips the box rather than producing a
 * negative size, and the result is clamped inside the sheet with a minimum
 * size so a box can never be dragged away to nothing.
 */
export function resizeBox(
  start: Box,
  handle: Handle,
  x: number,
  y: number,
  imgW: number,
  imgH: number,
  min = 4
): Box {
  let left = start.Left
  let top = start.Top
  let right = start.Left + start.Width
  let bottom = start.Top + start.Height

  if (handle.includes('w')) left = x
  if (handle.includes('e')) right = x
  if (handle.includes('n')) top = y
  if (handle.includes('s')) bottom = y

  // A flipped drag is a valid gesture; normalise instead of going negative.
  if (right < left) [left, right] = [right, left]
  if (bottom < top) [top, bottom] = [bottom, top]

  left = Math.max(0, Math.min(left, imgW))
  right = Math.max(0, Math.min(right, imgW))
  top = Math.max(0, Math.min(top, imgH))
  bottom = Math.max(0, Math.min(bottom, imgH))

  let width = Math.max(min, right - left)
  let height = Math.max(min, bottom - top)
  // Honouring the minimum must not push the box outside the sheet.
  if (left + width > imgW) left = Math.max(0, imgW - width)
  if (top + height > imgH) top = Math.max(0, imgH - height)
  width = Math.min(width, imgW)
  height = Math.min(height, imgH)

  return round({ Left: left, Top: top, Width: width, Height: height })
}

/** Move the whole box by a delta, kept inside the sheet. */
export function moveBox(start: Box, dx: number, dy: number, imgW: number, imgH: number): Box {
  const width = Math.min(start.Width, imgW)
  const height = Math.min(start.Height, imgH)
  return round({
    Left: Math.max(0, Math.min(start.Left + dx, imgW - width)),
    Top: Math.max(0, Math.min(start.Top + dy, imgH - height)),
    Width: width,
    Height: height,
  })
}

/* ---------------------------------------------------------------------------
   Wheel input
   --------------------------------------------------------------------------- */

/** Rough pixel equivalents for the non-pixel wheel delta modes. */
const LINE_PX = 16
const PAGE_PX = 800

/**
 * Wheel deltas in comparable pixels.
 *
 * `deltaY` is only pixels when `deltaMode` is 0. A mouse wheel commonly
 * reports lines (mode 1, ~3 per notch) and some browsers report pages
 * (mode 2), so treating the raw number as pixels makes a wheel notch nearly
 * imperceptible while a trackpad flick lurches.
 *
 * A trackpad pinch arrives as a wheel event with `ctrlKey` set and small
 * deltas; it is amplified so pinching and scrolling zoom at a similar rate.
 * The result is clamped because one frame's worth of input should never cross
 * the whole zoom range.
 */
export function wheelPixels(
  delta: number,
  deltaMode: number,
  ctrlKey = false,
  pagePx = PAGE_PX
): number {
  const unit = deltaMode === 1 ? LINE_PX : deltaMode === 2 ? pagePx : 1
  const px = delta * unit * (ctrlKey ? 4 : 1)
  return Math.max(-240, Math.min(240, px))
}

/**
 * Zoom about a point in viewport coordinates.
 *
 * Returns the new scale and the pan that keeps the image point currently under
 * (cx, cy) under it afterwards, already settled so no gutter opens up. Pure, so
 * a caller can apply both in one go rather than nesting one state update inside
 * another — an impure updater gets double-invoked under React StrictMode and
 * applies the zoom twice.
 */
export function zoomAbout(
  pan: Pan,
  scale: number,
  factor: number,
  cx: number,
  cy: number,
  imgW: number,
  imgH: number,
  viewW: number,
  viewH: number,
  min = 0.02,
  max = 8
): { scale: number; pan: Pan } {
  const next = Math.max(min, Math.min(max, scale * factor))
  if (next === scale) return { scale, pan }
  const imgX = (cx - pan.x) / scale
  const imgY = (cy - pan.y) / scale
  return {
    scale: next,
    pan: clampPan({ x: cx - imgX * next, y: cy - imgY * next }, next, imgW, imgH, viewW, viewH),
  }
}
