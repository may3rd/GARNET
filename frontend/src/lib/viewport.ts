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
