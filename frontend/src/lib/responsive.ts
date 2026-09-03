import { useEffect, useState } from 'react'
import type { Screen } from '@/lib/nav'

/**
 * The breakpoint contract, transcribed from the canvas's Breakpoints artboard.
 * Widths follow Tailwind's md / xl.
 */
export type Width = 'phone' | 'tablet' | 'desktop'

export const BREAKPOINTS = { tablet: 768, desktop: 1280 } as const

export function widthFor(px: number): Width {
  if (px < BREAKPOINTS.tablet) return 'phone'
  if (px < BREAKPOINTS.desktop) return 'tablet'
  return 'desktop'
}

/**
 * How usable each screen is at each width. `not-offered` is deliberate: the
 * artboard's note is that a phone "cannot place a port on a 30 000 px raster
 * or bridge a 50 px gap in a trace, so those gates send you to a tablet
 * instead of shipping a version that fails in the field."
 */
export type Availability = 'full' | 'read-only' | 'not-offered'

const AVAILABILITY: Record<Screen, Record<Width, Availability>> = {
  sheets: { phone: 'full', tablet: 'full', desktop: 'full' },
  task: { phone: 'full', tablet: 'full', desktop: 'full' },
  detection: { phone: 'not-offered', tablet: 'read-only', desktop: 'full' },
  run: { phone: 'full', tablet: 'full', desktop: 'full' },
  // The review queue itself is a set of decisions, which a phone can make.
  // The per-gate drawing surfaces are stricter; they carry their own gate.
  review: { phone: 'full', tablet: 'full', desktop: 'full' },
  merge: { phone: 'read-only', tablet: 'full', desktop: 'full' },
  exports: { phone: 'read-only', tablet: 'full', desktop: 'full' },
}

/** What each gate's editing surface needs, per the artboard's per-gate rows. */
export const GATE_AVAILABILITY: Record<1 | 2 | 3 | 4, Record<Width, Availability>> = {
  1: { phone: 'not-offered', tablet: 'full', desktop: 'full' },
  2: { phone: 'not-offered', tablet: 'full', desktop: 'full' },
  3: { phone: 'read-only', tablet: 'full', desktop: 'full' },
  4: { phone: 'full', tablet: 'full', desktop: 'full' },
}

export function availabilityOf(screen: Screen, width: Width): Availability {
  return AVAILABILITY[screen][width]
}

/** Why a screen is limited at this width, for the message shown to the user. */
export const GIVES_WAY: Partial<Record<Screen, string>> = {
  sheets: 'The staged-sheets table becomes cards below 768 px.',
  task: 'Cards stack and parameters go one per row.',
  detection: 'Box editing needs a pointer and room, so it is read-only on a tablet and not offered on a phone.',
  run: 'The timeline is a list at every width.',
  merge: 'Wide tables scroll horizontally on a phone.',
  exports: 'Wide tables scroll horizontally on a phone.',
}

/** Controls grow for touch: 36 px on desktop, 40 on tablet, 48 on phone. */
export function controlHeight(width: Width): number {
  return width === 'phone' ? 48 : width === 'tablet' ? 40 : 36
}

/**
 * Live viewport width bucket.
 *
 * Driven by matchMedia rather than a `resize` listener alone: a resize event
 * is not dispatched for every way a viewport can change (device-metrics
 * emulation and some embedded/preview surfaces change the viewport without
 * one), which left the layout stuck on whatever width it first loaded at.
 * The media queries fire in those cases; `resize` is kept as a backstop.
 */
export function useWidth(): Width {
  const [width, setWidth] = useState<Width>(() =>
    typeof window === 'undefined' ? 'desktop' : widthFor(window.innerWidth)
  )

  useEffect(() => {
    const queries = [
      window.matchMedia(`(min-width: ${BREAKPOINTS.tablet}px)`),
      window.matchMedia(`(min-width: ${BREAKPOINTS.desktop}px)`),
    ]
    const sync = () => setWidth(widthFor(window.innerWidth))

    queries.forEach((q) => q.addEventListener('change', sync))
    window.addEventListener('resize', sync)
    sync()

    return () => {
      queries.forEach((q) => q.removeEventListener('change', sync))
      window.removeEventListener('resize', sync)
    }
  }, [])

  return width
}
