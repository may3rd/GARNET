import type { DetectedObject } from '@/types'

/**
 * Class colours are literal hex, not theme tokens: boxes are drawn on the
 * white raster, so they are judged against paper in both themes — the same
 * reason the design keeps the sheet white on the dark page.
 *
 * The named entries are the design canvas's palette. The live weights emit
 * more classes than any one config lists (the current .pt returns ~19,
 * datasets/yaml/data.yaml names 11), so anything unnamed gets a stable colour
 * from FALLBACK rather than all sharing the accent — a legend where half the
 * classes are the same colour tells you nothing.
 */
const NAMED: Record<string, string> = {
  'gate valve': '#dc2626',
  'check valve': '#2563eb',
  'control valve': '#16a34a',
  pump: '#9333ea',
  instrument: '#ea580c',
  'instrument dcs': '#ea580c',
  'instrument logic': '#f97316',
  'instrument tag': '#fb923c',
  'line number': '#0891b2',
  vessel: '#0f766e',
  'heat exchanger': '#0f766e',
  'page connection': '#a16207',
}

const FALLBACK = [
  '#7c3aed',
  '#be123c',
  '#0369a1',
  '#4d7c0f',
  '#b45309',
  '#0f766e',
  '#a21caf',
  '#1d4ed8',
  '#65a30d',
  '#c2410c',
]

export function normalizeClass(name: string): string {
  return name.toLowerCase().replace(/_/g, ' ').trim()
}

export function classColor(name: string): string {
  const key = normalizeClass(name)
  const named = NAMED[key]
  if (named) return named
  // Stable per name, so a class keeps its colour across runs and reloads.
  let hash = 0
  for (let i = 0; i < key.length; i += 1) hash = (hash * 31 + key.charCodeAt(i)) >>> 0
  return FALLBACK[hash % FALLBACK.length]
}

export type ClassSummary = { name: string; count: number; color: string }

/** Class rows for the legend, alphabetical by class name. */
export function summarizeClasses(objects: DetectedObject[]): ClassSummary[] {
  const counts = new Map<string, number>()
  objects.forEach((o) => {
    const key = normalizeClass(o.Object)
    counts.set(key, (counts.get(key) ?? 0) + 1)
  })
  return [...counts.entries()]
    .map(([name, count]) => ({ name, count, color: classColor(name) }))
    .sort((a, b) => a.name.localeCompare(b.name))
}
