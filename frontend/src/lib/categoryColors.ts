const categoryColorMap: Record<string, string> = {
  'gate valve': 'var(--color-gate-valve)',
  'check valve': 'var(--color-check-valve)',
  'control valve': 'var(--color-control-valve)',
  'pump': 'var(--color-pump)',
  'instrument': 'var(--color-instrument)',
  'instrument dcs': 'var(--color-instrument)',
  'instrument logic': 'var(--color-instrument)',
  'instrument tag': 'var(--color-instrument)',
  'line number': 'var(--color-line-number)',
  'line_number': 'var(--color-line-number)',
}

export function getCategoryColor(name: string): string {
  const normalized = name.toLowerCase().replace(/_/g, ' ').trim()
  return categoryColorMap[normalized] || 'var(--accent)'
}
