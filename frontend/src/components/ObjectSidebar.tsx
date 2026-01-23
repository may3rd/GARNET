import { useMemo, useState } from 'react'
import { ChevronDown, ChevronRight, Download, Search } from 'lucide-react'
import type { DetectedObject } from '@/types'
import { cn } from '@/lib/utils'
import { getCategoryColor } from '@/lib/categoryColors'

type ObjectSidebarProps = {
  objects: DetectedObject[]
  onExport: () => void
}

export function ObjectSidebar({ objects, onExport }: ObjectSidebarProps) {
  const [query, setQuery] = useState('')
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set())

  const filtered = useMemo(() => {
    if (!query) return objects
    const lower = query.toLowerCase()
    return objects.filter((obj) =>
      obj.Object.toLowerCase().includes(lower) ||
      obj.Text.toLowerCase().includes(lower)
    )
  }, [objects, query])

  const grouped = useMemo(() => {
    const map = new Map<string, DetectedObject[]>()
    filtered.forEach((obj) => {
      const key = obj.Object || 'Unknown'
      const bucket = map.get(key)
      if (bucket) {
        bucket.push(obj)
      } else {
        map.set(key, [obj])
      }
    })
    return Array.from(map.entries()).sort((a, b) => a[0].localeCompare(b[0]))
  }, [filtered])

  const toggleGroup = (groupName: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(groupName)) {
        next.delete(groupName)
      } else {
        next.add(groupName)
      }
      return next
    })
  }

  return (
    <aside className="h-full w-full border-l border-[var(--border-muted)] bg-[var(--bg-secondary)] flex flex-col">
      <div className="p-5 border-b border-[var(--border-muted)]">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-sm font-semibold">Objects</div>
            <div className="text-xs text-[var(--text-secondary)]">{objects.length} detected</div>
          </div>
          <button
            onClick={onExport}
            className={cn(
              'inline-flex items-center gap-2 px-3 py-2 rounded-lg',
              'bg-[var(--accent)] text-white text-xs font-semibold',
              'hover:bg-[var(--accent-strong)] transition-colors'
            )}
          >
            <Download className="h-4 w-4" />
            Export JSON
          </button>
        </div>

        <label className="relative block">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-secondary)]">
            <Search className="h-4 w-4" />
          </span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search objects"
            className={cn(
              'w-full rounded-lg border border-[var(--border-muted)]',
              'bg-[var(--bg-primary)] px-9 py-2 text-sm'
            )}
          />
        </label>
      </div>

      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 && (
          <div className="p-5 text-sm text-[var(--text-secondary)]">No objects match your search.</div>
        )}
        {grouped.map(([groupName, groupItems]) => (
          <div key={groupName} className="border-b border-[var(--border-muted)]">
            <button
              type="button"
              onClick={() => toggleGroup(groupName)}
              className={cn(
                'w-full px-5 py-3 bg-[var(--bg-primary)] sticky top-0 z-10',
                'flex items-center justify-between text-xs font-semibold uppercase tracking-wide',
                'hover:bg-[var(--bg-secondary)] transition-colors'
              )}
            >
              <span className="flex items-center gap-2">
                {expandedGroups.has(groupName) ? (
                  <ChevronDown className="h-4 w-4 text-[var(--text-secondary)]" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-[var(--text-secondary)]" />
                )}
                {groupName}
              </span>
              <span className="text-[var(--text-secondary)]">{groupItems.length}</span>
            </button>
            {expandedGroups.has(groupName) && (
              <ul className="divide-y divide-[var(--border-muted)]">
                {groupItems.map((obj) => (
                  <li key={`${obj.CategoryID}-${obj.ObjectID}-${obj.Index}`} className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span
                          className="h-2.5 w-2.5 rounded-sm"
                          style={{ backgroundColor: getCategoryColor(obj.Object) }}
                          aria-hidden="true"
                        />
                        <span className="text-sm font-semibold">{obj.Text}</span>
                      </div>
                      <div className="text-xs text-[var(--text-secondary)]">{Math.round(obj.Score * 100)}%</div>
                    </div>
                    <div className="text-[11px] text-[var(--text-secondary)] mt-1">
                      #{obj.ObjectID} · ({obj.Left}, {obj.Top}) {obj.Width}×{obj.Height}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>
    </aside>
  )
}
