import { useMemo, useState } from 'react'
import { Check, ChevronDown, ChevronRight, Download, Eye, EyeOff, Search, X } from 'lucide-react'
import type { DetectedObject } from '@/types'
import { cn } from '@/lib/utils'
import { getCategoryColor } from '@/lib/categoryColors'
import { objectKey } from '@/lib/objectKey'

type ObjectSidebarProps = {
  objects: DetectedObject[]
  visibleObjects: DetectedObject[]
  hiddenClasses: Set<string>
  confidenceFilter: number
  onToggleClass: (classKey: string) => void
  onConfidenceChange: (value: number) => void
  reviewStatus: Record<string, 'accepted' | 'rejected'>
  onSetReviewStatus: (key: string, status: 'accepted' | 'rejected' | null) => void
  selectedObjectKey: string | null
  onSelectObject: (key: string) => void
  onExport: () => void
}

function classKeyFor(obj: DetectedObject) {
  return obj.Object.toLowerCase().replace(/_/g, ' ').trim()
}

export function ObjectSidebar({
  objects,
  visibleObjects,
  hiddenClasses,
  confidenceFilter,
  onToggleClass,
  onConfidenceChange,
  reviewStatus,
  onSetReviewStatus,
  selectedObjectKey,
  onSelectObject,
  onExport,
}: ObjectSidebarProps) {
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

  const filteredVisible = useMemo(() => {
    if (!query) return visibleObjects
    const lower = query.toLowerCase()
    return visibleObjects.filter((obj) =>
      obj.Object.toLowerCase().includes(lower) ||
      obj.Text.toLowerCase().includes(lower)
    )
  }, [visibleObjects, query])

  const grouped = useMemo(() => {
    const map = new Map<string, { label: string; items: DetectedObject[] }>()
    filtered.forEach((obj) => {
      const key = classKeyFor(obj) || 'unknown'
      const bucket = map.get(key)
      if (bucket) {
        bucket.items.push(obj)
      } else {
        map.set(key, { label: obj.Object || 'Unknown', items: [obj] })
      }
    })
    return Array.from(map.entries()).sort((a, b) => a[1].label.localeCompare(b[1].label))
  }, [filtered])

  const visibleByClass = useMemo(() => {
    const map = new Map<string, DetectedObject[]>()
    filteredVisible.forEach((obj) => {
      const key = classKeyFor(obj) || 'unknown'
      const bucket = map.get(key)
      if (bucket) {
        bucket.push(obj)
      } else {
        map.set(key, [obj])
      }
    })
    return map
  }, [filteredVisible])

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
        <label className="block text-[11px] font-semibold uppercase tracking-wide text-[var(--text-secondary)] mt-3">
          Confidence ≥ {Math.round(confidenceFilter * 100)}%
          <input
            type="range"
            min={0}
            max={0.95}
            step={0.01}
            value={confidenceFilter}
            onChange={(event) => onConfidenceChange(Number(event.target.value))}
            className="mt-3 w-full"
          />
        </label>
      </div>

      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 && (
          <div className="p-5 text-sm text-[var(--text-secondary)]">No objects match your search.</div>
        )}
        {grouped.map(([groupKey, groupMeta]) => {
          const groupItems = visibleByClass.get(groupKey) || []
          const isHidden = hiddenClasses.has(groupKey)
          return (
          <div key={groupKey} className="border-b border-[var(--border-muted)]">
            <div className={cn(
              'w-full px-5 py-3 bg-[var(--bg-primary)] sticky top-0 z-10',
              'flex items-center justify-between text-xs font-semibold uppercase tracking-wide'
            )}>
              <button
                type="button"
                onClick={() => toggleGroup(groupKey)}
                className="flex items-center gap-2 hover:text-[var(--accent)] transition-colors"
              >
                {expandedGroups.has(groupKey) ? (
                  <ChevronDown className="h-4 w-4 text-[var(--text-secondary)]" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-[var(--text-secondary)]" />
                )}
                {groupMeta.label}
              </button>
              <span className="flex items-center gap-3 text-[var(--text-secondary)]">
                <span>{groupMeta.items.length}</span>
                <button
                  type="button"
                  onClick={() => onToggleClass(groupKey)}
                  className={cn(
                    'h-7 w-7 rounded-md',
                    'flex items-center justify-center',
                    'hover:bg-[var(--bg-secondary)] transition-colors'
                  )}
                  aria-label={isHidden ? 'Show class' : 'Hide class'}
                >
                  {isHidden ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </span>
            </div>
            {expandedGroups.has(groupKey) && !isHidden && (
              <ul className="divide-y divide-[var(--border-muted)]">
                {groupItems.map((obj) => (
                  <li key={`${obj.CategoryID}-${obj.ObjectID}-${obj.Index}`} className="p-4">
                    <button
                      type="button"
                      onClick={() => onSelectObject(objectKey(obj))}
                      className={cn(
                        'w-full text-left rounded-lg p-2 -m-2 transition-colors',
                        selectedObjectKey === objectKey(obj)
                          ? 'bg-[var(--accent)]/10'
                          : 'hover:bg-[var(--bg-secondary)]'
                      )}
                    >
                      <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span
                          className="h-2.5 w-2.5 rounded-sm"
                          style={{ backgroundColor: getCategoryColor(obj.Object) }}
                          aria-hidden="true"
                        />
                        <span className="text-sm font-semibold">{obj.Text}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        {reviewStatus[objectKey(obj)] === 'accepted' && (
                          <span className="text-[10px] font-semibold text-[var(--success)]">Accepted</span>
                        )}
                        {reviewStatus[objectKey(obj)] === 'rejected' && (
                          <span className="text-[10px] font-semibold text-[var(--danger)]">Rejected</span>
                        )}
                        <span className="text-xs text-[var(--text-secondary)]">{Math.round(obj.Score * 100)}%</span>
                      </div>
                      </div>
                      <div className="text-[11px] text-[var(--text-secondary)] mt-1">
                        #{obj.ObjectID} · ({obj.Left}, {obj.Top}) {obj.Width}×{obj.Height}
                      </div>
                    </button>
                    <div className="flex items-center gap-2 mt-3">
                      <button
                        type="button"
                        onClick={() => onSetReviewStatus(objectKey(obj), 'accepted')}
                        className={cn(
                          'px-2.5 py-1.5 rounded-md text-xs font-semibold',
                          reviewStatus[objectKey(obj)] === 'accepted'
                            ? 'bg-[var(--success)] text-white'
                            : 'bg-[var(--bg-primary)] border border-[var(--border-muted)] text-[var(--text-secondary)]',
                          'hover:border-[var(--success)] hover:text-[var(--success)] transition-colors'
                        )}
                      >
                        <span className="inline-flex items-center gap-1">
                          <Check className="h-3.5 w-3.5" />
                          Accept
                        </span>
                      </button>
                      <button
                        type="button"
                        onClick={() => onSetReviewStatus(objectKey(obj), 'rejected')}
                        className={cn(
                          'px-2.5 py-1.5 rounded-md text-xs font-semibold',
                          reviewStatus[objectKey(obj)] === 'rejected'
                            ? 'bg-[var(--danger)] text-white'
                            : 'bg-[var(--bg-primary)] border border-[var(--border-muted)] text-[var(--text-secondary)]',
                          'hover:border-[var(--danger)] hover:text-[var(--danger)] transition-colors'
                        )}
                      >
                        <span className="inline-flex items-center gap-1">
                          <X className="h-3.5 w-3.5" />
                          Reject
                        </span>
                      </button>
                      {reviewStatus[objectKey(obj)] && (
                        <button
                          type="button"
                          onClick={() => onSetReviewStatus(objectKey(obj), null)}
                          className="px-2.5 py-1.5 rounded-md text-xs font-semibold border border-[var(--border-muted)] text-[var(--text-secondary)] hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors"
                        >
                          Reset
                        </button>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )})}
      </div>
    </aside>
  )
}
