import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, ChevronDown, ChevronRight, Download, Eye, EyeOff, Plus, Save, Search, X } from 'lucide-react'
import type { DetectedObject } from '@/types'
import { cn } from '@/lib/utils'
import { getCategoryColor } from '@/lib/categoryColors'
import { objectKey } from '@/lib/objectKey'
import type { ExportFormat } from '@/lib/exportFormats'

type ExportFilter = 'all' | 'accepted' | 'rejected' | 'visible'

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
  isCreating: boolean
  createDraft: {
    Object: string
    Left: number
    Top: number
    Width: number
    Height: number
    Text: string
  } | null
  onStartCreate: () => void
  onCancelCreate: () => void
  onUpdateCreateDraft: (field: 'Object' | 'Left' | 'Top' | 'Width' | 'Height' | 'Text', value: string) => void
  onSaveCreate: () => void
  onExport: (format: ExportFormat, filter: ExportFilter) => void
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
  isCreating,
  createDraft,
  onStartCreate,
  onCancelCreate,
  onUpdateCreateDraft,
  onSaveCreate,
  onExport,
}: ObjectSidebarProps) {
  const [queryInput, setQueryInput] = useState('')
  const [query, setQuery] = useState('')
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set())
  const [showAllGroups, setShowAllGroups] = useState<Set<string>>(new Set())
  const [exportOpen, setExportOpen] = useState(false)
  const [statsOpen, setStatsOpen] = useState(true)
  const exportRef = useRef<HTMLDivElement>(null)
  const queryDebounceRef = useRef<number | null>(null)

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

  const stats = useMemo(() => {
    const accepted = objects.filter((obj) => reviewStatus[objectKey(obj)] === 'accepted').length
    const rejected = objects.filter((obj) => reviewStatus[objectKey(obj)] === 'rejected').length
    const pending = objects.length - accepted - rejected
    const avgConfidence = objects.length
      ? Math.round(objects.reduce((sum, obj) => sum + obj.Score, 0) / objects.length * 100)
      : 0
    return { accepted, rejected, pending, avgConfidence }
  }, [objects, reviewStatus])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (!exportRef.current) return
      if (!exportRef.current.contains(event.target as Node)) {
        setExportOpen(false)
      }
    }
    if (exportOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [exportOpen])

  useEffect(() => {
    if (queryDebounceRef.current) {
      window.clearTimeout(queryDebounceRef.current)
    }
    queryDebounceRef.current = window.setTimeout(() => {
      setQuery(queryInput)
    }, 150)
    return () => {
      if (queryDebounceRef.current) {
        window.clearTimeout(queryDebounceRef.current)
        queryDebounceRef.current = null
      }
    }
  }, [queryInput])

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

  const createReady = createDraft && createDraft.Width > 2 && createDraft.Height > 2
  const MAX_GROUP_ITEMS = 200

  return (
    <aside className="h-full w-full border-l border-[var(--border-muted)] bg-[var(--bg-secondary)] flex flex-col">
      <div className="p-5 border-b border-[var(--border-muted)]">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <button
              onClick={isCreating ? onCancelCreate : onStartCreate}
              className={cn(
                'inline-flex items-center gap-2 px-3 py-2 rounded-lg',
                isCreating
                  ? 'bg-[var(--bg-primary)] text-[var(--text-secondary)] border border-[var(--border-muted)]'
                  : 'bg-[var(--accent)] text-white',
                'text-xs font-semibold hover:brightness-95 transition-colors'
              )}
            >
              {isCreating ? <X className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
              {isCreating ? 'Cancel' : 'Add Box'}
            </button>
            <div className="relative" ref={exportRef}>
              <button
                onClick={() => setExportOpen((prev) => !prev)}
                className={cn(
                  'inline-flex items-center gap-2 px-3 py-2 rounded-lg',
                  'bg-[var(--accent)] text-white text-xs font-semibold',
                  'hover:bg-[var(--accent-strong)] transition-colors'
                )}
              >
                <Download className="h-4 w-4" />
                Export
                <ChevronDown className="h-4 w-4" />
              </button>
            {exportOpen && (
              <div className="absolute right-0 mt-2 w-[280px] rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] shadow-lg z-20 p-2">
                {([
                  { id: 'json', label: 'JSON' },
                  { id: 'yolo', label: 'YOLO' },
                  { id: 'coco', label: 'COCO' },
                  { id: 'labelme', label: 'LabelMe' },
                ] as const).map((format) => (
                  <div key={format.id} className="mb-2 last:mb-0">
                    <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-secondary)]">
                      {format.label}
                    </div>
                    {([
                      { id: 'all', label: 'All objects' },
                      { id: 'visible', label: 'Visible only' },
                      { id: 'accepted', label: 'Accepted only' },
                      { id: 'rejected', label: 'Rejected only' },
                    ] as const).map((item) => (
                      <button
                        key={`${format.id}-${item.id}`}
                        type="button"
                        onClick={() => {
                          setExportOpen(false)
                          onExport(format.id, item.id)
                        }}
                        className="w-full text-left px-3 py-2 text-xs rounded-md hover:bg-[var(--bg-secondary)] transition-colors"
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>
                ))}
              </div>
            )}
            </div>
          </div>
        </div>

        <label className="relative block">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-secondary)]">
            <Search className="h-4 w-4" />
          </span>
          <input
            value={queryInput}
            onChange={(event) => setQueryInput(event.target.value)}
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
        {isCreating && (
          <div className="mt-4 p-3 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)]">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-secondary)]">
              New Box
            </div>
            <div className="text-[11px] text-[var(--text-secondary)] mt-1">
              Click and drag on the canvas to draw a box.
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <input
                value={createDraft?.Object ?? ''}
                onChange={(event) => onUpdateCreateDraft('Object', event.target.value)}
                placeholder="Class"
                className={cn(
                  'col-span-2 rounded-md border border-[var(--border-muted)]',
                  'bg-[var(--bg-secondary)] px-2 py-1.5'
                )}
              />
              <input
                value={createDraft?.Text ?? ''}
                onChange={(event) => onUpdateCreateDraft('Text', event.target.value)}
                placeholder="Label"
                className={cn(
                  'col-span-2 rounded-md border border-[var(--border-muted)]',
                  'bg-[var(--bg-secondary)] px-2 py-1.5'
                )}
              />
              <input
                value={createDraft?.Left ?? ''}
                onChange={(event) => onUpdateCreateDraft('Left', event.target.value)}
                placeholder="Left"
                className={cn(
                  'rounded-md border border-[var(--border-muted)]',
                  'bg-[var(--bg-secondary)] px-2 py-1.5'
                )}
              />
              <input
                value={createDraft?.Top ?? ''}
                onChange={(event) => onUpdateCreateDraft('Top', event.target.value)}
                placeholder="Top"
                className={cn(
                  'rounded-md border border-[var(--border-muted)]',
                  'bg-[var(--bg-secondary)] px-2 py-1.5'
                )}
              />
              <input
                value={createDraft?.Width ?? ''}
                onChange={(event) => onUpdateCreateDraft('Width', event.target.value)}
                placeholder="Width"
                className={cn(
                  'rounded-md border border-[var(--border-muted)]',
                  'bg-[var(--bg-secondary)] px-2 py-1.5'
                )}
              />
              <input
                value={createDraft?.Height ?? ''}
                onChange={(event) => onUpdateCreateDraft('Height', event.target.value)}
                placeholder="Height"
                className={cn(
                  'rounded-md border border-[var(--border-muted)]',
                  'bg-[var(--bg-secondary)] px-2 py-1.5'
                )}
              />
            </div>
            <button
              type="button"
              onClick={onSaveCreate}
              disabled={!createReady}
              className={cn(
                'mt-3 w-full inline-flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold',
                createReady
                  ? 'bg-[var(--accent)] text-white'
                  : 'bg-[var(--bg-secondary)] text-[var(--text-secondary)] cursor-not-allowed'
              )}
            >
              <Save className="h-4 w-4" />
              Save New Box
            </button>
            <button
              type="button"
              onClick={onCancelCreate}
              className="mt-2 w-full inline-flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold border border-[var(--border-muted)] text-[var(--text-secondary)] hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors"
            >
              <X className="h-4 w-4" />
              Cancel
            </button>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="border-b border-[var(--border-muted)]">
          <button
            type="button"
            onClick={() => setStatsOpen((prev) => !prev)}
            className="w-full px-5 py-3 flex items-center justify-between text-xs font-semibold uppercase tracking-wide text-[var(--text-secondary)]"
          >
            Statistics
            <ChevronDown className={cn('h-4 w-4 transition-transform', statsOpen ? 'rotate-180' : '')} />
          </button>
          {statsOpen && (
            <div className="px-5 pb-4 text-xs text-[var(--text-secondary)] space-y-2">
              <div className="flex items-center justify-between">
                <span>Total</span>
                <span className="font-semibold text-[var(--text-primary)]">{objects.length}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Accepted</span>
                <span className="font-semibold text-[var(--success)]">{stats.accepted}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Rejected</span>
                <span className="font-semibold text-[var(--danger)]">{stats.rejected}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Pending</span>
                <span className="font-semibold text-[var(--text-primary)]">{stats.pending}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Avg confidence</span>
                <span className="font-semibold text-[var(--text-primary)]">{stats.avgConfidence}%</span>
              </div>
            </div>
          )}
        </div>
        {filtered.length === 0 && (
          <div className="p-5 text-sm text-[var(--text-secondary)]">No objects match your search.</div>
        )}
        {grouped.map(([groupKey, groupMeta]) => {
          const groupItems = visibleByClass.get(groupKey) || []
          const isHidden = hiddenClasses.has(groupKey)
          const reviewedCount = groupMeta.items.filter((obj) => {
            const status = reviewStatus[objectKey(obj)]
            return status === 'accepted' || status === 'rejected'
          }).length
          const isExpanded = expandedGroups.has(groupKey)
          const showAll = showAllGroups.has(groupKey)
          const displayItems = showAll ? groupItems : groupItems.slice(0, MAX_GROUP_ITEMS)

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
                {isExpanded ? (
                  <ChevronDown className="h-4 w-4 text-[var(--text-secondary)]" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-[var(--text-secondary)]" />
                )}
                {groupMeta.label}
              </button>
              <span className="flex items-center gap-3 text-[var(--text-secondary)]">
                <span>{reviewedCount}/{groupMeta.items.length}</span>
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
            {isExpanded && !isHidden && (
              <ul className="divide-y divide-[var(--border-muted)]">
                {displayItems.map((obj) => (
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
                          'px-2.5 py-1.5 rounded-md text-xs font-semibold transition-all',
                          reviewStatus[objectKey(obj)] === 'accepted'
                            ? 'bg-[var(--success)] text-white hover:brightness-95'
                            : 'bg-[var(--bg-primary)] border border-[var(--border-muted)] text-[var(--text-secondary)] hover:border-[var(--success)] hover:text-[var(--success)] hover:bg-[var(--success)]/5'
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
                          'px-2.5 py-1.5 rounded-md text-xs font-semibold transition-all',
                          reviewStatus[objectKey(obj)] === 'rejected'
                            ? 'bg-[var(--danger)] text-white hover:brightness-95'
                            : 'bg-[var(--bg-primary)] border border-[var(--border-muted)] text-[var(--text-secondary)] hover:border-[var(--danger)] hover:text-[var(--danger)] hover:bg-[var(--danger)]/5'
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
                {!showAll && groupItems.length > MAX_GROUP_ITEMS && (
                  <li className="p-4">
                    <button
                      type="button"
                      onClick={() => setShowAllGroups((prev) => new Set(prev).add(groupKey))}
                      className="w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-primary)] px-3 py-2 text-xs font-semibold text-[var(--accent)] hover:bg-[var(--bg-secondary)] transition-colors"
                    >
                      Show all {groupItems.length} items
                    </button>
                  </li>
                )}
              </ul>
            )}
          </div>
        )})}
      </div>
    </aside>
  )
}
