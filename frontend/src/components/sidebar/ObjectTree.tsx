/**
 * ObjectTree component - category tree with object list
 */
import { useState, useMemo } from 'react'
import { ChevronRight, ChevronDown, Eye, EyeOff, Check, X } from 'lucide-react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { useDetectionStore } from '@/stores/detectionStore'
import { useUIStore } from '@/stores/uiStore'
import { CATEGORY_COLORS, CATEGORY_NAMES } from '@/lib/constants'

interface CategoryGroup {
    categoryId: number
    categoryName: string
    objects: Array<{
        id: number
        confidence: number
        ocrText?: string
        status: 'pending' | 'accepted' | 'rejected'
    }>
}

export function ObjectTree() {
    const [expandedCategories, setExpandedCategories] = useState<Set<number>>(
        new Set()
    )

    const { objects, selectedIds, selectObject } = useDetectionStore()
    const { confidenceFilter, searchQuery, hiddenCategories, toggleCategoryVisibility } =
        useUIStore()

    // Group and filter objects by category
    const categories = useMemo(() => {
        const groups: Record<number, CategoryGroup> = {}

        objects
            .filter((obj) => {
                // Confidence filter
                if (obj.confidence < confidenceFilter) return false
                // Search filter
                if (searchQuery) {
                    const query = searchQuery.toLowerCase()
                    return (
                        obj.categoryName.toLowerCase().includes(query) ||
                        obj.ocrText?.toLowerCase().includes(query)
                    )
                }
                return true
            })
            .forEach((obj) => {
                if (!groups[obj.categoryId]) {
                    groups[obj.categoryId] = {
                        categoryId: obj.categoryId,
                        categoryName: obj.categoryName,
                        objects: [],
                    }
                }
                groups[obj.categoryId].objects.push({
                    id: obj.id,
                    confidence: obj.confidence,
                    ocrText: obj.ocrText,
                    status: obj.status,
                })
            })

        return Object.values(groups).sort((a, b) =>
            a.categoryName.localeCompare(b.categoryName)
        )
    }, [objects, confidenceFilter, searchQuery])

    const toggleCategory = (categoryId: number) => {
        setExpandedCategories((prev) => {
            const next = new Set(prev)
            if (next.has(categoryId)) {
                next.delete(categoryId)
            } else {
                next.add(categoryId)
            }
            return next
        })
    }

    const getStatusIcon = (status: 'pending' | 'accepted' | 'rejected') => {
        switch (status) {
            case 'accepted':
                return <Check className="h-3 w-3 text-green-500" />
            case 'rejected':
                return <X className="h-3 w-3 text-red-500" />
            default:
                return null
        }
    }

    return (
        <ScrollArea className="flex-1">
            <div className="p-2">
                {categories.length === 0 ? (
                    <p className="p-4 text-center text-sm text-muted-foreground">
                        No objects match filters
                    </p>
                ) : (
                    categories.map((category) => {
                        const isExpanded = expandedCategories.has(category.categoryId)
                        const isHidden = hiddenCategories.has(category.categoryId)
                        const color = CATEGORY_COLORS[category.categoryId] || '#ffffff'

                        return (
                            <div key={category.categoryId} className="mb-1">
                                {/* Category Header */}
                                <div
                                    className={`
                    flex items-center justify-between rounded-md px-2 py-1.5
                    hover:bg-muted cursor-pointer transition-colors
                    ${isHidden ? 'opacity-50' : ''}
                  `}
                                >
                                    <button
                                        className="flex flex-1 items-center gap-2"
                                        onClick={() => toggleCategory(category.categoryId)}
                                    >
                                        {isExpanded ? (
                                            <ChevronDown className="h-4 w-4 text-muted-foreground" />
                                        ) : (
                                            <ChevronRight className="h-4 w-4 text-muted-foreground" />
                                        )}
                                        <div
                                            className="h-3 w-3 rounded-sm"
                                            style={{ backgroundColor: color }}
                                        />
                                        <span className="text-sm font-medium">
                                            {CATEGORY_NAMES[category.categoryId] || category.categoryName}
                                        </span>
                                        <span className="text-xs text-muted-foreground">
                                            ({category.objects.length})
                                        </span>
                                    </button>
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-6 w-6"
                                        onClick={() => toggleCategoryVisibility(category.categoryId)}
                                    >
                                        {isHidden ? (
                                            <EyeOff className="h-3.5 w-3.5" />
                                        ) : (
                                            <Eye className="h-3.5 w-3.5" />
                                        )}
                                    </Button>
                                </div>

                                {/* Object List */}
                                {isExpanded && !isHidden && (
                                    <div className="ml-4 border-l pl-2">
                                        {category.objects.map((obj) => {
                                            const isSelected = selectedIds.has(obj.id)

                                            return (
                                                <button
                                                    key={obj.id}
                                                    className={`
                            flex w-full items-center justify-between rounded-md px-2 py-1
                            text-sm transition-colors
                            ${isSelected ? 'bg-primary/10 text-primary' : 'hover:bg-muted'}
                          `}
                                                    onClick={() => selectObject(obj.id)}
                                                >
                                                    <span className="flex items-center gap-2">
                                                        {obj.ocrText || `#${obj.id}`}
                                                        {getStatusIcon(obj.status)}
                                                    </span>
                                                    <span className="text-xs text-muted-foreground">
                                                        {Math.round(obj.confidence * 100)}%
                                                    </span>
                                                </button>
                                            )
                                        })}
                                    </div>
                                )}
                            </div>
                        )
                    })
                )}
            </div>
        </ScrollArea>
    )
}
