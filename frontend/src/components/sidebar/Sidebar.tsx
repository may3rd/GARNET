/**
 * Sidebar component - main sidebar container
 */
import { useState } from 'react'
import { ChevronLeft, ChevronRight, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { FilterPanel } from './FilterPanel'
import { ObjectTree } from './ObjectTree'
import { ObjectDetail } from './ObjectDetail'
import { useDetectionStore } from '@/stores/detectionStore'

export function Sidebar() {
    const [collapsed, setCollapsed] = useState(false)
    const { objects } = useDetectionStore()

    // Count reviewed objects
    const accepted = objects.filter((o) => o.status === 'accepted').length
    const rejected = objects.filter((o) => o.status === 'rejected').length
    const pending = objects.length - accepted - rejected

    const handleExport = (format: 'json' | 'excel' | 'coco') => {
        const exportData = {
            timestamp: new Date().toISOString(),
            objects: objects.map((obj) => ({
                id: obj.id,
                category_id: obj.categoryId,
                category_name: obj.categoryName,
                bbox: [obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height],
                confidence: obj.confidence,
                ocr_text: obj.ocrText,
                status: obj.status,
            })),
        }

        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json',
        })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `detection_results.${format === 'coco' ? 'json' : format}`
        a.click()
        URL.revokeObjectURL(url)
    }

    if (collapsed) {
        return (
            <Button
                variant="ghost"
                size="icon"
                className="fixed right-4 top-20 z-10 shadow-lg bg-background border"
                onClick={() => setCollapsed(false)}
            >
                <ChevronLeft className="h-4 w-4" />
            </Button>
        )
    }

    return (
        <div className="flex h-full w-80 flex-col border-l bg-background">
            {/* Header */}
            <div className="flex items-center justify-between border-b p-4">
                <div>
                    <h2 className="font-semibold">Objects</h2>
                    <p className="text-xs text-muted-foreground">
                        {objects.length} total · {accepted} accepted · {pending} pending
                    </p>
                </div>
                <div className="flex items-center gap-1">
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                                <Download className="h-4 w-4" />
                            </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => handleExport('json')}>
                                📄 JSON
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleExport('coco')}>
                                📄 COCO Format
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleExport('excel')}>
                                📊 Excel (coming soon)
                            </DropdownMenuItem>
                        </DropdownMenuContent>
                    </DropdownMenu>
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => setCollapsed(true)}
                    >
                        <ChevronRight className="h-4 w-4" />
                    </Button>
                </div>
            </div>

            {/* Filters */}
            <FilterPanel />

            {/* Object Tree */}
            <ObjectTree />

            {/* Object Detail */}
            <ObjectDetail />
        </div>
    )
}
