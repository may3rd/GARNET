/**
 * ObjectDetail component - selected object details panel
 */
import { Check, X, Pencil } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { useDetectionStore } from '@/stores/detectionStore'
import { CATEGORY_COLORS, CATEGORY_NAMES } from '@/lib/constants'

export function ObjectDetail() {
    const { objects, selectedIds, updateObjectStatus } = useDetectionStore()

    // Get selected objects
    const selectedObjects = objects.filter((obj) => selectedIds.has(obj.id))

    if (selectedObjects.length === 0) {
        return (
            <div className="border-t p-4">
                <p className="text-center text-sm text-muted-foreground">
                    Select an object to view details
                </p>
            </div>
        )
    }

    // Single selection
    if (selectedObjects.length === 1) {
        const obj = selectedObjects[0]
        const color = CATEGORY_COLORS[obj.categoryId] || '#ffffff'

        return (
            <div className="border-t p-4 space-y-3">
                <div className="flex items-center gap-2">
                    <div
                        className="h-4 w-4 rounded"
                        style={{ backgroundColor: color }}
                    />
                    <span className="font-semibold">
                        {CATEGORY_NAMES[obj.categoryId] || obj.categoryName}
                    </span>
                </div>

                <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                        <span className="text-muted-foreground">Confidence</span>
                        <span>{Math.round(obj.confidence * 100)}%</span>
                    </div>
                    {obj.ocrText && (
                        <div className="flex justify-between">
                            <span className="text-muted-foreground">OCR</span>
                            <span className="font-mono">{obj.ocrText}</span>
                        </div>
                    )}
                    <div className="flex justify-between">
                        <span className="text-muted-foreground">Status</span>
                        <span className="capitalize">{obj.status}</span>
                    </div>
                </div>

                <Separator />

                <div className="flex gap-2">
                    <Button
                        variant={obj.status === 'accepted' ? 'default' : 'outline'}
                        size="sm"
                        className="flex-1"
                        onClick={() => updateObjectStatus(obj.id, 'accepted')}
                    >
                        <Check className="mr-1 h-3 w-3" />
                        Accept
                    </Button>
                    <Button
                        variant={obj.status === 'rejected' ? 'destructive' : 'outline'}
                        size="sm"
                        className="flex-1"
                        onClick={() => updateObjectStatus(obj.id, 'rejected')}
                    >
                        <X className="mr-1 h-3 w-3" />
                        Reject
                    </Button>
                </div>

                <Button variant="outline" size="sm" className="w-full">
                    <Pencil className="mr-1 h-3 w-3" />
                    Edit
                </Button>
            </div>
        )
    }

    // Multi-selection
    return (
        <div className="border-t p-4 space-y-3">
            <p className="text-sm font-medium">
                {selectedObjects.length} objects selected
            </p>

            <div className="flex gap-2">
                <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                    onClick={() =>
                        selectedObjects.forEach((obj) =>
                            updateObjectStatus(obj.id, 'accepted')
                        )
                    }
                >
                    <Check className="mr-1 h-3 w-3" />
                    Accept All
                </Button>
                <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                    onClick={() =>
                        selectedObjects.forEach((obj) =>
                            updateObjectStatus(obj.id, 'rejected')
                        )
                    }
                >
                    <X className="mr-1 h-3 w-3" />
                    Reject All
                </Button>
            </div>
        </div>
    )
}
