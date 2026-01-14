/**
 * ProgressPanel component - shows detection progress
 */
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { X, CheckCircle, Circle, Loader2 } from 'lucide-react'
import { useDetectionStore } from '@/stores/detectionStore'
import { useUIStore } from '@/stores/uiStore'
import { CATEGORY_COLORS, CATEGORY_NAMES } from '@/lib/constants'

interface ProgressStep {
    id: string
    label: string
    status: 'pending' | 'running' | 'complete'
}

export function ProgressPanel() {
    const { progress, reset, setLoading } = useDetectionStore()
    const { setCurrentView } = useUIStore()

    const steps: ProgressStep[] = [
        {
            id: 'preprocess',
            label: 'Preprocessing image',
            status:
                (progress?.percent ?? 0) > 0
                    ? (progress?.percent ?? 0) >= 30
                        ? 'complete'
                        : 'running'
                    : 'pending',
        },
        {
            id: 'detection',
            label: 'Object detection',
            status:
                (progress?.percent ?? 0) >= 30
                    ? (progress?.percent ?? 0) >= 80
                        ? 'complete'
                        : 'running'
                    : 'pending',
        },
        {
            id: 'ocr',
            label: 'Text extraction (OCR)',
            status:
                (progress?.percent ?? 0) >= 80
                    ? (progress?.percent ?? 0) >= 95
                        ? 'complete'
                        : 'running'
                    : 'pending',
        },
        {
            id: 'postprocess',
            label: 'Post-processing results',
            status: (progress?.percent ?? 0) >= 95 ? 'running' : 'pending',
        },
    ]

    const handleCancel = () => {
        reset()
        setLoading(false)
        setCurrentView('upload')
    }

    // Mock category counts (will be replaced with live data)
    const mockCategories = [
        { id: 4, count: 23 },
        { id: 1, count: 8 },
        { id: 9, count: 12 },
        { id: 13, count: 4 },
    ]

    return (
        <div className="flex h-full flex-col items-center justify-center p-8">
            <div className="w-full max-w-lg space-y-8">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <h2 className="text-xl font-semibold">Analyzing P&ID...</h2>
                    <Button variant="ghost" size="icon" onClick={handleCancel}>
                        <X className="h-4 w-4" />
                    </Button>
                </div>

                {/* Progress Bar */}
                <div className="space-y-2">
                    <Progress value={progress?.percent ?? 0} className="h-2" />
                    <div className="flex justify-between text-sm text-muted-foreground">
                        <span>{progress?.step}</span>
                        <span>{progress?.percent ?? 0}%</span>
                    </div>
                </div>

                {/* Step List */}
                <div className="space-y-3">
                    {steps.map((step) => (
                        <div key={step.id} className="flex items-center gap-3 text-sm">
                            {step.status === 'complete' && (
                                <CheckCircle className="h-4 w-4 text-green-500" />
                            )}
                            {step.status === 'running' && (
                                <Loader2 className="h-4 w-4 animate-spin text-primary" />
                            )}
                            {step.status === 'pending' && (
                                <Circle className="h-4 w-4 text-muted-foreground" />
                            )}
                            <span
                                className={
                                    step.status === 'pending' ? 'text-muted-foreground' : ''
                                }
                            >
                                {step.label}
                            </span>
                        </div>
                    ))}
                </div>

                {/* Live Preview */}
                {(progress?.percent ?? 0) > 20 && (
                    <div className="rounded-lg border bg-muted/30 p-4">
                        <div className="mb-3 flex items-center justify-between">
                            <span className="text-sm font-medium">Live Preview</span>
                            <span className="text-sm text-muted-foreground">
                                🔲 {progress?.objectCount ?? 47} objects detected
                            </span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {mockCategories.map((cat) => (
                                <div
                                    key={cat.id}
                                    className="flex items-center gap-1.5 rounded-md bg-background px-2 py-1 text-xs"
                                >
                                    <div
                                        className="h-3 w-3 rounded-sm"
                                        style={{ backgroundColor: CATEGORY_COLORS[cat.id] }}
                                    />
                                    <span>{CATEGORY_NAMES[cat.id]}</span>
                                    <span className="text-muted-foreground">{cat.count}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Time Estimate */}
                <p className="text-center text-sm text-muted-foreground">
                    Estimated time remaining: ~15 seconds
                </p>
            </div>
        </div>
    )
}
