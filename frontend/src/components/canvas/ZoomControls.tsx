/**
 * ZoomControls component - canvas zoom toolbar
 */
import { Minus, Plus, Maximize, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TooltipProvider,
} from '@/components/ui/tooltip'
import { useCanvasStore } from '@/stores/canvasStore'

export function ZoomControls() {
    const { zoom, zoomIn, zoomOut, resetZoom, fitToWindow } = useCanvasStore()

    return (
        <TooltipProvider>
            <div className="absolute bottom-4 left-4 flex items-center gap-1 rounded-lg border bg-background/90 p-1 shadow-lg backdrop-blur">
                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={zoomOut}>
                            <Minus className="h-4 w-4" />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="top">Zoom Out (-)</TooltipContent>
                </Tooltip>

                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={zoomIn}>
                            <Plus className="h-4 w-4" />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="top">Zoom In (+)</TooltipContent>
                </Tooltip>

                <div className="w-px h-6 bg-border mx-1" />

                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={resetZoom}>
                            <RotateCcw className="h-4 w-4" />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="top">Reset (0)</TooltipContent>
                </Tooltip>

                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={fitToWindow}>
                            <Maximize className="h-4 w-4" />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="top">Fit (F)</TooltipContent>
                </Tooltip>

                <div className="w-px h-6 bg-border mx-1" />

                <span className="px-2 text-xs font-medium tabular-nums">
                    {Math.round(zoom * 100)}%
                </span>
            </div>
        </TooltipProvider>
    )
}
