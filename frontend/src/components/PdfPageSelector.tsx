import { useEffect, useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { cn } from '@/lib/utils'
import { Loader2 } from 'lucide-react'

export type PdfPageSelectorProps = {
    open: boolean
    pages: string[] // base64-encoded PNG images
    fileName: string
    isLoading?: boolean
    onConfirm: (selectedIndices: number[]) => void
    onCancel: () => void
}

export function PdfPageSelector({
    open,
    pages,
    fileName,
    isLoading,
    onConfirm,
    onCancel,
}: PdfPageSelectorProps) {
    const [selected, setSelected] = useState<Set<number>>(() => new Set(pages.map((_, i) => i)))

    useEffect(() => {
        if (!open) return
        setSelected(new Set(pages.map((_, i) => i)))
    }, [open, pages])

    const togglePage = (index: number) => {
        setSelected((prev) => {
            const next = new Set(prev)
            if (next.has(index)) {
                next.delete(index)
            } else {
                next.add(index)
            }
            return next
        })
    }

    const selectAll = () => {
        setSelected(new Set(pages.map((_, i) => i)))
    }

    const selectNone = () => {
        setSelected(new Set())
    }

    const handleConfirm = () => {
        const indices = Array.from(selected).sort((a, b) => a - b)
        onConfirm(indices)
    }

    return (
        <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onCancel()}>
            <DialogContent className="max-w-4xl max-h-[85vh] flex flex-col">
                <DialogHeader>
                    <DialogTitle>Select PDF Pages</DialogTitle>
                    <div className="text-xs text-[var(--text-secondary)]">
                        {fileName} · {pages.length} page{pages.length !== 1 ? 's' : ''}
                    </div>
                </DialogHeader>

                {isLoading ? (
                    <div className="flex-1 flex items-center justify-center py-12">
                        <Loader2 className="h-8 w-8 animate-spin text-[var(--accent)]" />
                        <span className="ml-3 text-sm text-[var(--text-secondary)]">Extracting pages...</span>
                    </div>
                ) : (
                    <>
                        <div className="flex items-center gap-2 text-xs mb-2">
                            <Button variant="ghost" size="sm" onClick={selectAll}>
                                Select all
                            </Button>
                            <Button variant="ghost" size="sm" onClick={selectNone}>
                                Clear
                            </Button>
                            <span className="text-[var(--text-secondary)]">
                                {selected.size} of {pages.length} selected
                            </span>
                        </div>

                        <div className="flex-1 overflow-y-auto grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 p-1">
                            {pages.map((b64, index) => (
                                <button
                                    key={index}
                                    type="button"
                                    onClick={() => togglePage(index)}
                                    className={cn(
                                        'relative rounded-lg border-2 overflow-hidden transition-all',
                                        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]',
                                        selected.has(index)
                                            ? 'border-[var(--accent)] ring-2 ring-[var(--accent)]/30'
                                            : 'border-[var(--border-muted)] hover:border-[var(--accent)]/50'
                                    )}
                                >
                                    <img
                                        src={`data:image/png;base64,${b64}`}
                                        alt={`Page ${index + 1}`}
                                        className="w-full h-auto"
                                    />
                                    <div className="absolute top-1 left-1">
                                        <Checkbox
                                            checked={selected.has(index)}
                                            onCheckedChange={() => togglePage(index)}
                                            onClick={(e) => e.stopPropagation()}
                                        />
                                    </div>
                                    <div className="absolute bottom-1 right-1 text-[10px] px-1.5 py-0.5 rounded bg-black/60 text-white">
                                        {index + 1}
                                    </div>
                                </button>
                            ))}
                        </div>
                    </>
                )}

                <DialogFooter className="mt-4">
                    <Button variant="outline" onClick={onCancel}>
                        Cancel
                    </Button>
                    <Button onClick={handleConfirm} disabled={selected.size === 0 || isLoading}>
                        Process {selected.size} page{selected.size !== 1 ? 's' : ''}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}
