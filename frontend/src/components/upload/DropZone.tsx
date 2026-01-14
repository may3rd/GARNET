/**
 * DropZone component - drag-and-drop file upload
 */
import { useCallback, useState } from 'react'
import { Upload, FileImage, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useUIStore } from '@/stores/uiStore'
import { useDetectionStore } from '@/stores/detectionStore'
import { SUPPORTED_FILE_TYPES } from '@/lib/constants'

interface RecentFile {
    name: string
    timestamp: number
}

export function DropZone() {
    const [isDragging, setIsDragging] = useState(false)
    const [recentFiles, setRecentFiles] = useState<RecentFile[]>(() => {
        const saved = localStorage.getItem('garnet-recent-files')
        return saved ? JSON.parse(saved) : []
    })

    const { setCurrentView } = useUIStore()
    const { setImage } = useDetectionStore()

    const handleFile = useCallback(
        (file: File) => {
            if (!SUPPORTED_FILE_TYPES.includes(file.type)) {
                alert('Unsupported file type. Please use JPG, PNG, or PDF.')
                return
            }

            // Create object URL for preview
            const url = URL.createObjectURL(file)
            setImage(url, file.name)

            // Save to recent files
            const newRecent = [
                { name: file.name, timestamp: Date.now() },
                ...recentFiles.filter((f) => f.name !== file.name).slice(0, 4),
            ]
            setRecentFiles(newRecent)
            localStorage.setItem('garnet-recent-files', JSON.stringify(newRecent))

            // Store file for later upload
            localStorage.setItem('garnet-pending-file', file.name)
                ; (window as unknown as { pendingFile: File }).pendingFile = file

            setCurrentView('preview')
        },
        [setImage, setCurrentView, recentFiles]
    )

    const handleDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault()
            setIsDragging(false)

            const file = e.dataTransfer.files[0]
            if (file) handleFile(file)
        },
        [handleFile]
    )

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(true)
    }, [])

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)
    }, [])

    const handleClick = () => {
        const input = document.createElement('input')
        input.type = 'file'
        input.accept = SUPPORTED_FILE_TYPES.join(',')
        input.onchange = (e) => {
            const file = (e.target as HTMLInputElement).files?.[0]
            if (file) handleFile(file)
        }
        input.click()
    }

    const handleSample = async () => {
        try {
            const response = await fetch('/sample-pid.png')
            const blob = await response.blob()
            const file = new File([blob], 'sample-pid.png', { type: 'image/png' })
            handleFile(file)
        } catch {
            alert('Failed to load sample P&ID')
        }
    }

    const removeRecentFile = (name: string, e: React.MouseEvent) => {
        e.stopPropagation()
        const newRecent = recentFiles.filter((f) => f.name !== name)
        setRecentFiles(newRecent)
        localStorage.setItem('garnet-recent-files', JSON.stringify(newRecent))
    }

    return (
        <div className="flex h-full flex-col items-center justify-center p-8">
            <div
                role="button"
                tabIndex={0}
                className={`
          flex w-full max-w-2xl cursor-pointer flex-col items-center justify-center
          rounded-xl border-2 border-dashed p-12 transition-all
          ${isDragging
                        ? 'border-primary bg-primary/5 scale-[1.02]'
                        : 'border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50'
                    }
        `}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={handleClick}
                onKeyDown={(e) => e.key === 'Enter' && handleClick()}
            >
                <div
                    className={`
            mb-4 rounded-full p-4 transition-colors
            ${isDragging ? 'bg-primary/10' : 'bg-muted'}
          `}
                >
                    <Upload
                        className={`h-10 w-10 ${isDragging ? 'text-primary' : 'text-muted-foreground'}`}
                    />
                </div>

                <h2 className="mb-2 text-xl font-semibold">Drop P&ID image here</h2>
                <p className="mb-4 text-sm text-muted-foreground">
                    or click to browse
                </p>
                <p className="text-xs text-muted-foreground">
                    Supports: JPG, PNG, PDF
                </p>
            </div>

            <div className="my-6 flex w-full max-w-2xl items-center gap-4">
                <div className="h-px flex-1 bg-border" />
                <span className="text-sm text-muted-foreground">or</span>
                <div className="h-px flex-1 bg-border" />
            </div>

            <Button variant="outline" size="lg" onClick={handleSample}>
                <FileImage className="mr-2 h-4 w-4" />
                Try with Sample P&ID
            </Button>

            {recentFiles.length > 0 && (
                <div className="mt-8 w-full max-w-2xl">
                    <p className="mb-2 text-sm text-muted-foreground">Recent:</p>
                    <div className="flex flex-wrap gap-2">
                        {recentFiles.map((file) => (
                            <div
                                key={file.name}
                                className="flex items-center gap-1 rounded-md bg-muted px-3 py-1.5 text-sm"
                            >
                                <FileImage className="h-3 w-3 text-muted-foreground" />
                                <span className="max-w-[150px] truncate">{file.name}</span>
                                <button
                                    className="ml-1 text-muted-foreground hover:text-foreground"
                                    onClick={(e) => removeRecentFile(file.name, e)}
                                >
                                    <X className="h-3 w-3" />
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
