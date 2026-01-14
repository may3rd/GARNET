/**
 * FilePreview component - image preview with detection setup
 */
import { useState, useEffect } from 'react'
import { Play, Info, FileCode, Weight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TooltipProvider,
} from '@/components/ui/tooltip'
import { useDetectionStore } from '@/stores/detectionStore'
import { useUIStore } from '@/stores/uiStore'
import { getModelTypes, getWeightFiles, getConfigFiles, detectObjects } from '@/lib/api'
import {
    DEFAULT_CONFIDENCE,
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
} from '@/lib/constants'
import type { ModelType, WeightFile, ConfigFile } from '@/types/api'

export function FilePreview() {
    const [modelTypes, setModelTypes] = useState<ModelType[]>([])
    const [weightFiles, setWeightFiles] = useState<WeightFile[]>([])
    const [configFiles, setConfigFiles] = useState<ConfigFile[]>([])

    const [selectedModelType, setSelectedModelType] = useState('ultralytics')
    const [selectedWeightFile, setSelectedWeightFile] = useState('')
    const [selectedConfigFile, setSelectedConfigFile] = useState('')
    const [confidence, setConfidence] = useState(DEFAULT_CONFIDENCE)
    const [imageSize, setImageSize] = useState(640)
    const [enableOcr, setEnableOcr] = useState(false)
    const [isAdvancedOpen, setIsAdvancedOpen] = useState(false)

    const { imageUrl, imageName, setObjects, setLoading, setProgress, setImage } =
        useDetectionStore()
    const { setCurrentView } = useUIStore()

    // Fetch available models, weights, and configs
    useEffect(() => {
        Promise.all([getModelTypes(), getWeightFiles(), getConfigFiles()])
            .then(([types, weights, configs]) => {
                // Safely set state with fallbacks
                setModelTypes(Array.isArray(types) ? types : [])
                setWeightFiles(Array.isArray(weights) ? weights : [])
                setConfigFiles(Array.isArray(configs) ? configs : [])

                // Set defaults if available
                if (Array.isArray(weights) && weights.length > 0) {
                    setSelectedWeightFile(weights[0].item)
                }
                if (Array.isArray(configs) && configs.length > 0) {
                    setSelectedConfigFile(configs[0].item)
                }
            })
            .catch((error) => {
                console.warn('Backend API not available:', error)
                // Fallback defaults if API not available
                setModelTypes([
                    { name: 'ultralytics', value: 'ultralytics' },
                    { name: 'azure_custom_vision', value: 'azure_custom_vision' },
                ])
            })
    }, [])

    // Helper to extract filename from path
    const getFilename = (path: string) => {
        return path.split('/').pop() || path
    }

    const handleRunDetection = async () => {
        const pendingFile = (window as unknown as { pendingFile: File }).pendingFile
        if (!pendingFile) {
            alert('No file selected')
            return
        }

        if (!selectedWeightFile) {
            alert('Please select a weight file')
            return
        }

        // Clear previous detection results
        setObjects([])

        setLoading(true)
        setCurrentView('processing')
        setProgress({ step: 'Uploading image...', percent: 0, objectCount: 0 })

        try {
            const objects = await detectObjects(
                {
                    file: pendingFile,
                    modelType: selectedModelType,
                    weightFile: selectedWeightFile,
                    configFile: selectedConfigFile,
                    confidence,
                    imageSize,
                    enableOcr,
                },
                (percent) => {
                    setProgress({ step: 'Processing...', percent, objectCount: 0 })
                }
            )

            // Update image URL to the processed result with cache-busting timestamp
            const cacheBuster = Date.now()
            setImage(`/static/images/prediction_results.png?t=${cacheBuster}`, imageName || 'result')
            setObjects(objects)
            setProgress(null)
            setLoading(false)
            setCurrentView('results')
        } catch (error) {
            console.error('Detection failed:', error)
            setProgress(null)
            setLoading(false)
            alert('Detection failed. Please check if the backend is running.')
            setCurrentView('preview')
        }
    }

    return (
        <TooltipProvider>
            <div className="flex h-full">
                {/* Image Preview */}
                <div className="flex flex-1 items-center justify-center bg-muted/30 p-8">
                    <div className="relative max-h-[70vh] max-w-[70vw] overflow-hidden rounded-lg border bg-background shadow-lg">
                        {imageUrl && (
                            <img
                                src={imageUrl}
                                alt={imageName || 'P&ID Preview'}
                                className="h-auto max-h-[70vh] w-auto object-contain"
                            />
                        )}
                    </div>
                </div>

                {/* Detection Setup Panel */}
                <div className="w-96 border-l bg-background p-6 overflow-y-auto">
                    <h2 className="mb-6 text-lg font-semibold">Detection Setup</h2>

                    <div className="space-y-5">
                        {/* Model Type Selection */}
                        <div className="space-y-2">
                            <label className="flex items-center gap-2 text-sm font-medium">
                                <FileCode className="h-4 w-4 text-muted-foreground" />
                                Model Type
                            </label>
                            <select
                                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                                value={selectedModelType}
                                onChange={(e) => setSelectedModelType(e.target.value)}
                            >
                                {modelTypes.map((model) => (
                                    <option key={model.value} value={model.value}>
                                        {model.name}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Weight File Selection */}
                        <div className="space-y-2">
                            <label className="flex items-center gap-2 text-sm font-medium">
                                <Weight className="h-4 w-4 text-muted-foreground" />
                                Weight File
                            </label>
                            <select
                                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                                value={selectedWeightFile}
                                onChange={(e) => setSelectedWeightFile(e.target.value)}
                            >
                                {weightFiles.length === 0 ? (
                                    <option value="">No weight files found</option>
                                ) : (
                                    weightFiles.map((weight) => (
                                        <option key={weight.item} value={weight.item}>
                                            {getFilename(weight.item)}
                                        </option>
                                    ))
                                )}
                            </select>
                            {selectedWeightFile && (
                                <p className="text-xs text-muted-foreground truncate">
                                    {selectedWeightFile}
                                </p>
                            )}
                        </div>

                        {/* Config File Selection */}
                        <div className="space-y-2">
                            <label className="flex items-center gap-2 text-sm font-medium">
                                <FileCode className="h-4 w-4 text-muted-foreground" />
                                Config File
                            </label>
                            <select
                                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                                value={selectedConfigFile}
                                onChange={(e) => setSelectedConfigFile(e.target.value)}
                            >
                                {configFiles.length === 0 ? (
                                    <option value="">No config files found</option>
                                ) : (
                                    configFiles.map((config) => (
                                        <option key={config.item} value={config.item}>
                                            {getFilename(config.item)}
                                        </option>
                                    ))
                                )}
                            </select>
                        </div>

                        {/* Confidence Threshold */}
                        <div className="space-y-2">
                            <div className="flex items-center justify-between">
                                <label className="text-sm font-medium">Confidence</label>
                                <span className="text-sm text-muted-foreground">
                                    {Math.round(confidence * 100)}%
                                </span>
                            </div>
                            <Slider
                                value={[confidence]}
                                onValueChange={([v]) => setConfidence(v)}
                                min={MIN_CONFIDENCE}
                                max={MAX_CONFIDENCE}
                                step={0.05}
                            />
                        </div>

                        {/* OCR Toggle */}
                        <div className="flex items-center gap-3">
                            <input
                                type="checkbox"
                                id="ocr"
                                checked={enableOcr}
                                onChange={(e) => setEnableOcr(e.target.checked)}
                                className="h-4 w-4 rounded border-gray-300"
                            />
                            <label htmlFor="ocr" className="text-sm font-medium">
                                Extract text (OCR)
                            </label>
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <Info className="h-4 w-4 text-muted-foreground" />
                                </TooltipTrigger>
                                <TooltipContent>
                                    Enable text extraction for instrument tags and line numbers
                                </TooltipContent>
                            </Tooltip>
                        </div>

                        <div className="h-px bg-border" />

                        {/* Run Detection Button */}
                        <Button className="w-full" size="lg" onClick={handleRunDetection}>
                            <Play className="mr-2 h-4 w-4" />
                            Run Detection
                        </Button>

                        {/* Advanced Options */}
                        <button
                            className="flex w-full items-center justify-between text-sm text-muted-foreground hover:text-foreground"
                            onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
                        >
                            <span>Advanced</span>
                            <span>{isAdvancedOpen ? '▲' : '▼'}</span>
                        </button>

                        {isAdvancedOpen && (
                            <div className="space-y-4 rounded-md bg-muted/50 p-4">
                                {/* Image Size */}
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Image Size</label>
                                    <select
                                        className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                                        value={imageSize}
                                        onChange={(e) => setImageSize(Number(e.target.value))}
                                    >
                                        <option value={320}>320px</option>
                                        <option value={640}>640px (default)</option>
                                        <option value={1280}>1280px</option>
                                    </select>
                                    <p className="text-xs text-muted-foreground">
                                        Larger size = more accurate but slower
                                    </p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </TooltipProvider>
    )
}
