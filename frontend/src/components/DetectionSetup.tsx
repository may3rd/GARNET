import { useEffect, useState } from 'react'
import { Play, SlidersHorizontal, Loader2 } from 'lucide-react'
import { useAppStore } from '@/stores/appStore'
import { getModels, getWeightFiles } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Checkbox } from '@/components/ui/checkbox'

export function DetectionSetup() {
  const options = useAppStore((state) => state.options)
  const setOptions = useAppStore((state) => state.setOptions)
  const runDetection = useAppStore((state) => state.runDetection)
  const runBatchDetection = useAppStore((state) => state.runBatchDetection)
  const batch = useAppStore((state) => state.batch)
  const error = useAppStore((state) => state.error)
  const [models, setModels] = useState<string[]>([])
  const [weights, setWeights] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let active = true
    const load = async () => {
      setIsLoading(true)
      const [modelList, weightList] = await Promise.all([getModels(), getWeightFiles()])
      if (!active) return
      setModels(modelList)
      setWeights(weightList)
      if (options.selectedModel !== 'gemini' && !options.weightFile && weightList.length) {
        setOptions({ weightFile: weightList[0] })
      }
      setIsLoading(false)
    }
    load()
    return () => {
      active = false
    }
  }, [])

  const isBatchMode = batch.items.length > 0
  const isLocked = batch.locked
  const isGeminiModel = options.selectedModel === 'gemini'
  const hasRunnableBatchItems = batch.items.some((item) => item.status === 'queued' || item.status === 'failed')
  const runAction = isBatchMode ? runBatchDetection : runDetection
  const runLabel = isBatchMode ? `Run Batch (${batch.items.length})` : 'Run Detection'

  const handleModelChange = (value: string) => {
    if (value === 'gemini') {
      setOptions({ selectedModel: value, weightFile: '' })
      return
    }
    setOptions({
      selectedModel: value,
      weightFile: options.weightFile || weights[0] || '',
    })
  }

  return (
    <div className="p-6 flex flex-col gap-6">
      <div className="flex items-center gap-2 text-sm font-semibold">
        <SlidersHorizontal className="h-4 w-4 text-[var(--accent)]" />
        Detection Setup
      </div>

      {isBatchMode && (
        <div className="text-xs text-[var(--text-secondary)] bg-[var(--bg-primary)] border border-[var(--border-muted)] p-3 rounded-lg">
          Batch mode is active. The selected model and settings apply to all images in the batch.
        </div>
      )}

      {isLoading ? (
        <div className="flex items-center justify-center py-8 text-[var(--text-secondary)]">
          <Loader2 className="h-5 w-5 animate-spin mr-2" />
          <span className="text-sm">Loading configuration...</span>
        </div>
      ) : (
      <div className="space-y-4">
        <label className="block text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
          Model
          <Select
            value={options.selectedModel}
            onValueChange={handleModelChange}
            disabled={isLoading || isLocked}
          >
            <SelectTrigger className="mt-2">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              {(models.length ? models : [options.selectedModel]).map((model) => (
                <SelectItem key={model} value={model}>
                  {model}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>

        {!isGeminiModel ? (
          <label className="block text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
            Weight file
            <Select
              value={options.weightFile || '_default'}
              onValueChange={(value) => setOptions({ weightFile: value === '_default' ? '' : value })}
              disabled={isLocked}
            >
              <SelectTrigger className="mt-2">
                <SelectValue placeholder="Select weight file" />
              </SelectTrigger>
              <SelectContent>
                {weights.length === 0 && (
                  <SelectItem value="_default">Default</SelectItem>
                )}
                {weights.map((weight) => (
                  <SelectItem key={weight} value={weight}>
                    {weight || 'Default'}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </label>
        ) : (
          <div className="text-xs text-[var(--text-secondary)] bg-[var(--bg-primary)] border border-[var(--border-muted)] p-3 rounded-lg">
            Gemini model uses OpenRouter LLM inference and does not require a weight file.
          </div>
        )}

        <div className="space-y-2">
          <div className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
            Confidence: {Math.round(options.confTh * 100)}%
          </div>
          <Slider
            value={[options.confTh]}
            onValueChange={([value]) => setOptions({ confTh: value })}
            min={0.2}
            max={0.95}
            step={0.01}
            className="mt-2"
            disabled={isLocked}
          />
        </div>

        <div className="space-y-2">
          <div className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
            Overlay ratio: {options.overlapRatio.toFixed(2)}
          </div>
          <Slider
            value={[options.overlapRatio]}
            onValueChange={([value]) => setOptions({ overlapRatio: value })}
            min={0}
            max={0.5}
            step={0.01}
            className="mt-2"
            disabled={isLocked}
          />
        </div>

        <div className="space-y-2">
          <div className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
            Image size: {options.imageSize}px
          </div>
          <Slider
            value={[options.imageSize]}
            onValueChange={([value]) => setOptions({ imageSize: value })}
            min={128}
            max={1280}
            step={32}
            className="mt-2"
            disabled={isLocked}
          />
        </div>

        <label className="block text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
          Postprocess type
          <Select
            value={options.postprocessType}
            onValueChange={(value) => setOptions({ postprocessType: value as 'NMM' | 'GREEDYNMM' | 'NMS' })}
            disabled={isLocked}
          >
            <SelectTrigger className="mt-2">
              <SelectValue placeholder="Select postprocess type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="NMM">NMM</SelectItem>
              <SelectItem value="GREEDYNMM">GREEDYNMM</SelectItem>
              <SelectItem value="NMS">NMS</SelectItem>
            </SelectContent>
          </Select>
        </label>

        <label className="block text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
          Postprocess match metric
          <Select
            value={options.postprocessMatchMetric}
            onValueChange={(value) => setOptions({ postprocessMatchMetric: value as 'IOU' | 'IOS' })}
            disabled={isLocked}
          >
            <SelectTrigger className="mt-2">
              <SelectValue placeholder="Select match metric" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="IOU">IOU</SelectItem>
              <SelectItem value="IOS">IOS</SelectItem>
            </SelectContent>
          </Select>
        </label>

        <div className="space-y-2">
          <div className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
            Postprocess match threshold: {options.postprocessMatchThreshold.toFixed(2)}
          </div>
          <Slider
            value={[options.postprocessMatchThreshold]}
            onValueChange={([value]) => setOptions({ postprocessMatchThreshold: value })}
            min={0}
            max={1}
            step={0.01}
            className="mt-2"
            disabled={isLocked}
          />
        </div>

        <div className="flex items-center gap-2">
          <Checkbox
            id="textOCR"
            checked={options.textOCR}
            onCheckedChange={(checked) => setOptions({ textOCR: checked as boolean })}
            disabled={isLocked}
          />
          <label htmlFor="textOCR" className="text-sm text-[var(--text-secondary)] cursor-pointer">
            Extract text with OCR
          </label>
        </div>

        {error && (
          <div className="text-xs text-[var(--danger)] bg-[var(--bg-primary)] border border-[var(--border-muted)] p-3 rounded-lg">
            {error}
          </div>
        )}

        <Button
          onClick={runAction}
          variant="cta"
          className="mt-auto"
          disabled={isLoading || isLocked || (isBatchMode && !hasRunnableBatchItems)}
        >
          <Play className="h-4 w-4" />
          {runLabel}
        </Button>
      </div>
      )}
    </div>
  )
}
