import { useEffect, useState } from 'react'
import { Play, SlidersHorizontal, Loader2 } from 'lucide-react'
import { useAppStore } from '@/stores/appStore'
import { getConfigFiles, getModels, getWeightFiles } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'

export function DetectionSetup() {
  const options = useAppStore((state) => state.options)
  const setOptions = useAppStore((state) => state.setOptions)
  const runDetection = useAppStore((state) => state.runDetection)
  const error = useAppStore((state) => state.error)
  const [models, setModels] = useState<string[]>([])
  const [weights, setWeights] = useState<string[]>([])
  const [configs, setConfigs] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let active = true
    const load = async () => {
      setIsLoading(true)
      const [modelList, weightList, configList] = await Promise.all([
        getModels(),
        getWeightFiles(),
        getConfigFiles(),
      ])
      if (!active) return
      setModels(modelList)
      setWeights(weightList)
      setConfigs(configList)
      if (!options.weightFile && weightList.length) {
        setOptions({ weightFile: weightList[0] })
      }
      if (!options.configFile && configList.length) {
        setOptions({ configFile: configList[0] })
      }
      setIsLoading(false)
    }
    load()
    return () => {
      active = false
    }
  }, [])

  return (
    <div className="p-6 flex flex-col gap-6">
      <div className="flex items-center gap-2 text-sm font-semibold">
        <SlidersHorizontal className="h-4 w-4 text-[var(--accent)]" />
        Detection Setup
      </div>

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
            onValueChange={(value) => setOptions({ selectedModel: value })}
            disabled={isLoading}
          >
            <SelectTrigger className="mt-2">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              {(models.length ? models : [options.selectedModel])
                .filter((model) => model !== 'azure_custom_vision')
                .map((model) => (
                  <SelectItem key={model} value={model}>
                    {model}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        </label>

        <label className="block text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
          Weight file
          <Select
            value={options.weightFile || '_default'}
            onValueChange={(value) => setOptions({ weightFile: value === '_default' ? '' : value })}
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

        <label className="block text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
          Config file
          <Select
            value={options.configFile}
            onValueChange={(value) => setOptions({ configFile: value })}
          >
            <SelectTrigger className="mt-2">
              <SelectValue placeholder="Select config file" />
            </SelectTrigger>
            <SelectContent>
              {(configs.length ? configs : [options.configFile]).map((config) => (
                <SelectItem key={config} value={config}>
                  {config}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>

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
          />
        </div>

        <label className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
          <input
            type="checkbox"
            checked={options.textOCR}
            onChange={(event) => setOptions({ textOCR: event.target.checked })}
            className="h-4 w-4"
          />
          Extract text with OCR
        </label>

        {error && (
          <div className="text-xs text-[var(--danger)] bg-[var(--bg-primary)] border border-[var(--border-muted)] p-3 rounded-lg">
            {error}
          </div>
        )}

        <Button onClick={runDetection} variant="cta" className="mt-auto" disabled={isLoading}>
          <Play className="h-4 w-4" />
          Run Detection
        </Button>
      </div>
      )}
    </div>
  )
}
