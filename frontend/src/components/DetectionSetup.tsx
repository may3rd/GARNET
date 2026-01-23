import { useEffect, useState } from 'react'
import { Play, SlidersHorizontal } from 'lucide-react'
import { useAppStore } from '@/stores/appStore'
import { getConfigFiles, getModels, getWeightFiles } from '@/lib/api'
import { cn } from '@/lib/utils'

export function DetectionSetup() {
  const options = useAppStore((state) => state.options)
  const setOptions = useAppStore((state) => state.setOptions)
  const runDetection = useAppStore((state) => state.runDetection)
  const error = useAppStore((state) => state.error)
  const [models, setModels] = useState<string[]>([])
  const [weights, setWeights] = useState<string[]>([])
  const [configs, setConfigs] = useState<string[]>([])

  useEffect(() => {
    let active = true
    const load = async () => {
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

      <div className="space-y-4">
        <label className="block text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
          Model
          <select
            value={options.selectedModel}
            onChange={(event) => setOptions({ selectedModel: event.target.value })}
            className={cn(
              'mt-2 w-full rounded-lg border border-[var(--border-muted)]',
              'bg-[var(--bg-primary)] px-3 py-2 text-sm'
            )}
          >
            {(models.length ? models : [options.selectedModel]).map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </label>

        <label className="block text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
          Weight file
          <select
            value={options.weightFile}
            onChange={(event) => setOptions({ weightFile: event.target.value })}
            className={cn(
              'mt-2 w-full rounded-lg border border-[var(--border-muted)]',
              'bg-[var(--bg-primary)] px-3 py-2 text-sm'
            )}
          >
            {(weights.length ? weights : [options.weightFile]).map((weight) => (
              <option key={weight || 'default'} value={weight}>{weight || 'Default'}</option>
            ))}
          </select>
        </label>

        <label className="block text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
          Config file
          <select
            value={options.configFile}
            onChange={(event) => setOptions({ configFile: event.target.value })}
            className={cn(
              'mt-2 w-full rounded-lg border border-[var(--border-muted)]',
              'bg-[var(--bg-primary)] px-3 py-2 text-sm'
            )}
          >
            {(configs.length ? configs : [options.configFile]).map((config) => (
              <option key={config} value={config}>{config}</option>
            ))}
          </select>
        </label>

        <label className="block text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wide">
          Confidence: {Math.round(options.confTh * 100)}%
          <input
            type="range"
            min={0.2}
            max={0.95}
            step={0.01}
            value={options.confTh}
            onChange={(event) => setOptions({ confTh: Number(event.target.value) })}
            className="mt-2 w-full"
          />
        </label>

        <label className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
          <input
            type="checkbox"
            checked={options.textOCR}
            onChange={(event) => setOptions({ textOCR: event.target.checked })}
            className="h-4 w-4"
          />
          Extract text with OCR
        </label>
      </div>

      {error && (
        <div className="text-xs text-[var(--danger)] bg-[var(--bg-primary)] border border-[var(--border-muted)] p-3 rounded-lg">
          {error}
        </div>
      )}

      <button
        onClick={runDetection}
        className={cn(
          'mt-auto px-4 py-3 rounded-lg',
          'bg-[var(--accent-cta)] text-white font-semibold',
          'hover:brightness-95 transition'
        )}
      >
        <span className="inline-flex items-center gap-2">
          <Play className="h-4 w-4" />
          Run Detection
        </span>
      </button>
    </div>
  )
}
