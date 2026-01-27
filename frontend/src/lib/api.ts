import type { DetectionResult, DetectedObject } from '@/types'

export type DetectionOptions = {
  selectedModel: string
  weightFile: string
  configFile: string
  confTh: number
  imageSize: number
  overlapRatio: number
  textOCR: boolean
}

const defaultOptions: DetectionOptions = {
  selectedModel: 'ultralytics',
  weightFile: '',
  configFile: 'datasets/yaml/data.yaml',
  confTh: 0.8,
  imageSize: 640,
  overlapRatio: 0.2,
  textOCR: false,
}

export async function runDetection(
  file: File,
  options: Partial<DetectionOptions> = {},
  signal?: AbortSignal
): Promise<DetectionResult> {
  const payload = { ...defaultOptions, ...options }
  const formData = new FormData()
  formData.append('file_input', file)
  formData.append('selected_model', payload.selectedModel)
  formData.append('weight_file', payload.weightFile)
  formData.append('config_file', payload.configFile)
  formData.append('conf_th', String(payload.confTh))
  formData.append('image_size', String(payload.imageSize))
  formData.append('overlap_ratio', String(payload.overlapRatio))
  formData.append('text_OCR', String(payload.textOCR))

  const response = await fetch('/api/detect', {
    method: 'POST',
    body: formData,
    signal,
  })

  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || 'Detection failed')
  }

  return response.json()
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || 'Request failed')
  }
  return response.json()
}

async function getJson<T>(url: string, fallback: T): Promise<T> {
  try {
    const response = await fetch(url)
    if (!response.ok) {
      return fallback
    }
    return response.json()
  } catch {
    return fallback
  }
}

function normalizeStringList(input: unknown, key?: string): string[] {
  if (!Array.isArray(input)) return []
  const result: string[] = []
  input.forEach((item) => {
    if (typeof item === 'string') {
      result.push(item)
      return
    }
    if (item && typeof item === 'object' && key && key in item) {
      const value = (item as Record<string, unknown>)[key]
      if (typeof value === 'string') {
        result.push(value)
      }
    }
  })
  return result
}

export async function getModels(): Promise<string[]> {
  const data = await getJson<unknown>('/api/models', [])
  const models = normalizeStringList(data, 'value')
  return models.length ? models : ['ultralytics']
}

export async function getWeightFiles(): Promise<string[]> {
  const data = await getJson<unknown>('/api/weight-files', [])
  return normalizeStringList(data, 'item')
}

export async function getConfigFiles(): Promise<string[]> {
  const data = await getJson<unknown>('/api/config-files', [])
  const configs = normalizeStringList(data, 'item')
  return configs.length ? configs : ['datasets/yaml/data.yaml']
}

export async function updateResultObject(
  resultId: string,
  objectId: number,
  payload: Partial<Pick<DetectedObject, 'Object' | 'Left' | 'Top' | 'Width' | 'Height' | 'Text' | 'ReviewStatus'>>
): Promise<DetectedObject> {
  return requestJson(`/api/results/${resultId}/objects/${objectId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export async function deleteResultObject(resultId: string, objectId: number): Promise<{ status: string }> {
  return requestJson(`/api/results/${resultId}/objects/${objectId}`, {
    method: 'DELETE',
  })
}

export async function createResultObject(
  resultId: string,
  payload: Pick<DetectedObject, 'Object' | 'Left' | 'Top' | 'Width' | 'Height' | 'Text'> & Partial<Pick<DetectedObject, 'CategoryID' | 'ObjectID' | 'Score'>>
): Promise<DetectedObject> {
  return requestJson(`/api/results/${resultId}/objects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}
