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

// Default timeout values
const DEFAULT_TIMEOUT = 300000 // 5 minutes for detection (can be slow for large images)
const DEFAULT_REQUEST_TIMEOUT = 30000 // 30 seconds for regular API requests

function createTimeoutSignal(timeoutMs: number): AbortSignal {
  const controller = new AbortController()
  setTimeout(() => controller.abort(), timeoutMs)
  return controller.signal
}

type RequestSignalBundle = {
  signal: AbortSignal
  cleanup: () => void
  isCanceled: () => boolean
}

function createRequestSignal(parentSignal: AbortSignal | undefined, timeoutMs: number): RequestSignalBundle {
  let canceled = false
  const timeoutController = new AbortController()
  const timeoutId = window.setTimeout(() => {
    timeoutController.abort()
  }, timeoutMs)

  if (!parentSignal) {
    return {
      signal: timeoutController.signal,
      cleanup: () => window.clearTimeout(timeoutId),
      isCanceled: () => false,
    }
  }

  const combinedController = new AbortController()

  const abortFromParent = () => {
    if (combinedController.signal.aborted) return
    canceled = true
    combinedController.abort()
  }

  const abortFromTimeout = () => {
    if (combinedController.signal.aborted) return
    combinedController.abort()
  }

  parentSignal.addEventListener('abort', abortFromParent, { once: true })
  timeoutController.signal.addEventListener('abort', abortFromTimeout, { once: true })

  if (parentSignal.aborted) {
    abortFromParent()
  }

  return {
    signal: combinedController.signal,
    cleanup: () => {
      window.clearTimeout(timeoutId)
      parentSignal.removeEventListener('abort', abortFromParent)
      timeoutController.signal.removeEventListener('abort', abortFromTimeout)
    },
    isCanceled: () => canceled,
  }
}

/**
 * Custom error class for API errors with timeout support
 */
export class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public isTimeout = false,
    public isCanceled = false
  ) {
    super(message)
    this.name = 'APIError'
  }
}

export async function runDetection(
  file: File,
  options: Partial<DetectionOptions> = {},
  signal?: AbortSignal,
  timeoutMs = DEFAULT_TIMEOUT
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

  const requestSignal = createRequestSignal(signal, timeoutMs)

  try {
    const response = await fetch('/api/detect', {
      method: 'POST',
      body: formData,
      signal: requestSignal.signal,
    })

    if (!response.ok) {
      const message = await response.text()
      throw new APIError(message || 'Detection failed', response.status)
    }

    return response.json()
  } catch (error) {
    if (error instanceof APIError) {
      throw error
    }
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        if (requestSignal.isCanceled()) {
          throw new APIError('Detection canceled', undefined, false, true)
        }
        throw new APIError(
          `Request timed out after ${timeoutMs / 1000} seconds. The detection may be taking too long for this image.`,
          undefined,
          true
        )
      }
      throw new APIError(error.message)
    }
    throw new APIError('An unknown error occurred')
  } finally {
    requestSignal.cleanup()
  }
}

async function requestJson<T>(
  url: string,
  init?: RequestInit,
  timeoutMs = DEFAULT_REQUEST_TIMEOUT
): Promise<T> {
  const requestSignal = createRequestSignal(init?.signal, timeoutMs)

  try {
    const response = await fetch(url, {
      ...init,
      signal: requestSignal.signal,
    })
    if (!response.ok) {
      const message = await response.text()
      throw new APIError(message || 'Request failed', response.status)
    }
    return response.json()
  } catch (error) {
    if (error instanceof APIError) {
      throw error
    }
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        if (requestSignal.isCanceled()) {
          throw new APIError('Request canceled', undefined, false, true)
        }
        throw new APIError(
          `Request timed out after ${timeoutMs / 1000} seconds`,
          undefined,
          true
        )
      }
      throw new APIError(error.message)
    }
    throw new APIError('An unknown error occurred')
  } finally {
    requestSignal.cleanup()
  }
}

async function getJson<T>(url: string, fallback: T, timeoutMs = DEFAULT_REQUEST_TIMEOUT): Promise<T> {
  try {
    const response = await fetch(url, {
      signal: createTimeoutSignal(timeoutMs),
    })
    if (!response.ok) {
      return fallback
    }
    return response.json()
  } catch (error) {
    // Return fallback on any error (including timeout)
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

export async function getModels(timeoutMs = DEFAULT_REQUEST_TIMEOUT): Promise<string[]> {
  const data = await getJson<unknown>('/api/models', [], timeoutMs)
  const models = normalizeStringList(data, 'value')
  return models.length ? models : ['ultralytics', 'gemini']
}

export async function getWeightFiles(timeoutMs = DEFAULT_REQUEST_TIMEOUT): Promise<string[]> {
  const data = await getJson<unknown>('/api/weight-files', [], timeoutMs)
  return normalizeStringList(data, 'item')
}

export async function getConfigFiles(timeoutMs = DEFAULT_REQUEST_TIMEOUT): Promise<string[]> {
  const data = await getJson<unknown>('/api/config-files', [], timeoutMs)
  const configs = normalizeStringList(data, 'item')
  return configs.length ? configs : ['datasets/yaml/data.yaml']
}

/**
 * Check API health status
 */
export async function checkHealth(timeoutMs = 10000): Promise<{
  status: string
  service: string
  models_loaded: boolean
  models_available: number
}> {
  return requestJson('/api/health', {}, timeoutMs)
}

export async function updateResultObject(
  resultId: string,
  objectId: number,
  payload: Partial<Pick<DetectedObject, 'Object' | 'Left' | 'Top' | 'Width' | 'Height' | 'Text' | 'ReviewStatus'>>,
  timeoutMs = DEFAULT_REQUEST_TIMEOUT
): Promise<DetectedObject> {
  return requestJson(
    `/api/results/${resultId}/objects/${objectId}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
    timeoutMs
  )
}

export async function deleteResultObject(
  resultId: string,
  objectId: number,
  timeoutMs = DEFAULT_REQUEST_TIMEOUT
): Promise<{ status: string }> {
  return requestJson(
    `/api/results/${resultId}/objects/${objectId}`,
    {
      method: 'DELETE',
    },
    timeoutMs
  )
}

export async function createResultObject(
  resultId: string,
  payload: Pick<DetectedObject, 'Object' | 'Left' | 'Top' | 'Width' | 'Height' | 'Text'> &
    Partial<Pick<DetectedObject, 'CategoryID' | 'ObjectID' | 'Score'>>,
  timeoutMs = DEFAULT_REQUEST_TIMEOUT
): Promise<DetectedObject> {
  return requestJson(
    `/api/results/${resultId}/objects`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
    timeoutMs
  )
}

export type PdfExtractResult = {
  count: number
  pages: string[] // base64-encoded PNG images
}

/**
 * Extract pages from a PDF file as base64-encoded PNG images.
 * Conversion settings (DPI/page limit) are enforced by backend configuration.
 */
export async function extractPdfPages(
  file: File,
  timeoutMs = 120000 // 2 minutes for potentially large PDFs
): Promise<PdfExtractResult> {
  const formData = new FormData()
  formData.append('file_input', file)

  const response = await fetch('/api/pdf-extract', {
    method: 'POST',
    body: formData,
    signal: createTimeoutSignal(timeoutMs),
  })

  if (!response.ok) {
    const message = await response.text()
    throw new APIError(message || 'PDF extraction failed', response.status)
  }

  return response.json()
}
