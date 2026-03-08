export type AppView = 'empty' | 'preview' | 'processing' | 'results' | 'batch'
export type ProcessingMode = 'detection' | 'pipeline'
export type OcrRoute = 'easyocr' | 'gemini'

export type DetectedObject = {
  Index: number
  Object: string
  CategoryID: number
  ObjectID: number
  Left: number
  Top: number
  Width: number
  Height: number
  Score: number
  Text: string
  ReviewStatus?: 'accepted' | 'rejected' | null
}

export type DetectionResult = {
  id: string
  objects: DetectedObject[]
  image_url: string
  image_width: number
  image_height: number
  count: number
}

export type BatchItemStatus = 'queued' | 'running' | 'done' | 'failed' | 'canceled'

export type BatchItem = {
  id: string
  file: File
  fileName: string
  status: BatchItemStatus
  result?: DetectionResult
  error?: string
}

export type PipelineArtifact = {
  name: string
  url: string
}

export type PipelineStageManifest = {
  num: number
  name: string
  status: 'started' | 'completed' | 'failed'
  started_at: number
  ended_at?: number
  duration_sec?: number
  artifacts: string[]
  error?: string
}

export type PipelineManifest = {
  image_path: string
  out_dir: string
  stop_after: number
  ocr_route: OcrRoute
  stages: PipelineStageManifest[]
}

export type PipelineJob = {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  current_stage: string | null
  error: string | null
  job_dir: string
  created_at: number
  stop_after: number
  ocr_route: OcrRoute
  gemini_postprocess_match_threshold?: number
  manifest: PipelineManifest | null
  artifacts: PipelineArtifact[]
}
