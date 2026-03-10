export type AppView = 'empty' | 'preview' | 'processing' | 'results' | 'batch'
export type ProcessingMode = 'detection' | 'pipeline'
export type OcrRoute = 'easyocr' | 'gemini' | 'paddleocr' | 'ocrmac'

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

export type PipelineReviewDecision = 'accepted' | 'rejected' | 'deferred'
export type PipelineReviewBucket =
  | 'stage4_line_number'
  | 'stage4_instrument'
  | 'stage12_line_attachment'
  | 'stage12_instrument_attachment'

export type PipelineReviewItem = {
  bucket: PipelineReviewBucket
  id: string
  title: string
  subtitle: string
  text: string
  normalizedText: string
  artifactName: string
  statusHint: string
  bbox?: Record<string, unknown>
  ocrSource?: string
  reviewState?: string
  distancePx?: number
  thresholdPx?: number
  edgeId?: string
  sourceObjectId?: string
}

export type PipelineReviewStateItem = {
  item_id: string
  bucket: PipelineReviewBucket
  source_stage?: string | null
  source_artifact?: string | null
  entity_id?: string | null
  decision: PipelineReviewDecision
  reviewer?: string | null
  reason?: string | null
  edited_object?: Record<string, unknown> | null
}

export type PipelineReviewState = {
  job_id: string
  image_path: string | null
  version: number
  updated_at: number
  items: PipelineReviewStateItem[]
  workspace_objects: Record<PipelineReviewBucket, Array<Record<string, unknown>>>
}
