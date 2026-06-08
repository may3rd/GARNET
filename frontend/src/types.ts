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
  SourceItemId?: string
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
  status: 'pending' | 'started' | 'running' | 'completed' | 'failed' | 'stale'
  started_at?: number
  ended_at?: number
  duration_sec?: number
  artifacts?: string[]
  error?: string
  stale_reason?: string
  stale_source_artifact?: string
  stale_at?: number
}

export type PipelineManifest = {
  image_path: string
  out_dir: string
  stop_after: number
  ocr_route: OcrRoute
  detection_weight_path?: string
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
  weight_file?: string
  debug_artifacts?: boolean
  gemini_postprocess_match_threshold?: number
  manifest: PipelineManifest | null
  artifacts: PipelineArtifact[]
}

export type PipelineReviewDecision = 'accepted' | 'rejected' | 'deferred'
export type PipelineReviewBucket =
  | 'stage3_equipment'
  | 'stage4_object'
  | 'stage4_line_number'
  | 'stage4_instrument'
  | 'stage6_line_association'
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

export type PipelineReviewWorkspaceState = {
  version: number
  job_id?: string | null
  image_id?: string | null
  updated_at: number
  objects: Array<Record<string, unknown>>
  equipment: Array<Record<string, unknown>>
  manual_ports: Array<Record<string, unknown>>
  deleted_entities: Array<Record<string, unknown>>
  line_association_overrides: Array<Record<string, unknown>>
  trace_overrides: Array<Record<string, unknown>>
}

export type PipelineManualPort = {
  port_id: string
  owner_id: string
  owner_type: 'equipment' | 'object' | string
  x: number
  y: number
  direction: 'UP' | 'DOWN' | 'LEFT' | 'RIGHT'
  source?: string
  review_state?: string
}

export type PipelineReviewWorkspaceResponse = {
  job_id: string
  workspace: PipelineReviewWorkspaceState
  artifact: PipelineArtifact
}

export type PipelineReviewRecomputeResponse = {
  job_id: string
  workspace: PipelineReviewWorkspaceState
  layers: Record<string, Record<string, unknown>>
  stages: PipelineStageManifest[]
}

export type PipelineReviewCommitResponse = {
  job_id: string
  workspace: PipelineReviewWorkspaceState
  stages: PipelineStageManifest[]
}

export type Stage3EquipmentArtifact = {
  equipment: Array<{
    id: string
    class_name: string
    bbox: {
      x_min: number
      y_min: number
      x_max: number
      y_max: number
    }
    source?: string
    review_state?: string
  }>
}

export type PipelineStageStatusResponse = {
  job_id: string
  stages: PipelineStageManifest[]
}

export type PipelineReviewedGraphResponse = {
  graph: Record<string, unknown>
  summary: Record<string, unknown>
}

export type PipelineReviewedQaResponse = {
  anomaly_report: Record<string, unknown>
  review_queue: Record<string, unknown>
  summary: Record<string, unknown>
}
