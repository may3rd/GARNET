export type AppView = 'empty' | 'preview' | 'processing' | 'results' | 'batch'

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
