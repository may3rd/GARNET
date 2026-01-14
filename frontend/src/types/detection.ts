/**
 * Detection types for GARNET P&ID Recognition
 */

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export type ObjectStatus = 'pending' | 'accepted' | 'rejected'

export interface DetectedObject {
  id: number
  categoryId: number
  categoryName: string
  bbox: BoundingBox
  confidence: number
  ocrText?: string
  status: ObjectStatus
}

export interface DetectionResult {
  objects: DetectedObject[]
  imageUrl: string
  imageName: string
  timestamp: number
}

export interface CategoryInfo {
  id: number
  name: string
  color: string
  hasText: boolean
}
