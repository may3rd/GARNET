export type AppView = 'empty' | 'preview' | 'processing' | 'results'

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
}

export type DetectionResult = {
  id: string
  objects: DetectedObject[]
  image_url: string
  image_width: number
  image_height: number
  count: number
}
