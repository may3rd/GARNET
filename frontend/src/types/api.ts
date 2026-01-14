/**
 * API types for GARNET backend communication
 */

export interface DetectionRequest {
    file: File
    modelType: string        // e.g. "ultralytics", "azure_custom_vision"
    weightFile: string       // path to weight file
    configFile: string       // path to config YAML
    confidence: number
    imageSize: number
    enableOcr: boolean
}

export interface APIDetectedObject {
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

export interface DetectionResponse {
    objects: APIDetectedObject[]
    image_url: string
}

export interface ModelType {
    name: string
    value: string
}

export interface WeightFile {
    item: string  // full path to weight file
}

export interface ConfigFile {
    item: string  // full path to config file
}

export interface HealthResponse {
    status: string
}
