import type { DetectedObject } from '@/types'

export function objectKey(obj: DetectedObject): string {
  return `${obj.CategoryID}-${obj.ObjectID}-${obj.Index}`
}
