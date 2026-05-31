import { useState } from 'react'
import type { DetectedObject } from '@/types'

export type EditDraft = {
  Object: string
  Left: number
  Top: number
  Width: number
  Height: number
  Text: string
}

export function buildDraft(obj: DetectedObject): EditDraft {
  return {
    Object: obj.Object,
    Left: obj.Left,
    Top: obj.Top,
    Width: obj.Width,
    Height: obj.Height,
    Text: obj.Text ?? '',
  }
}

export function useInlineEdit() {
  const [draft, setDraft] = useState<EditDraft | null>(null)
  const [isEditing, setIsEditing] = useState(false)

  const startEditing = (obj: DetectedObject) => {
    setDraft(buildDraft(obj))
    setIsEditing(true)
  }

  const cancelEditing = () => {
    setDraft(null)
    setIsEditing(false)
  }

  return {
    draft,
    isEditing,
    setDraft,
    startEditing,
    cancelEditing,
  }
}
