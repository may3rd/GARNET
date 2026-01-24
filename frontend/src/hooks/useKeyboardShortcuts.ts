import { useEffect } from 'react'
import type { DetectedObject } from '@/types'
import { objectKey } from '@/lib/objectKey'

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName.toLowerCase()
  return tag === 'input' || tag === 'textarea' || tag === 'select' || target.isContentEditable
}

type ShortcutHandlers = {
  objects: DetectedObject[]
  selectedObjectKey: string | null
  onSelectObject: (key: string | null) => void
  onAccept: (key: string) => void
  onReject: (key: string) => void
  onFit: () => void
  onReset: () => void
  onZoomIn: () => void
  onZoomOut: () => void
}

export function useKeyboardShortcuts({
  objects,
  selectedObjectKey,
  onSelectObject,
  onAccept,
  onReject,
  onFit,
  onReset,
  onZoomIn,
  onZoomOut,
}: ShortcutHandlers) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || isEditableTarget(event.target)) return

      if (event.key === 'Tab') {
        event.preventDefault()
        if (!objects.length) return
        const currentIndex = selectedObjectKey
          ? objects.findIndex((obj) => objectKey(obj) === selectedObjectKey)
          : -1
        const direction = event.shiftKey ? -1 : 1
        const nextIndex = currentIndex === -1
          ? 0
          : (currentIndex + direction + objects.length) % objects.length
        const nextKey = objectKey(objects[nextIndex])
        onSelectObject(nextKey)
        return
      }

      if (event.key === 'Enter' && selectedObjectKey) {
        event.preventDefault()
        onAccept(selectedObjectKey)
        return
      }

      if ((event.key === 'Backspace' || event.key === 'Delete') && selectedObjectKey) {
        event.preventDefault()
        onReject(selectedObjectKey)
        return
      }

      if (event.key === 'f' || event.key === 'F') {
        event.preventDefault()
        onFit()
        return
      }

      if (event.key === '0') {
        event.preventDefault()
        onReset()
        return
      }

      if (event.key === '-' || event.key === '_') {
        event.preventDefault()
        onZoomOut()
        return
      }

      if (event.key === '=' || event.key === '+') {
        event.preventDefault()
        onZoomIn()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [
    objects,
    selectedObjectKey,
    onSelectObject,
    onAccept,
    onReject,
    onFit,
    onReset,
    onZoomIn,
    onZoomOut,
  ])
}
