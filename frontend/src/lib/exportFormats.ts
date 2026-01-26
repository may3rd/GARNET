import type { DetectedObject } from '@/types'

export type ExportFormat = 'json' | 'yolo' | 'coco' | 'labelme'

export function exportYolo(objects: DetectedObject[], imageWidth: number, imageHeight: number): string {
  const lines: string[] = []
  for (const obj of objects) {
    const classId = Number.isFinite(obj.CategoryID) ? obj.CategoryID : 0
    const xCenter = (obj.Left + obj.Width / 2) / imageWidth
    const yCenter = (obj.Top + obj.Height / 2) / imageHeight
    const w = obj.Width / imageWidth
    const h = obj.Height / imageHeight
    lines.push(
      `${classId} ${xCenter.toFixed(6)} ${yCenter.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`
    )
  }
  return lines.join('\n') + (lines.length ? '\n' : '')
}

export function exportCoco(
  objects: DetectedObject[],
  imageWidth: number,
  imageHeight: number,
  fileName: string,
) {
  const imageId = 1
  const categoriesMap = new Map<number, string>()
  for (const obj of objects) {
    if (!categoriesMap.has(obj.CategoryID)) {
      categoriesMap.set(obj.CategoryID, obj.Object)
    }
  }

  const categories = Array.from(categoriesMap.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([id, name]) => ({ id, name, supercategory: 'object' }))

  const annotations = objects.map((obj, idx) => ({
    id: idx + 1,
    image_id: imageId,
    category_id: obj.CategoryID,
    bbox: [obj.Left, obj.Top, obj.Width, obj.Height],
    area: obj.Width * obj.Height,
    iscrowd: 0,
    score: obj.Score,
    text: obj.Text,
  }))

  return {
    info: {
      description: 'GARNET export',
      version: '1.0',
      year: new Date().getFullYear(),
    },
    images: [
      {
        id: imageId,
        file_name: fileName,
        width: imageWidth,
        height: imageHeight,
      },
    ],
    annotations,
    categories,
  }
}

export function exportLabelMe(
  objects: DetectedObject[],
  imageWidth: number,
  imageHeight: number,
  imagePath: string,
) {
  const shapes = objects.map((obj) => ({
    label: obj.Object,
    points: [
      [obj.Left, obj.Top],
      [obj.Left + obj.Width, obj.Top + obj.Height],
    ],
    group_id: null,
    shape_type: 'rectangle',
    flags: {},
    description: obj.Text ?? '',
  }))

  return {
    version: '5.0.1',
    flags: {},
    shapes,
    imagePath,
    imageData: null,
    imageHeight,
    imageWidth,
  }
}

