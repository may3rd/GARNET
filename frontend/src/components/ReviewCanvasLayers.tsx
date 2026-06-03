import type { PipelineReviewWorkspaceState } from '@/types'

type ReviewCanvasLayersProps = {
  workspace: PipelineReviewWorkspaceState | null
  layers: Record<string, Record<string, unknown>>
  visibleLayers: Set<string>
  imageSize: { width: number; height: number } | null
  selectedEntity?: { collection: 'equipment' | 'objects'; id: string } | null
  onSelectEntity?: (entity: { collection: 'equipment' | 'objects'; id: string }) => void
}

type BBoxEntity = {
  id: string
  className: string
  bbox: { x_min: number; y_min: number; x_max: number; y_max: number }
}

type PortEntity = {
  id: string
  x: number
  y: number
}

type Point = { x: number; y: number }

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function bboxFrom(value: unknown): BBoxEntity['bbox'] | null {
  const record = asRecord(value)
  if (!record) return null
  const bbox = asRecord(record.bbox) ?? record
  const xMin = num(bbox.x_min)
  const yMin = num(bbox.y_min)
  const xMax = num(bbox.x_max)
  const yMax = num(bbox.y_max)
  if (xMin === null || yMin === null || xMax === null || yMax === null) return null
  return { x_min: xMin, y_min: yMin, x_max: xMax, y_max: yMax }
}

function boxEntities(items: Array<Record<string, unknown>> | undefined, fallbackPrefix: string): BBoxEntity[] {
  return (items ?? []).flatMap((item, index) => {
    const bbox = bboxFrom(item)
    if (!bbox) return []
    return [{
      id: String(item.id ?? item.Text ?? `${fallbackPrefix}_${index + 1}`),
      className: String(item.class_name ?? item.Object ?? fallbackPrefix),
      bbox,
    }]
  })
}

function pointFrom(value: unknown): Point | null {
  if (Array.isArray(value) && value.length >= 2) {
    const x = num(value[0])
    const y = num(value[1])
    return x === null || y === null ? null : { x, y }
  }
  const record = asRecord(value)
  if (!record) return null
  const x = num(record.x ?? record.terminal_x ?? record.start_x)
  const y = num(record.y ?? record.terminal_y ?? record.start_y)
  return x === null || y === null ? null : { x, y }
}

function collectPorts(value: unknown, out: PortEntity[] = [], limit = 800): PortEntity[] {
  if (out.length >= limit) return out
  const point = pointFrom(value)
  const record = asRecord(value)
  if (point && record && ('port_id' in record || 'id' in record || 'x' in record)) {
    out.push({ id: String(record.port_id ?? record.id ?? `port_${out.length + 1}`), x: point.x, y: point.y })
    return out
  }
  if (Array.isArray(value)) {
    value.forEach((item) => collectPorts(item, out, limit))
  } else if (record) {
    Object.values(record).forEach((item) => collectPorts(item, out, limit))
  }
  return out
}

function arrayToPolyline(value: unknown): Point[] | null {
  if (!Array.isArray(value) || value.length < 2) return null
  const points = value.map(pointFrom)
  if (points.some((point) => point === null)) return null
  return points as Point[]
}

function collectPolylines(value: unknown, out: Point[][] = [], limit = 1200): Point[][] {
  if (out.length >= limit) return out
  const polyline = arrayToPolyline(value)
  if (polyline) {
    out.push(polyline)
    return out
  }
  const record = asRecord(value)
  if (Array.isArray(value)) {
    value.forEach((item) => collectPolylines(item, out, limit))
  } else if (record) {
    Object.entries(record).forEach(([key, item]) => {
      if (key === 'bbox') return
      collectPolylines(item, out, limit)
    })
  }
  return out
}

function pointsAttr(points: Point[]): string {
  return points.map((point) => `${point.x},${point.y}`).join(' ')
}

export function ReviewCanvasLayers({ workspace, layers, visibleLayers, imageSize, selectedEntity, onSelectEntity }: ReviewCanvasLayersProps) {
  if (!imageSize) return null

  const equipment = boxEntities(workspace?.equipment, 'equipment')
  const objects = boxEntities(workspace?.objects, 'object')
  const ports = collectPorts(layers.stage5_connection_ports)
  const traces = collectPolylines(layers.stage5b_trace_results)
  const branches = collectPolylines(layers.stage5b_branch_trace_results)

  return (
    <svg
      className="absolute inset-2 h-[calc(100%-1rem)] w-[calc(100%-1rem)]"
      viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
      preserveAspectRatio="xMinYMin meet"
      aria-hidden="true"
    >
      {visibleLayers.has('traces') ? (
        <g>
          {traces.map((points, index) => (
            <polyline key={`trace-${index}`} points={pointsAttr(points)} fill="none" stroke="#0f9f47" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" opacity="0.78" />
          ))}
        </g>
      ) : null}

      {visibleLayers.has('branches') ? (
        <g>
          {branches.map((points, index) => (
            <polyline key={`branch-${index}`} points={pointsAttr(points)} fill="none" stroke="#dc2626" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" opacity="0.78" />
          ))}
        </g>
      ) : null}

      {visibleLayers.has('equipment') ? (
        <g>
          {equipment.map((item) => (
            <g key={item.id}>
              <rect
                x={item.bbox.x_min}
                y={item.bbox.y_min}
                width={item.bbox.x_max - item.bbox.x_min}
                height={item.bbox.y_max - item.bbox.y_min}
                fill="rgba(2,132,199,0.08)"
                stroke={selectedEntity?.collection === 'equipment' && selectedEntity.id === item.id ? '#f97316' : '#0284c7'}
                strokeWidth={selectedEntity?.collection === 'equipment' && selectedEntity.id === item.id ? '7' : '4'}
                className={onSelectEntity ? 'pointer-events-auto cursor-pointer' : undefined}
                onClick={(event) => {
                  event.stopPropagation()
                  onSelectEntity?.({ collection: 'equipment', id: item.id })
                }}
              />
              <text x={item.bbox.x_min} y={Math.max(14, item.bbox.y_min - 6)} fill="#0369a1" fontSize="22" fontWeight="700">{item.id}</text>
            </g>
          ))}
        </g>
      ) : null}

      {visibleLayers.has('objects') ? (
        <g>
          {objects.map((item) => (
            <g key={item.id}>
              <rect
                x={item.bbox.x_min}
                y={item.bbox.y_min}
                width={item.bbox.x_max - item.bbox.x_min}
                height={item.bbox.y_max - item.bbox.y_min}
                fill="rgba(245,158,11,0.12)"
                stroke={selectedEntity?.collection === 'objects' && selectedEntity.id === item.id ? '#f97316' : '#d97706'}
                strokeWidth={selectedEntity?.collection === 'objects' && selectedEntity.id === item.id ? '7' : '3'}
                className={onSelectEntity ? 'pointer-events-auto cursor-pointer' : undefined}
                onClick={(event) => {
                  event.stopPropagation()
                  onSelectEntity?.({ collection: 'objects', id: item.id })
                }}
              />
              <text x={item.bbox.x_min} y={Math.max(14, item.bbox.y_min - 5)} fill="#b45309" fontSize="18" fontWeight="700">{item.className}</text>
            </g>
          ))}
        </g>
      ) : null}

      {visibleLayers.has('ports') ? (
        <g>
          {ports.map((port, index) => (
            <g key={`${port.id}-${index}`}>
              <circle cx={port.x} cy={port.y} r="9" fill="#06b6d4" stroke="#ffffff" strokeWidth="3" />
              <text x={port.x + 10} y={port.y - 10} fill="#0891b2" fontSize="18" fontWeight="700">{port.id}</text>
            </g>
          ))}
        </g>
      ) : null}
    </svg>
  )
}
