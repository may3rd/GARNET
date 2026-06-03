import type { PipelineReviewWorkspaceState } from '@/types'

type ReviewCanvasLayersProps = {
  workspace: PipelineReviewWorkspaceState | null
  layers: Record<string, Record<string, unknown>>
  visibleLayers: Set<string>
  imageSize: { width: number; height: number } | null
  selectedEntity?: { collection: 'equipment' | 'objects'; id: string } | null
  onSelectEntity?: (entity: { collection: 'equipment' | 'objects'; id: string }) => void
  embedded?: boolean
  showBoxes?: boolean
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
type TraceEntity = {
  id: string
  segments: Point[][]
  terminal?: Point
  terminalType?: string
}

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

function collectPorts(value: unknown, out: PortEntity[] = [], limit = 800, ownerId?: string): PortEntity[] {
  if (out.length >= limit) return out
  const point = pointFrom(value)
  const record = asRecord(value)
  if (point && record && ('port_id' in record || 'id' in record || 'x' in record)) {
    out.push({ id: String(record.port_id ?? record.id ?? ownerId ?? `port_${out.length + 1}`), x: point.x, y: point.y })
    return out
  }
  if (point && Array.isArray(value)) {
    out.push({ id: ownerId ?? `port_${out.length + 1}`, x: point.x, y: point.y })
    return out
  }
  if (Array.isArray(value)) {
    value.forEach((item) => collectPorts(item, out, limit, ownerId))
  } else if (record) {
    Object.entries(record).forEach(([key, item]) => collectPorts(item, out, limit, key))
  }
  return out
}

function segmentPolyline(value: unknown): Point[] | null {
  const record = asRecord(value)
  if (!record) return null
  const x1 = num(record.x1)
  const y1 = num(record.y1)
  const x2 = num(record.x2)
  const y2 = num(record.y2)
  if (x1 === null || y1 === null || x2 === null || y2 === null) return null
  return [{ x: x1, y: y1 }, { x: x2, y: y2 }]
}

function traceEntityFromRecord(id: string, value: unknown): TraceEntity | null {
  const record = asRecord(value)
  if (!record) return null
  const rawSegments = Array.isArray(record.segments) ? record.segments : []
  const segments = rawSegments.flatMap((item) => {
    const segment = segmentPolyline(item)
    return segment ? [segment] : []
  })
  if (!segments.length) return null
  const tx = num(record.terminal_x)
  const ty = num(record.terminal_y)
  return {
    id,
    segments,
    terminal: tx === null || ty === null ? undefined : { x: tx, y: ty },
    terminalType: typeof record.terminal_type === 'string' ? record.terminal_type : undefined,
  }
}

function collectTraceEntities(payload: unknown, branches = false): TraceEntity[] {
  const record = asRecord(payload)
  if (!record) return []
  const source = branches ? asRecord(record.branches) : record
  if (!source) return []
  return Object.entries(source).flatMap(([id, value]) => {
    const item = asRecord(value)
    if (branches && item?.status !== 'traced') return []
    if (!branches && item?.status && item.status !== 'ok' && item.status !== 'traced') return []
    const entity = traceEntityFromRecord(id, value)
    return entity ? [entity] : []
  })
}

function pointsAttr(points: Point[]): string {
  return points.map((point) => `${point.x},${point.y}`).join(' ')
}

export function ReviewCanvasLayers({
  workspace,
  layers,
  visibleLayers,
  imageSize,
  selectedEntity,
  onSelectEntity,
  embedded = false,
  showBoxes = true,
}: ReviewCanvasLayersProps) {
  if (!imageSize) return null

  const equipment = boxEntities(workspace?.equipment, 'equipment')
  const objects = boxEntities(workspace?.objects, 'object')
  const ports = collectPorts(layers.stage5_connection_ports)
  const traces = collectTraceEntities(layers.stage5b_trace_results)
  const branches = collectTraceEntities(layers.stage5b_branch_trace_results, true)

  return (
    <svg
      className={embedded ? 'pointer-events-none absolute inset-0 h-full w-full' : 'absolute inset-2 h-[calc(100%-1rem)] w-[calc(100%-1rem)]'}
      viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
      preserveAspectRatio="xMinYMin meet"
      aria-hidden="true"
    >
      {visibleLayers.has('traces') ? (
        <g>
          {traces.map((trace) => (
            <g key={`trace-${trace.id}`}>
              {trace.segments.map((points, index) => (
                <polyline key={`${trace.id}-${index}`} points={pointsAttr(points)} fill="none" stroke="rgb(0,200,0)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              ))}
              {trace.terminal ? (
                <g>
                  <circle cx={trace.terminal.x} cy={trace.terminal.y} r="8" fill="#b400b4" stroke="#ffffff" strokeWidth="1" />
                  <text x={trace.terminal.x + 12} y={trace.terminal.y - 12} fill="#b400b4" fontSize="18" fontWeight="700">{trace.id}:{trace.terminalType ?? ''}</text>
                </g>
              ) : null}
            </g>
          ))}
        </g>
      ) : null}

      {visibleLayers.has('branches') ? (
        <g>
          {branches.map((branch) => (
            <g key={`branch-${branch.id}`}>
              {branch.segments.map((points, index) => (
                <polyline key={`${branch.id}-${index}`} points={pointsAttr(points)} fill="none" stroke="rgb(255,0,0)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
              ))}
              {branch.terminal ? (
                <g>
                  <circle cx={branch.terminal.x} cy={branch.terminal.y} r="6" fill="rgb(255,0,0)" />
                  <text x={branch.terminal.x + 8} y={branch.terminal.y + 14} fill="rgb(255,0,0)" fontSize="14" fontWeight="700">{branch.id}:branch</text>
                </g>
              ) : null}
            </g>
          ))}
        </g>
      ) : null}

      {showBoxes && visibleLayers.has('equipment') ? (
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

      {showBoxes && visibleLayers.has('objects') ? (
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
