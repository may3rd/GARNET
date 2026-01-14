# GARNET Frontend Implementation Plan

**Project**: React + TypeScript web application for P&ID object detection
**Reference**: [product_overview.md](./product_overview.md)

---

## Phase 1: Project Setup & Core Infrastructure

### 1.1 Initialize Project

```bash
# Create Vite + React + TypeScript project
cd /Volumes/Ginnungagap/maetee/Code/GARNET
bunx create-vite@latest frontend --template react-ts

# Navigate to frontend
cd frontend

# Install core dependencies
bun install
```

### 1.2 Install Dependencies

```bash
# UI Components (Shadcn/ui)
bunx shadcn@latest init

# When prompted:
# - Style: Default
# - Base color: Slate
# - CSS variables: Yes
# - tailwind.config.js location: tailwind.config.js
# - Components location: @/components
# - Utils location: @/lib/utils

# Add Shadcn components
bunx shadcn@latest add button dialog dropdown-menu input slider tabs tooltip scroll-area separator sheet

# State Management
bun add zustand

# Canvas Library
bun add fabric

# HTTP Client
bun add axios

# Icons
bun add lucide-react

# Data Grid (virtualized)
bun add @tanstack/react-table @tanstack/react-virtual

# Type definitions
bun add -d @types/fabric
```

### 1.3 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/                 # Shadcn components (auto-generated)
│   │   ├── canvas/
│   │   │   ├── Canvas.tsx           # Main canvas component
│   │   │   ├── BoundingBox.tsx      # Individual box component
│   │   │   ├── ZoomControls.tsx     # Zoom toolbar
│   │   │   └── Minimap.tsx          # Navigation minimap
│   │   ├── sidebar/
│   │   │   ├── Sidebar.tsx          # Main sidebar container
│   │   │   ├── ObjectTree.tsx       # Category tree view
│   │   │   ├── ObjectDetail.tsx     # Selected object details
│   │   │   └── FilterPanel.tsx      # Confidence filter
│   │   ├── upload/
│   │   │   ├── DropZone.tsx         # Drag-drop upload area
│   │   │   └── FilePreview.tsx      # Image preview before detect
│   │   ├── detection/
│   │   │   ├── DetectionConfig.tsx  # Model/threshold settings
│   │   │   └── ProgressPanel.tsx    # Detection progress
│   │   └── layout/
│   │       ├── Header.tsx           # Top navigation bar
│   │       └── Layout.tsx           # Main layout wrapper
│   ├── hooks/
│   │   ├── useCanvas.ts             # Canvas interactions
│   │   ├── useDetection.ts          # Detection API calls
│   │   ├── useKeyboardShortcuts.ts  # Keyboard navigation
│   │   └── useTheme.ts              # Dark/light mode
│   ├── stores/
│   │   ├── detectionStore.ts        # Detection results state
│   │   ├── canvasStore.ts           # Canvas/viewport state
│   │   └── uiStore.ts               # UI state (sidebar, dialogs)
│   ├── lib/
│   │   ├── api.ts                   # API client (axios)
│   │   ├── utils.ts                 # Utility functions
│   │   └── constants.ts             # App constants, colors
│   ├── types/
│   │   ├── detection.ts             # Detection result types
│   │   └── api.ts                   # API request/response types
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css                    # Tailwind imports + globals
├── public/
│   └── sample-pid.png               # Demo P&ID image
├── tailwind.config.js
├── tsconfig.json
├── vite.config.ts
└── package.json
```

### 1.4 Configure Tailwind

```javascript
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Category colors for bounding boxes
        category: {
          1: '#dc2626', // Gate Valve
          2: '#2563eb', // Check Valve
          3: '#16a34a', // Control Valve
          4: '#9333ea', // Pump
          5: '#ea580c', // Instrument
          6: '#0891b2', // Line Number
        }
      }
    },
  },
  plugins: [require("tailwindcss-animate")],
}
```

### 1.5 Configure Path Aliases

```json
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

```typescript
// vite.config.ts
import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
```

---

## Phase 2: Core Components

### 2.1 State Management Setup

```typescript
// src/stores/detectionStore.ts
import { create } from 'zustand'

interface DetectedObject {
  id: number
  categoryId: number
  categoryName: string
  bbox: { x: number; y: number; width: number; height: number }
  confidence: number
  ocrText?: string
  status: 'pending' | 'accepted' | 'rejected'
}

interface DetectionState {
  objects: DetectedObject[]
  selectedId: number | null
  imageUrl: string | null
  isLoading: boolean
  progress: { step: string; percent: number } | null
  
  // Actions
  setObjects: (objects: DetectedObject[]) => void
  selectObject: (id: number | null) => void
  updateObjectStatus: (id: number, status: 'accepted' | 'rejected') => void
  setImageUrl: (url: string) => void
  setProgress: (progress: { step: string; percent: number } | null) => void
}

export const useDetectionStore = create<DetectionState>((set) => ({
  objects: [],
  selectedId: null,
  imageUrl: null,
  isLoading: false,
  progress: null,
  
  setObjects: (objects) => set({ objects }),
  selectObject: (id) => set({ selectedId: id }),
  updateObjectStatus: (id, status) => set((state) => ({
    objects: state.objects.map(obj => 
      obj.id === id ? { ...obj, status } : obj
    )
  })),
  setImageUrl: (url) => set({ imageUrl: url }),
  setProgress: (progress) => set({ progress }),
}))
```

### 2.2 API Client

```typescript
// src/lib/api.ts
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
})

export interface DetectionRequest {
  file: File
  model: string
  confidence: number
  enableOcr: boolean
}

export interface DetectionResponse {
  objects: Array<{
    id: number
    category_id: number
    category_name: string
    bbox: [number, number, number, number] // x, y, w, h
    confidence: number
    ocr_text?: string
  }>
  image_url: string
}

export const detectObjects = async (request: DetectionRequest): Promise<DetectionResponse> => {
  const formData = new FormData()
  formData.append('file_input', request.file)
  formData.append('selected_model', request.model)
  formData.append('conf_th', request.confidence.toString())
  formData.append('text_OCR', request.enableOcr.toString())
  
  const response = await api.post('/detect', formData)
  return response.data
}

export const getModels = async (): Promise<string[]> => {
  const response = await api.get('/models')
  return response.data
}
```

### 2.3 Main App Layout

```tsx
// src/App.tsx
import { useState } from 'react'
import { Layout } from '@/components/layout/Layout'
import { DropZone } from '@/components/upload/DropZone'
import { Canvas } from '@/components/canvas/Canvas'
import { Sidebar } from '@/components/sidebar/Sidebar'
import { ProgressPanel } from '@/components/detection/ProgressPanel'
import { useDetectionStore } from '@/stores/detectionStore'

function App() {
  const { imageUrl, isLoading } = useDetectionStore()
  
  return (
    <Layout>
      {!imageUrl && !isLoading && (
        <DropZone />
      )}
      
      {isLoading && (
        <ProgressPanel />
      )}
      
      {imageUrl && !isLoading && (
        <div className="flex h-full">
          <div className="flex-1">
            <Canvas />
          </div>
          <Sidebar />
        </div>
      )}
    </Layout>
  )
}

export default App
```

---

## Phase 3: Canvas Implementation

### 3.1 Canvas Component with Fabric.js

```tsx
// src/components/canvas/Canvas.tsx
import { useEffect, useRef } from 'react'
import { fabric } from 'fabric'
import { useDetectionStore } from '@/stores/detectionStore'
import { useCanvasStore } from '@/stores/canvasStore'
import { ZoomControls } from './ZoomControls'
import { CATEGORY_COLORS } from '@/lib/constants'

export function Canvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fabricRef = useRef<fabric.Canvas | null>(null)
  
  const { objects, imageUrl, selectedId, selectObject } = useDetectionStore()
  const { zoom, setZoom } = useCanvasStore()
  
  // Initialize canvas
  useEffect(() => {
    if (!canvasRef.current) return
    
    fabricRef.current = new fabric.Canvas(canvasRef.current, {
      backgroundColor: '#1e293b',
      selection: false,
    })
    
    return () => {
      fabricRef.current?.dispose()
    }
  }, [])
  
  // Load image
  useEffect(() => {
    if (!fabricRef.current || !imageUrl) return
    
    fabric.Image.fromURL(imageUrl, (img) => {
      fabricRef.current!.clear()
      img.selectable = false
      fabricRef.current!.add(img)
      fabricRef.current!.renderAll()
    })
  }, [imageUrl])
  
  // Render bounding boxes
  useEffect(() => {
    if (!fabricRef.current) return
    
    // Remove existing boxes
    const existingBoxes = fabricRef.current.getObjects('rect')
    existingBoxes.forEach(box => fabricRef.current!.remove(box))
    
    // Add new boxes
    objects.forEach((obj) => {
      const rect = new fabric.Rect({
        left: obj.bbox.x,
        top: obj.bbox.y,
        width: obj.bbox.width,
        height: obj.bbox.height,
        fill: 'transparent',
        stroke: CATEGORY_COLORS[obj.categoryId] || '#ffffff',
        strokeWidth: selectedId === obj.id ? 4 : 2,
        opacity: obj.status === 'rejected' ? 0.3 : 1,
        selectable: false,
        data: { objectId: obj.id },
      })
      
      fabricRef.current!.add(rect)
    })
    
    fabricRef.current.renderAll()
  }, [objects, selectedId])
  
  // Handle object click
  useEffect(() => {
    if (!fabricRef.current) return
    
    fabricRef.current.on('mouse:down', (e) => {
      if (e.target && e.target.data?.objectId) {
        selectObject(e.target.data.objectId)
      } else {
        selectObject(null)
      }
    })
  }, [selectObject])
  
  return (
    <div className="relative h-full w-full">
      <canvas ref={canvasRef} />
      <ZoomControls />
    </div>
  )
}
```

---

## Phase 4: Sidebar & Object Tree

### 4.1 Sidebar Component

```tsx
// src/components/sidebar/Sidebar.tsx
import { useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ObjectTree } from './ObjectTree'
import { ObjectDetail } from './ObjectDetail'
import { FilterPanel } from './FilterPanel'

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  
  if (collapsed) {
    return (
      <Button
        variant="ghost"
        size="icon"
        className="fixed right-4 top-20"
        onClick={() => setCollapsed(false)}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>
    )
  }
  
  return (
    <div className="w-80 border-l bg-background flex flex-col">
      <div className="flex items-center justify-between p-4 border-b">
        <h2 className="font-semibold">Objects</h2>
        <Button variant="ghost" size="icon" onClick={() => setCollapsed(true)}>
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
      
      <FilterPanel />
      
      <div className="flex-1 overflow-auto">
        <ObjectTree />
      </div>
      
      <ObjectDetail />
    </div>
  )
}
```

---

## Phase 5: Keyboard Shortcuts

### 5.1 Keyboard Hook

```typescript
// src/hooks/useKeyboardShortcuts.ts
import { useEffect } from 'react'
import { useDetectionStore } from '@/stores/detectionStore'

export function useKeyboardShortcuts() {
  const { objects, selectedId, selectObject, updateObjectStatus } = useDetectionStore()
  
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (e.target instanceof HTMLInputElement) return
      
      switch (e.key) {
        case 'Tab':
          e.preventDefault()
          const currentIndex = objects.findIndex(o => o.id === selectedId)
          const nextIndex = e.shiftKey
            ? (currentIndex - 1 + objects.length) % objects.length
            : (currentIndex + 1) % objects.length
          selectObject(objects[nextIndex]?.id ?? null)
          break
          
        case 'Enter':
          if (selectedId) updateObjectStatus(selectedId, 'accepted')
          break
          
        case 'Backspace':
        case 'Delete':
          if (selectedId) updateObjectStatus(selectedId, 'rejected')
          break
          
        case 'Escape':
          selectObject(null)
          break
          
        case 'f':
          // Fit to window
          break
          
        case '0':
          // Zoom 100%
          break
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [objects, selectedId, selectObject, updateObjectStatus])
}
```

---

## Phase 6: Backend API Updates

### 6.1 New FastAPI Endpoints

Add to `main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Add CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/models")
async def get_models():
    """List available detection models."""
    return [{"name": f["item"]} for f in MODEL_LIST]

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Keep existing /submit endpoint but also add /api/detect
@app.post("/api/detect")
async def detect_api(
    file_input: UploadFile = File(...),
    selected_model: str = Form("ultralytics"),
    conf_th: float = Form(0.8),
    text_OCR: bool = Form(False),
):
    """API endpoint returning JSON instead of HTML."""
    # ... detection logic ...
    return JSONResponse({
        "objects": sorted_data,
        "image_url": "/static/images/prediction_results.png",
    })
```

---

## Implementation Checklist

### Week 1: Foundation
- [ ] Initialize Vite + React + TypeScript project
- [ ] Install and configure Shadcn/ui
- [ ] Set up Zustand stores
- [ ] Create basic layout components
- [ ] Implement DropZone component

### Week 2: Canvas & Detection
- [ ] Implement Fabric.js canvas
- [ ] Render bounding boxes
- [ ] Add zoom/pan controls
- [ ] Connect to FastAPI backend
- [ ] Show detection progress

### Week 3: Sidebar & Interactions
- [ ] Build ObjectTree component
- [ ] Implement object selection
- [ ] Add Accept/Reject buttons
- [ ] Create detail panel
- [ ] Add confidence filter

### Week 4: Polish
- [ ] Implement keyboard shortcuts
- [ ] Add dark mode
- [ ] Create export functionality
- [ ] Add minimap
- [ ] Performance optimization

---

## Commands Reference

```bash
# Start frontend dev server
cd frontend && bun run dev

# Start backend
cd .. && uvicorn main:app --reload

# Build for production
cd frontend && bun run build

# Add new Shadcn component
bunx shadcn@latest add [component-name]

# Type check
bun run tsc --noEmit
```
