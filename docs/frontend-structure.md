# Frontend Structure

```
frontend/
├── src/
│   ├── components/           # React components
│   │   ├── ui/              # Reusable UI primitives (Radix-based)
│   │   ├── App.tsx          # Main application component
│   │   ├── UploadZone.tsx   # Image upload with drag-and-drop
│   │   ├── DetectionSetup.tsx  # Detection parameter configuration
│   │   ├── ResultsView.tsx  # Detection results with interactive canvas
│   │   ├── BatchResultsView.tsx  # Batch processing interface
│   │   ├── ProcessingView.tsx  # Processing status view
│   │   ├── ZoomControls.tsx  # Canvas zoom controls
│   │   ├── ObjectSidebar.tsx  # Object list and review controls
│   │   ├── Header.tsx       # Application header
│   │   ├── ErrorBoundary.tsx # Error boundary component
│   │   ├── CanvasView.tsx   # Main canvas implementation
│   │   ├── ReviewCanvasLayers.tsx # Canvas layers for review
│   │   ├── PipelineResultsView.tsx # Pipeline results view
│   │   ├── PipelineHitlReviewView.tsx # Pipeline HITL review
│   │   ├── PipelineReviewWorkspaceView.tsx # Pipeline workspace
│   │   ├── Stage6LineAssociationReview.tsx # Stage 6 line association review
│   │   ├── PdfPageSelector.tsx # PDF page selector
│   │   ├── PipelineArtifactCanvas.tsx # Pipeline artifact display
│   │   └── ...
│   ├── stores/
│   │   ├── appStore.ts      # Main application state (Zustand)
│   │   └── historyStore.ts  # Undo/redo action history
│   ├── lib/
│   │   ├── api.ts           # API client functions
│   │   ├── pdfExport.ts     # PDF generation utilities
│   │   ├── exportFormats.ts # CSV/JSON export utilities
│   │   ├── utils.ts         # Utility functions
│   │   ├── categoryColors.ts # Category color mappings
│   │   └── objectKey.ts     # Object key generation
│   ├── hooks/               # Custom React hooks
│   │   ├── useKeyboardShortcuts.ts # Keyboard shortcuts handler
│   │   └── useInlineEdit.ts # Inline editing hook
│   ├── types.ts             # TypeScript type definitions
│   └── styles/              # Global styles and CSS variables
│       └── index.css        # Global CSS file
├── package.json
├── vite.config.ts           # Vite configuration with API proxy
└── tsconfig.json
```