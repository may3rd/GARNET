# GARNET UI/UX Reference

These docs capture the UI/UX patterns used by the `frontend/` app in this repo so you can reproduce the same look and behavior in another repository.

## What this UI is

A single-page, desktop-first “workbench” for reviewing model output on an image:

- **App shell:** fixed header + full-height main content.
- **Primary layout pattern:** canvas (left) + inspector/sidebar (right, ~320px).
- **Design style:** clean, tool-like, low visual noise, with strong affordances for review actions.
- **Theme:** light + dark via CSS variables and a `dark` class on `<html>`.

## Screens / states

The app is intentionally state-driven (no deep routing required). The core states:

1. **Empty**: Upload dropzone (single or batch).
2. **Preview**: Image preview + settings sidebar.
3. **Processing**: Centered progress card with step list + cancel.
4. **Batch**: Batch list/results with the same settings sidebar.
5. **Results**: Review workbench (canvas + object sidebar).

If you mirror the UI elsewhere, keep these states as top-level “modes” so the layout stays predictable.

## Visual language (quick)

- **Surfaces:** `bg-primary` (page), `bg-secondary` (panels), `bg-canvas` (image/canvas area).
- **Borders:** thin, muted separators; avoid heavy shadows except for floating controls.
- **Corners:** rounded everywhere (mostly `lg`/`xl`).
- **Typography:** Plus Jakarta Sans; small, dense labels; uppercase micro-labels for settings.
- **Icons:** Lucide (outline), used as the primary ornamentation.

## Where to look in this repo

- **App shell + layout:** `frontend/src/App.tsx`
- **Theme + tokens:** `frontend/src/styles/index.css`
- **Header pattern:** `frontend/src/components/Header.tsx`
- **Empty state (upload):** `frontend/src/components/UploadZone.tsx`
- **Settings sidebar:** `frontend/src/components/DetectionSetup.tsx`
- **Processing state:** `frontend/src/components/ProcessingView.tsx`
- **Results workbench:** `frontend/src/components/ResultsView.tsx`
- **Component primitives (shadcn/Radix):** `frontend/src/components/ui/*`

## How to use these docs in another repo

Copy this `design/` folder as-is, then implement:

- the same token names (CSS variables) and Tailwind usage (`bg-[var(--...)]` etc.)
- the same layout primitives (header height, right sidebar width, full-height main)
- the same interaction rules (keyboard shortcuts, undo/redo, selection, export menu behavior)

Next: read `design/tokens.md`, then `design/components.md`.

If you’re implementing this in a new React repo, use `design/react-implementation.md` as a practical checklist.
