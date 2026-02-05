# UI implementation notes (for reproducing this stack)

This is not a full setup guide; it’s the minimal “what are we using” list so another repo can match behavior and styling.

## Stack choices

- **React + TypeScript**
- **Vite** for bundling/dev server
- **Tailwind CSS** for layout/spacing/typography utilities
- **shadcn/ui** (New York style) + **Radix UI** for accessible primitives
- **Zustand** for app state (view mode, options, results, batch state, theme)
- **Lucide React** for icons

## Theming approach

- Theme is controlled by a `dark` class on `document.documentElement`.
- Colors are CSS variables defined in a single global stylesheet.
- Components reference tokens via Tailwind arbitrary values (example: `border-[var(--border-muted)]`).
- Theme choice persists via `localStorage`.

## Utility conventions

- Use a `cn()` helper that merges Tailwind classes (`clsx` + `tailwind-merge`).
- Keep “panel” styling consistent: `bg-secondary` + `border-muted` + rounded corners.

## Recommended folder layout (port-friendly)

If you want easy copy/paste from this repo, keep similar paths:

- `src/components/ui/*` (primitives)
- `src/components/*` (app composites)
- `src/styles/index.css` (tokens + globals)
- `src/stores/*` (Zustand)

