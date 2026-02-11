# React implementation checklist (to match this UI)

This is the “make it look and feel the same” checklist for a new React repo.

## Baseline stack

Use the same ingredients unless you have a strong reason not to:

- React + TypeScript
- Vite
- Tailwind CSS
- shadcn/ui (New York) + Radix UI
- Zustand
- Lucide React

You can still use Next.js, but you’ll spend time translating file structure and build assumptions. If the goal is visual parity fast, Vite is the closest match to this repo.

## Setup steps

1) **Add global tokens**

- Create `src/styles/index.css` and copy the token block from `design/tokens.md`.
- Import it once at app startup (typically `src/main.tsx`).

2) **Tailwind config**

- Enable `darkMode: ['class']`.
- Add the content globs for your TS/TSX files.
- Keep `tailwindcss-animate` if you use shadcn components that rely on it.

3) **shadcn/ui wiring**

Match the assumptions used here:

- `components` alias: `@/components`
- `utils` alias: `@/lib/utils`
- Tailwind CSS variables enabled
- Base color: `slate` (or keep it consistent with your token scheme)

4) **Utility helper**

Add `cn()` (clsx + tailwind-merge) and use it for every component that accepts `className`.

5) **Primitives first**

Port (or recreate) these primitives so all screens share the same look:

- `Button` (variants include `cta`)
- `Input`
- `Select`, `Slider`, `Checkbox`, `Dialog`, `DropdownMenu`, `Badge`

Then build app composites on top.

## App structure (recommended)

Keep the same folder layout so component code moves cleanly between repos:

- `src/components/ui/*`
- `src/components/*`
- `src/stores/*`
- `src/styles/index.css`
- `src/lib/utils.ts`

## Theme toggle (must-match behavior)

Requirements:

- Theme stored in `localStorage` (key can differ, behavior shouldn’t).
- Toggle sets/removes `dark` on `document.documentElement`.
- Default theme respects stored value (don’t flash the wrong theme).

## Layout constants to keep identical

- Header height: `h-14`
- Right sidebar width: `w-[320px]`
- Main content: `flex-1 overflow-hidden`
- Panel borders: `border-[var(--border-muted)]`

## Interaction parity (the stuff people notice)

1) **Keyboard shortcuts** from `design/flows-and-interactions.md` work in results mode.
2) **Undo/redo** exists and the header buttons reflect availability.
3) **Dropdown behavior**: click outside closes; Escape closes and returns focus.
4) **Reduced motion** is respected (no “growing” controls for users who opt out).

## “Definition of done” for parity

You’re close enough when:

- tokens match (light/dark), and all surfaces read the same
- header + sidebar proportions match at 1440px wide
- buttons, inputs, and menus feel consistent (focus rings, radius, borders)
- results workbench is usable without the mouse (shortcuts + visible focus)

