# Component spec (primitives + app-level patterns)

This UI is built from a small set of primitives (shadcn/Radix) plus a few app-level composites. The goal is consistency, not novelty.

## Primitive components (baseline behavior)

These are shadcn-style components with styling wired to the CSS variables from `design/tokens.md`.

- **Button**
  - Variants: `default`, `cta`, `outline`, `secondary`, `ghost`, `link`, `destructive`
  - Sizes: `default`, `sm`, `lg`, `icon`
  - Focus: ring uses `--accent`
- **Input**
  - Border uses `--border-muted`, background uses `--bg-primary`
  - Placeholder uses `--text-secondary`
- **Select / Slider / Checkbox / Dialog / DropdownMenu / Badge**
  - Stick to the same border, radius, and muted surface conventions.

If you port the UI to a new repo, keep these primitives first; everything else composes from them.

## App shell components

### Header

Goals:

- Always present (height `h-14`), subtle border at the bottom.
- Left: back affordance (when not in empty state) + app mark + product name.
- Middle (desktop only): lightweight context stats in results mode.
- Right: undo/redo (results only), batch nav (results + batch context), theme toggle.

Interaction requirements:

- Undo/redo buttons must be disabled when not available.
- Theme toggle flips a `dark` class at the document root and persists choice.
- Buttons are icon-first and use `ghost` styling to stay quiet.

### Main layout

The main layout is “full height, no page scroll”:

- `main` is `flex-1 overflow-hidden`.
- Panels that scroll do so internally (`overflow-y-auto`), never the whole page.

## Workspace patterns

### Canvas area

The canvas region is the primary workspace:

- Background: `bg-canvas`.
- Image is centered with a max size (typically `max-w-[85%] max-h-[85%]` in preview).
- In results mode, canvas supports:
  - zoom controls (floating)
  - selection highlighting
  - minimap (if present)
  - click-to-select, keyboard navigation, and edit/create modes

Floating controls use:

- `bg-primary/95` with `backdrop-blur`
- a stronger shadow (so they read as “tooling”)
- smooth hover growth, but must respect reduced-motion.

### Right sidebar (“inspector”)

Default width is **320px**. It is used both for settings (pre-run) and object review (post-run).

Rules:

- Background: `bg-secondary`
- Border: left border using `border-muted`
- Scroll: internal, not page-level
- Section headers are small, high-contrast, and don’t waste vertical space

### Cards and callouts

Cards are used for:

- processing progress
- batch mode notice
- inline errors

All cards follow:

- `bg-secondary` or `bg-primary` (depending on nesting)
- `border-muted`
- `rounded-2xl` (or `xl` where smaller)

## Iconography

- Use Lucide outline icons.
- Default size `h-4 w-4` (or `h-5 w-5` in header).
- Icon color usually follows semantic text color; reserve `--accent` for emphasis.

