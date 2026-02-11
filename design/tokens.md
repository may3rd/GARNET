# Design tokens (colors, radius, typography)

This UI uses **CSS variables as the source of truth**, then references them from Tailwind using arbitrary values like `bg-[var(--bg-primary)]`.

## Color tokens

Use semantic names. Keep the same names across repos so components can be copied without rewiring.

```css
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-canvas: #e2e8f0;

  --text-primary: #1e293b;
  --text-secondary: #64748b;

  --accent: #2563eb;
  --accent-strong: #1d4ed8;
  --accent-cta: #f97316;

  --border-muted: #e2e8f0;

  --danger: #dc2626;
  --success: #16a34a;

  /* Category colors used in results UI */
  --color-gate-valve: #dc2626;
  --color-check-valve: #2563eb;
  --color-control-valve: #16a34a;
  --color-pump: #9333ea;
  --color-instrument: #ea580c;
  --color-line-number: #0891b2;
}

.dark {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-canvas: #334155;

  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;

  --accent: #60a5fa;
  --accent-strong: #3b82f6;
  --accent-cta: #fb923c;

  --border-muted: #334155;

  --danger: #f87171;
  --success: #4ade80;

  --color-gate-valve: #f87171;
  --color-check-valve: #60a5fa;
  --color-control-valve: #4ade80;
  --color-pump: #c084fc;
  --color-instrument: #fb923c;
  --color-line-number: #22d3ee;
}
```

### Token semantics

- `bg-primary`: app/page background.
- `bg-secondary`: panel background (header, sidebars, cards).
- `bg-canvas`: canvas/backdrop behind images.
- `accent`: primary action + focus affordances.
- `accent-cta`: “Run” / “primary conversion” button.
- `border-muted`: subtle separators; used everywhere (don’t introduce new border colors).
- `danger` / `success`: status and review outcomes.

## Radius tokens

Keep radius consistent; the UI relies on rounded shapes as its “softness”.

```css
:root {
  --radius-sm: 0.375rem;
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --radius-xl: 1rem;
  --radius-2xl: 1.5rem;
}
```

## Typography

- **Font:** Plus Jakarta Sans for UI text; system fallback.
- **Default body:** regular weight, compact line length.
- **Micro-labels:** `text-xs`, often `uppercase tracking-wide` for settings labels.
- **Numbers/monospace:** system monospace for occasional code-like values.

In this repo the font is loaded in CSS:

- `frontend/src/styles/index.css` imports the Google font.
- Tailwind adds the same family via `fontFamily.sans` in `frontend/tailwind.config.js`.

## Focus + motion

- Focus states use `--accent` and `:focus-visible` (keyboard-only).
- Reduced-motion support is handled via a global `prefers-reduced-motion` rule; keep it.

