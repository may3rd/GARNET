# Flows, interactions, and states

This UI lives or dies on predictability. Keep the flows simple and make the “next action” obvious.

## Primary flows

### 1) Upload → preview → run

- Empty state is a **single large dropzone** with click-to-browse support.
- Drag-over state changes border + adds a light accent wash.
- If multiple files are selected, the app enters batch mode (the user shouldn’t have to choose batch explicitly).
- Preview state shows the image on the left and settings on the right.

### 2) Processing

- Processing state replaces the workspace with a centered progress card.
- The card shows:
  - current step label
  - a percent bar
  - a short checklist of steps with done/active/pending markers
- A single “Cancel detection” button is always visible.

### 3) Results review

Results is the “workbench”:

- Left: interactive canvas (pan/zoom/select/edit).
- Right: object sidebar (filter/search/group/review/export).
- Header shows object totals and review progress on desktop.

## Keyboard shortcuts (results)

These are the current shortcuts to preserve:

- `Tab` / `Shift+Tab`: cycle selection
- `Enter`: accept selected object
- `Backspace` or `Delete`: reject selected object
- `F`: fit to screen
- `0`: reset zoom
- `-`: zoom out
- `+` / `=`: zoom in
- `Ctrl/Cmd+Z`: undo
- `Ctrl/Cmd+Shift+Z` or `Ctrl/Cmd+Y`: redo

Important: shortcuts must not fire when focus is inside an input/textarea/select.

## Review states

Objects have three review states:

- **pending**: default
- **accepted**: success styling
- **rejected**: danger styling

Review actions should feel immediate:

- update UI optimistically
- persist in the background
- don’t block the user on network latency

Undo/redo must cover review state changes and other user edits that matter.

## Menus and overlays

Export is a dropdown attached to the sidebar toolbar:

- click outside closes
- `Escape` closes and returns focus to the export button
- menu is scrollable if tall (`max-h` + `overflow-y-auto`)

Dialogs (if any) should follow the same rule: escape closes, focus returns sensibly, content stays compact.

## Responsiveness

Desktop-first, but don’t break on smaller widths:

- Hide non-essential header text on small screens.
- Keep the sidebar usable; if you introduce a collapse affordance, preserve the 320px default when open.

## Accessibility baseline

Minimum bar to match this UI:

- All icon buttons have `aria-label`.
- Focus-visible rings are clear and consistent (`--accent`).
- Click targets are at least ~36px (icon buttons meet this).
- Reduced-motion is respected (turn off hover growth transitions).

