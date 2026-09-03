import type { ReactNode } from 'react'
import {
  ChevronRight,
  CircleHelp,
  Download,
  Eye,
  FileText,
  Moon,
  ScanSearch,
  Settings,
  Share2,
  Sun,
  Workflow,
} from 'lucide-react'
import {
  breadcrumbFor,
  RAIL_FOR,
  RAIL_LABEL,
  RAIL_ORDER,
  RAIL_TARGET,
  topbarChip,
  type RailKey,
} from '@/lib/nav'
import { useRunStore } from '@/stores/runStore'

const RAIL_ICON: Record<RailKey, ReactNode> = {
  sheets: <FileText size={20} strokeWidth={1.5} />,
  detection: <ScanSearch size={20} strokeWidth={1.5} />,
  extraction: <Workflow size={20} strokeWidth={1.5} />,
  review: <Eye size={20} strokeWidth={1.5} />,
  merge: <Share2 size={20} strokeWidth={1.5} />,
  exports: <Download size={20} strokeWidth={1.5} />,
}

export function AppShell({ children }: { children: ReactNode }) {
  const screen = useRunStore((s) => s.screen)
  const setScreen = useRunStore((s) => s.setScreen)
  const theme = useRunStore((s) => s.theme)
  const toggleTheme = useRunStore((s) => s.toggleTheme)
  const sheets = useRunStore((s) => s.sheets)
  const selectedSheetId = useRunStore((s) => s.selectedSheetId)
  const gateFor = useRunStore((s) => s.gateFor)

  const selected = sheets.find((s) => s.id === selectedSheetId)
  const ctx = {
    sheetLabel: selected?.label,
    gate: selected ? gateFor(selected.id) : null,
    sheetCount: sheets.length,
  }
  const crumbs = breadcrumbFor(screen, ctx)
  const activeRail = RAIL_FOR[screen]
  const chip = topbarChip(screen, ctx)

  /** Sheets with a gate waiting, for the Review badge. */
  const openGates = sheets.filter((s) => gateFor(s.id) !== null).length

  return (
    <div
      className="flex min-h-screen"
      style={{ background: 'var(--background)', color: 'var(--foreground)' }}
    >
      <aside
        className="flex w-16 shrink-0 flex-col items-center gap-1.5 py-3"
        style={{ background: 'var(--surface)' }}
      >
        <button
          type="button"
          onClick={() => setScreen('sheets')}
          title="GARNET — start over"
          className="flex items-center justify-center text-[15px] font-semibold tracking-tight"
          style={{
            width: 40,
            height: 40,
            border: 0,
            cursor: 'pointer',
            borderRadius: 12,
            background: 'linear-gradient(145deg, oklch(0.62 0.195 253.83), oklch(0.72 0.15 220))',
            color: 'var(--snow)',
          }}
        >
          G
        </button>
        <div className="h-2.5" />

        {RAIL_ORDER.map((key) => {
          const isActive = key === activeRail
          const badge = key === 'review' && openGates > 0 ? openGates : null
          return (
            <button
              key={key}
              type="button"
              title={RAIL_LABEL[key]}
              aria-label={RAIL_LABEL[key]}
              aria-current={isActive ? 'page' : undefined}
              onClick={() => setScreen(RAIL_TARGET[key])}
              className="relative flex items-center justify-center"
              style={{
                width: 40,
                height: 40,
                border: 0,
                cursor: 'pointer',
                borderRadius: 'var(--r-field)',
                background: isActive ? 'var(--accent-soft)' : 'transparent',
                color: isActive ? 'var(--accent-soft-fg)' : 'var(--muted)',
              }}
            >
              {RAIL_ICON[key]}
              {badge !== null && (
                <span
                  className="mono absolute flex items-center justify-center"
                  style={{
                    top: 4,
                    right: 3,
                    minWidth: 15,
                    height: 15,
                    padding: '0 3px',
                    borderRadius: 999,
                    background: 'var(--warning)',
                    color: 'var(--eclipse)',
                    fontSize: 9,
                    fontWeight: 600,
                  }}
                >
                  {badge}
                </span>
              )}
            </button>
          )
        })}

        <div className="flex-1" />

        <button
          type="button"
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
          aria-label="Toggle theme"
          className="flex items-center justify-center"
          style={{
            width: 40,
            height: 40,
            border: 0,
            background: 'transparent',
            borderRadius: 'var(--r-field)',
            color: 'var(--muted)',
            cursor: 'pointer',
          }}
        >
          {theme === 'dark' ? <Sun size={20} strokeWidth={1.5} /> : <Moon size={20} strokeWidth={1.5} />}
        </button>
        <button
          type="button"
          title="Settings — not built yet"
          aria-label="Settings"
          aria-disabled
          className="flex items-center justify-center"
          style={{
            width: 40,
            height: 40,
            border: 0,
            background: 'transparent',
            borderRadius: 'var(--r-field)',
            color: 'var(--muted)',
            opacity: 0.55,
            cursor: 'default',
          }}
        >
          <Settings size={20} strokeWidth={1.5} />
        </button>
        <span
          className="inline-flex shrink-0 items-center justify-center text-xs font-semibold"
          style={{
            width: 32,
            height: 32,
            borderRadius: 999,
            background: 'var(--surface-tertiary)',
            color: 'var(--foreground)',
          }}
        >
          ML
        </span>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header
          className="flex h-14 shrink-0 items-center gap-2 px-5"
          style={{ borderBottom: '1px solid var(--separator)' }}
        >
          <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-2">
            {crumbs.map((crumb, i) => {
              const isLast = i === crumbs.length - 1
              return (
                <span key={`${crumb.label}-${i}`} className="flex items-center gap-2">
                  {i > 0 && (
                    <ChevronRight size={14} strokeWidth={1.5} style={{ color: 'var(--muted)' }} />
                  )}
                  {crumb.to && !isLast ? (
                    <button
                      type="button"
                      onClick={() => setScreen(crumb.to!)}
                      style={{
                        border: 0,
                        background: 'transparent',
                        padding: 0,
                        fontSize: 14,
                        fontFamily: 'inherit',
                        color: 'var(--muted)',
                        cursor: 'pointer',
                        textDecoration: 'none',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = 'var(--foreground)'
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.color = 'var(--muted)'
                      }}
                    >
                      {crumb.label}
                    </button>
                  ) : (
                    <span
                      aria-current={isLast ? 'page' : undefined}
                      style={{
                        fontSize: 14,
                        color: isLast ? 'var(--foreground)' : 'var(--muted)',
                        fontWeight: isLast ? 500 : 400,
                      }}
                    >
                      {crumb.label}
                    </span>
                  )}
                </span>
              )
            })}
          </nav>

          <div className="flex-1" />

          <div className="flex items-center gap-2.5">
            {chip && (
              <span
                className="inline-flex w-fit shrink-0 items-center"
                style={{
                  padding: '2px 8px',
                  borderRadius: 'var(--r-chip)',
                  fontSize: 12,
                  lineHeight: '20px',
                  fontWeight: 500,
                  background: 'transparent',
                  color: 'var(--muted)',
                  boxShadow: 'inset 0 0 0 1px var(--border)',
                }}
              >
                {chip}
              </span>
            )}
            <button
              type="button"
              aria-label="Help"
              aria-disabled
              title="Help — not built yet"
              className="flex items-center justify-center"
              style={{
                width: 32,
                height: 32,
                border: 0,
                background: 'transparent',
                borderRadius: 'var(--r-btn)',
                color: 'var(--muted)',
                opacity: 0.55,
                cursor: 'default',
              }}
            >
              <CircleHelp size={17} strokeWidth={1.5} />
            </button>
            <span
              className="inline-flex shrink-0 items-center justify-center"
              style={{
                width: 30,
                height: 30,
                borderRadius: 999,
                background: 'var(--accent)',
                color: 'var(--accent-foreground)',
                fontSize: 12,
                fontWeight: 600,
              }}
            >
              ML
            </span>
          </div>
        </header>

        <main className="min-h-0 flex-1">{children}</main>
      </div>
    </div>
  )
}
