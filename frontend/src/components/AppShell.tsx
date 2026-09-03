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
import { useRunStore } from '@/stores/runStore'

type RailItem = { key: string; label: string; icon: ReactNode }

const RAIL: RailItem[] = [
  { key: 'sheets', label: 'Sheets', icon: <FileText size={20} strokeWidth={1.5} /> },
  { key: 'detection', label: 'Detection', icon: <ScanSearch size={20} strokeWidth={1.5} /> },
  { key: 'extraction', label: 'Extraction', icon: <Workflow size={20} strokeWidth={1.5} /> },
  { key: 'review', label: 'Review', icon: <Eye size={20} strokeWidth={1.5} /> },
  { key: 'merge', label: 'Merge', icon: <Share2 size={20} strokeWidth={1.5} /> },
  { key: 'exports', label: 'Exports', icon: <Download size={20} strokeWidth={1.5} /> },
]

export function AppShell({
  active,
  breadcrumb,
  project = 'GARNET',
  children,
}: {
  active: string
  breadcrumb: string[]
  project?: string
  children: ReactNode
}) {
  const theme = useRunStore((s) => s.theme)
  const toggleTheme = useRunStore((s) => s.toggleTheme)

  return (
    <div className="flex min-h-screen" style={{ background: 'var(--background)', color: 'var(--foreground)' }}>
      {/* Icon rail */}
      <aside
        className="flex w-16 shrink-0 flex-col items-center gap-1.5 py-3"
        style={{ background: 'var(--surface)' }}
      >
        <div
          className="flex h-10 w-10 items-center justify-center text-[15px] font-semibold tracking-tight"
          style={{
            borderRadius: 12,
            background: 'linear-gradient(145deg, oklch(0.62 0.195 253.83), oklch(0.72 0.15 220))',
            color: 'var(--snow)',
          }}
        >
          G
        </div>
        <div className="h-2.5" />

        {RAIL.map((item) => {
          const isActive = item.key === active
          return (
            <button
              key={item.key}
              type="button"
              title={item.label}
              aria-label={item.label}
              aria-current={isActive ? 'page' : undefined}
              className="flex h-10 w-10 items-center justify-center"
              style={{
                borderRadius: 'var(--r-field)',
                background: isActive ? 'var(--accent-soft)' : 'transparent',
                color: isActive ? 'var(--accent-soft-fg)' : 'var(--muted)',
              }}
            >
              {item.icon}
            </button>
          )
        })}

        <div className="flex-1" />

        <button
          type="button"
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
          aria-label="Toggle theme"
          className="flex h-10 w-10 items-center justify-center"
          style={{ borderRadius: 'var(--r-field)', color: 'var(--muted)' }}
        >
          {theme === 'dark' ? <Sun size={20} strokeWidth={1.5} /> : <Moon size={20} strokeWidth={1.5} />}
        </button>
        <button
          type="button"
          title="Settings"
          aria-label="Settings"
          className="flex h-10 w-10 items-center justify-center"
          style={{ borderRadius: 'var(--r-field)', color: 'var(--muted)' }}
        >
          <Settings size={20} strokeWidth={1.5} />
        </button>
        <span
          className="inline-flex h-8 w-8 shrink-0 items-center justify-center text-xs font-semibold"
          style={{ borderRadius: 999, background: 'var(--surface-tertiary)', color: 'var(--foreground)' }}
        >
          ML
        </span>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header
          className="flex h-14 shrink-0 items-center gap-2 px-5"
          style={{ borderBottom: '1px solid var(--separator)' }}
        >
          {breadcrumb.map((crumb, i) => (
            <span key={crumb} className="flex items-center gap-2">
              {i > 0 && <ChevronRight size={14} strokeWidth={1.5} style={{ color: 'var(--muted)' }} />}
              <span
                style={{
                  fontSize: 14,
                  color: i === breadcrumb.length - 1 ? 'var(--foreground)' : 'var(--muted)',
                  fontWeight: i === breadcrumb.length - 1 ? 500 : 400,
                }}
              >
                {crumb}
              </span>
            </span>
          ))}

          <div className="flex-1" />

          <div className="flex items-center gap-2.5">
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
              {project}
            </span>
            <button
              type="button"
              aria-label="Help"
              className="flex items-center justify-center"
              style={{
                width: 32,
                height: 32,
                border: 0,
                background: 'transparent',
                borderRadius: 'var(--r-btn)',
                color: 'var(--muted)',
                cursor: 'pointer',
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
