import type { CSSProperties, ReactNode } from 'react'

/**
 * Primitives transcribed from the design canvas. Values are the artboards'
 * literal values, not approximations — chips are 2px/8px at 12px/20px, cards
 * are --r-card on --surface, page titles are 22px/600.
 */

export type TagTone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger' | 'outline'

const TAG_TONES: Record<TagTone, CSSProperties> = {
  neutral: { background: 'var(--default)', color: 'var(--default-foreground)' },
  accent: { background: 'var(--accent-soft)', color: 'var(--accent-soft-fg)' },
  success: { background: 'var(--success-soft)', color: 'var(--success-soft-fg)' },
  warning: { background: 'var(--warning-soft)', color: 'var(--warning-soft-fg)' },
  danger: { background: 'var(--danger-soft)', color: 'var(--danger-soft-fg)' },
  outline: {
    background: 'transparent',
    color: 'var(--muted)',
    boxShadow: 'inset 0 0 0 1px var(--border)',
  },
}

export function Tag({
  children,
  tone = 'outline',
  dot,
  style,
}: {
  children: ReactNode
  tone?: TagTone
  /** Leading 6px status dot, as used by the Status column and timing chips. */
  dot?: string
  style?: CSSProperties
}) {
  return (
    <span
      className="inline-flex w-fit shrink-0 items-center gap-[5px]"
      style={{
        padding: '2px 8px',
        borderRadius: 'var(--r-chip)',
        fontSize: 12,
        lineHeight: '20px',
        fontWeight: 500,
        ...TAG_TONES[tone],
        ...style,
      }}
    >
      {dot && (
        <span
          className="shrink-0"
          style={{ width: 6, height: 6, borderRadius: 999, background: dot }}
        />
      )}
      <span>{children}</span>
    </span>
  )
}

export function Card({
  children,
  padding = 16,
  className,
  style,
}: {
  children: ReactNode
  padding?: number
  className?: string
  style?: CSSProperties
}) {
  return (
    <div
      className={className}
      style={{
        background: 'var(--surface)',
        borderRadius: 'var(--r-card)',
        padding,
        boxSizing: 'border-box',
        ...style,
      }}
    >
      {children}
    </div>
  )
}

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string
  subtitle?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="flex items-center gap-4">
      <div className="min-w-0 flex-1">
        <div style={{ fontSize: 22, fontWeight: 600 }}>{title}</div>
        {subtitle && (
          <div style={{ fontSize: 14, color: 'var(--muted)' }}>{subtitle}</div>
        )}
      </div>
      {actions && <div className="flex items-stretch gap-2.5">{actions}</div>}
    </div>
  )
}

/** Card sub-header: 14px/500 title over a 14px muted line, with optional actions. */
export function SectionHeader({
  title,
  description,
  actions,
}: {
  title: string
  description?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="min-w-0 flex-1">
        <div style={{ fontSize: 14, fontWeight: 500, lineHeight: '24px' }}>{title}</div>
        {description && (
          <div style={{ fontSize: 14, lineHeight: '20px', color: 'var(--muted)' }}>
            {description}
          </div>
        )}
      </div>
      {actions && <div className="flex items-stretch gap-2">{actions}</div>}
    </div>
  )
}

export function Separator() {
  return <div style={{ height: 1, background: 'var(--separator)' }} />
}

export function IconTile({
  children,
  size = 40,
  radius = 13,
}: {
  children: ReactNode
  size?: number
  radius?: number
}) {
  return (
    <div
      className="flex shrink-0 items-center justify-center"
      style={{
        width: size,
        height: size,
        borderRadius: radius,
        background: 'var(--accent-soft)',
        color: 'var(--accent-soft-fg)',
      }}
    >
      {children}
    </div>
  )
}

/** The design's 38×22 pill switch. */
export function Toggle({
  checked,
  onChange,
  label,
  description,
}: {
  checked: boolean
  onChange: (next: boolean) => void
  label: string
  description?: string
}) {
  return (
    <div className="flex flex-1 items-center gap-2.5">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className="relative shrink-0"
        style={{
          width: 38,
          height: 22,
          border: 0,
          padding: 0,
          cursor: 'pointer',
          borderRadius: 999,
          background: checked
            ? 'var(--accent)'
            : 'color-mix(in oklab, var(--foreground) 22%, transparent)',
        }}
      >
        <span
          style={{
            position: 'absolute',
            top: 2,
            left: checked ? 18 : 2,
            width: 18,
            height: 18,
            borderRadius: 999,
            background: 'var(--white)',
            boxShadow: '0 1px 2px rgba(0,0,0,.28)',
            transition: 'left .15s',
          }}
        />
      </button>
      <div>
        <div style={{ fontSize: 13, fontWeight: 500 }}>{label}</div>
        {description && (
          <div style={{ fontSize: 12, color: 'var(--muted)' }}>{description}</div>
        )}
      </div>
    </div>
  )
}

/**
 * The design's 36px field: a label over a --field-background box with a
 * trailing chevron. Rendered as a native select so it stays keyboard- and
 * touch-accessible without a popover.
 */
export function SelectField<T extends string>({
  label,
  value,
  options,
  hint,
  disabled,
  onChange,
  flex,
}: {
  label: string
  value: T
  options: { key: T; label: string }[]
  hint?: string
  disabled?: boolean
  onChange: (value: T) => void
  flex?: number
}) {
  return (
    <div className="flex flex-col gap-1.5" style={{ flex: flex ?? 1, minWidth: 0 }}>
      <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--muted)' }}>{label}</div>
      <div
        className="flex items-center gap-2"
        style={{
          height: 36,
          padding: '0 12px',
          borderRadius: 'var(--r-field)',
          background: 'var(--field-background)',
          boxShadow: 'inset 0 0 0 1px var(--border)',
          opacity: disabled ? 0.5 : 1,
        }}
      >
        <select
          aria-label={label}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value as T)}
          className="min-w-0 flex-1"
          style={{
            appearance: 'none',
            border: 0,
            outline: 'none',
            background: 'transparent',
            color: 'var(--foreground)',
            fontSize: 14,
            fontFamily: 'inherit',
          }}
        >
          {options.map((o) => (
            <option key={o.key} value={o.key}>
              {o.label}
            </option>
          ))}
        </select>
        {hint ? (
          <span className="mono shrink-0" style={{ fontSize: 12, color: 'var(--muted)' }}>
            {hint}
          </span>
        ) : (
          <svg
            width="16"
            height="16"
            viewBox="0 0 20 20"
            fill="none"
            stroke="var(--muted)"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="shrink-0"
          >
            <path d="M5 8l5 5 5-5" />
          </svg>
        )}
      </div>
    </div>
  )
}
