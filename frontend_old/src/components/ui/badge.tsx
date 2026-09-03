import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
    "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
    {
        variants: {
            variant: {
                default:
                    "border-transparent bg-[var(--accent)] text-white",
                secondary:
                    "border-transparent bg-[var(--bg-primary)] text-[var(--text-primary)]",
                success:
                    "border-transparent bg-[var(--success)]/15 text-[var(--success)]",
                destructive:
                    "border-transparent bg-[var(--danger)]/15 text-[var(--danger)]",
                outline: "border-[var(--border-muted)] text-[var(--text-secondary)]",
            },
        },
        defaultVariants: {
            variant: "default",
        },
    }
)

export interface BadgeProps
    extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> { }

function Badge({ className, variant, ...props }: BadgeProps) {
    return (
        <div className={cn(badgeVariants({ variant }), className)} {...props} />
    )
}

export { Badge, badgeVariants }
