/**
 * Layout component - main app shell
 */
import { useEffect, type ReactNode } from 'react'
import { Header } from './Header'
import { useUIStore } from '@/stores/uiStore'

interface LayoutProps {
    children: ReactNode
}

export function Layout({ children }: LayoutProps) {
    const { theme } = useUIStore()

    // Apply theme on mount and changes
    useEffect(() => {
        const root = document.documentElement
        if (theme === 'system') {
            const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
            root.classList.toggle('dark', isDark)
        } else {
            root.classList.toggle('dark', theme === 'dark')
        }
    }, [theme])

    return (
        <div className="flex h-screen flex-col bg-background text-foreground">
            <Header />
            <main className="flex-1 overflow-hidden">{children}</main>
        </div>
    )
}
