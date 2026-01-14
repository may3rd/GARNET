/**
 * Header component - top navigation bar
 */
import { ArrowLeft, Moon, Settings, Sun } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TooltipProvider,
} from '@/components/ui/tooltip'
import { useUIStore } from '@/stores/uiStore'
import { useDetectionStore } from '@/stores/detectionStore'

export function Header() {
    const { theme, setTheme, currentView, setCurrentView } = useUIStore()
    const { imageName, reset } = useDetectionStore()

    const handleBack = () => {
        reset()
        setCurrentView('upload')
    }

    const toggleTheme = () => {
        setTheme(theme === 'dark' ? 'light' : 'dark')
    }

    return (
        <TooltipProvider>
            <header className="flex h-14 shrink-0 items-center justify-between border-b bg-background px-4">
                <div className="flex items-center gap-3">
                    {currentView !== 'upload' && (
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button variant="ghost" size="icon" onClick={handleBack}>
                                    <ArrowLeft className="h-4 w-4" />
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent>Back to Upload</TooltipContent>
                        </Tooltip>
                    )}

                    <div className="flex items-center gap-2">
                        <span className="text-xl font-bold tracking-tight text-primary">
                            GARNET
                        </span>
                        {imageName && (
                            <>
                                <span className="text-muted-foreground">|</span>
                                <span className="text-sm text-muted-foreground">{imageName}</span>
                            </>
                        )}
                    </div>
                </div>

                <div className="flex items-center gap-1">
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button variant="ghost" size="icon" onClick={toggleTheme}>
                                {theme === 'dark' ? (
                                    <Sun className="h-4 w-4" />
                                ) : (
                                    <Moon className="h-4 w-4" />
                                )}
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
                        </TooltipContent>
                    </Tooltip>

                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button variant="ghost" size="icon">
                                <Settings className="h-4 w-4" />
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent>Settings</TooltipContent>
                    </Tooltip>
                </div>
            </header>
        </TooltipProvider>
    )
}
