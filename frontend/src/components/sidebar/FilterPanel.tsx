/**
 * FilterPanel component - confidence filter and search
 */
import { Search } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { useUIStore } from '@/stores/uiStore'
import { MIN_CONFIDENCE, MAX_CONFIDENCE } from '@/lib/constants'

export function FilterPanel() {
    const { confidenceFilter, setConfidenceFilter, searchQuery, setSearchQuery } =
        useUIStore()

    return (
        <div className="space-y-4 border-b p-4">
            {/* Search */}
            <div className="relative">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                    type="text"
                    placeholder="Search objects..."
                    className="pl-9"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                />
            </div>

            {/* Confidence Filter */}
            <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Confidence ≥</span>
                    <span className="font-medium">{Math.round(confidenceFilter * 100)}%</span>
                </div>
                <Slider
                    value={[confidenceFilter]}
                    onValueChange={([v]) => setConfidenceFilter(v)}
                    min={MIN_CONFIDENCE}
                    max={MAX_CONFIDENCE}
                    step={0.05}
                />
            </div>
        </div>
    )
}
