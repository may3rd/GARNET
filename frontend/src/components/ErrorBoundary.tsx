import React, { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertCircle, RefreshCw, Home } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onReset?: () => void
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

/**
 * ErrorBoundary component to catch React errors and display a fallback UI.
 * Prevents the entire app from crashing when a component throws an error.
 */
export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ error, errorInfo })

    // Log errors in development for debugging
    if (import.meta.env.DEV) {
      console.error('ErrorBoundary caught an error:', error, errorInfo)
    }

    // Log to error tracking service in production
    if (import.meta.env.PROD) {
      // TODO: Send to error tracking service (e.g., Sentry)
      // logErrorToService(error, errorInfo);
    }
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null })
    this.props.onReset?.()
  }

  private handleReload = () => {
    window.location.reload()
  }

  private handleGoHome = () => {
    window.location.href = '/'
  }

  public render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="min-h-screen flex items-center justify-center bg-[var(--bg-primary)] p-4">
          <div className="max-w-md w-full bg-[var(--bg-secondary)] rounded-xl border border-[var(--border-muted)] p-6 shadow-lg">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-red-500/10 flex items-center justify-center">
                <AlertCircle className="w-5 h-5 text-red-500" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-[var(--text-primary)]">
                  Something went wrong
                </h2>
                <p className="text-sm text-[var(--text-muted)]">
                  An unexpected error occurred
                </p>
              </div>
            </div>

            <div className="bg-[var(--bg-primary)] rounded-lg p-3 mb-4 overflow-auto">
              <code className="text-xs text-red-400 font-mono break-all">
                {this.state.error?.message || 'Unknown error'}
              </code>
            </div>

            {import.meta.env.DEV && this.state.errorInfo && (
              <details className="mb-4">
                <summary className="text-sm text-[var(--text-secondary)] cursor-pointer hover:text-[var(--text-primary)]">
                  Stack trace
                </summary>
                <pre className="mt-2 text-xs text-[var(--text-muted)] bg-[var(--bg-primary)] rounded-lg p-3 overflow-auto max-h-40">
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}

            <div className="flex flex-wrap gap-2">
              <Button
                onClick={this.handleReset}
                variant="default"
                size="sm"
                className="flex-1"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Try Again
              </Button>
              <Button
                onClick={this.handleReload}
                variant="outline"
                size="sm"
              >
                Reload
              </Button>
              <Button
                onClick={this.handleGoHome}
                variant="outline"
                size="sm"
              >
                <Home className="w-4 h-4 mr-2" />
                Home
              </Button>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

/**
 * Hook to get error boundary functionality in functional components
 */
export function useErrorBoundary() {
  const [error, setError] = React.useState<Error | null>(null)

  if (error) {
    throw error
  }

  return { showBoundary: setError }
}

export default ErrorBoundary
