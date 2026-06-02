import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current working directory.
  // Set the third parameter to '' to load all env regardless of the `VITE_` prefix.
  const env = loadEnv(mode, process.cwd(), '')

  // API target URL from environment variable or default
  const apiTarget = env.VITE_API_URL || 'http://localhost:8001'

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: parseInt(env.VITE_PORT || '5173'),
      host: env.VITE_HOST || 'localhost',
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
        },
        '/runs': {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
    build: {
      // Generate source maps for production debugging
      sourcemap: env.VITE_SOURCEMAP === 'true',
      // Output directory (relative to project root)
      outDir: env.VITE_OUT_DIR || 'dist',
      // Report bundle size
      reportCompressedSize: true,
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (!id.includes('node_modules')) return undefined
            if (id.includes('/react/') || id.includes('/react-dom/') || id.includes('/scheduler/')) {
              return 'vendor-react'
            }
            if (id.includes('/konva/') || id.includes('/react-konva/')) {
              return 'vendor-canvas'
            }
            if (id.includes('/jspdf/')) {
              return 'vendor-jspdf'
            }
            if (id.includes('/html2canvas/')) {
              return 'vendor-html2canvas'
            }
            if (id.includes('/dompurify/')) {
              return 'vendor-dompurify'
            }
            if (id.includes('/@radix-ui/') || id.includes('/lucide-react/')) {
              return 'vendor-ui'
            }
            return undefined
          },
        },
      },
    },
  }
})
