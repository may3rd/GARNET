import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

const here = import.meta.dirname

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = env.VITE_API_URL || 'http://localhost:8001'

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: { '@': path.resolve(here, './src') },
    },
    server: {
      port: parseInt(env.VITE_PORT || '5173'),
      host: env.VITE_HOST || 'localhost',
      proxy: {
        '/api': { target: apiTarget, changeOrigin: true },
        '/runs': { target: apiTarget, changeOrigin: true },
      },
    },
    build: {
      sourcemap: env.VITE_SOURCEMAP === 'true',
      outDir: env.VITE_OUT_DIR || 'dist',
    },
  }
})
