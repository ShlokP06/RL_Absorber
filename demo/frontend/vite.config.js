import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In Docker Compose the backend is reachable by service name, not localhost.
// Set BACKEND_URL=http://backend:8000 in the frontend container environment.
const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',   // expose outside the container
    port: 3000,
    proxy: {
      '/api': {
        target: backendUrl,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ws': {
        target: backendUrl.replace('http', 'ws'),
        ws: true,
      },
    },
  },
})
