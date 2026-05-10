import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3005,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/video_feed': {
        target: 'http://localhost:5000',
        changeOrigin: true
      },
      '/validation_demos': {
        target: 'http://localhost:5000',
        changeOrigin: true
      }
    }
  }
})
