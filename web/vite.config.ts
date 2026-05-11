// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3005,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000', // 5001을 5000으로 변경
        changeOrigin: true
      },
      '/video_feed': {
        target: 'http://127.0.0.1:5000', // 여기도 변경
        changeOrigin: true
      },
      '/validation_demos': {
        target: 'http://127.0.0.1:5000', // 여기도 변경
        changeOrigin: true
      }
    }
  }
})