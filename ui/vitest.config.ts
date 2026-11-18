import path from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'
import { FS_ALLOW, boloVersion } from './vite.config'

export default defineConfig({
  plugins: [react()],
  define: { 'import.meta.env.VITE_BOLO_VERSION': JSON.stringify(boloVersion()) },
  server: { fs: { allow: FS_ALLOW } },
  resolve: { alias: { '@': path.resolve(import.meta.dirname, './src') } },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
  },
})
