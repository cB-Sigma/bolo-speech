import path from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'
import { boloVersion } from './vite.config'

export default defineConfig({
  plugins: [react()],
  define: { 'import.meta.env.VITE_BOLO_VERSION': JSON.stringify(boloVersion()) },
  resolve: { alias: { '@': path.resolve(import.meta.dirname, './src') } },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
  },
})
