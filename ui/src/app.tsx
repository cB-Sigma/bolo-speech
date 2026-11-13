import { lazy, Suspense, useEffect } from 'react'
import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/layout/app-shell'
import { Skeleton } from './components/ui/skeleton'
import { api } from './lib/api/client'
import { live } from './lib/live'

const SttPage = lazy(() => import('./routes/stt'))
const TtsPage = lazy(() => import('./routes/tts'))
const ModelsPage = lazy(() => import('./routes/models'))
const LatencyPage = lazy(() => import('./routes/latency'))
const BenchmarkPage = lazy(() => import('./routes/benchmark'))
const ConnectPage = lazy(() => import('./routes/connect'))

function PageFallback() {
  return (
    <div className="mx-auto grid w-full max-w-[1400px] gap-4 px-4 py-6 sm:px-6" aria-busy="true" aria-label="Loading">
      <Skeleton className="h-9 w-64" />
      <div className="grid gap-4 md:grid-cols-3">
        <Skeleton className="h-28" />
        <Skeleton className="h-28" />
        <Skeleton className="h-28" />
      </div>
      <Skeleton className="h-72" />
    </div>
  )
}

export function App() {
  // Recent traces predate the live feed; load them once so the dashboard opens populated.
  useEffect(() => {
    api
      .requests({ limit: 300 })
      .then((r) => live.seedRequests(r.requests))
      .catch(() => undefined)
  }, [])

  return (
    <HashRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Navigate to="/stt" replace />} />
          <Route
            path="stt"
            element={
              <Suspense fallback={<PageFallback />}>
                <SttPage />
              </Suspense>
            }
          />
          <Route
            path="tts"
            element={
              <Suspense fallback={<PageFallback />}>
                <TtsPage />
              </Suspense>
            }
          />
          <Route
            path="models"
            element={
              <Suspense fallback={<PageFallback />}>
                <ModelsPage />
              </Suspense>
            }
          />
          <Route
            path="latency"
            element={
              <Suspense fallback={<PageFallback />}>
                <LatencyPage />
              </Suspense>
            }
          />
          <Route
            path="benchmark"
            element={
              <Suspense fallback={<PageFallback />}>
                <BenchmarkPage />
              </Suspense>
            }
          />
          <Route
            path="connect"
            element={
              <Suspense fallback={<PageFallback />}>
                <ConnectPage />
              </Suspense>
            }
          />
          <Route path="*" element={<Navigate to="/stt" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}
