import { lazy, Suspense, useEffect } from 'react'
import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/layout/app-shell'
import { Skeleton } from './components/ui/skeleton'
import { api } from './lib/api/client'
import { isDemo } from './lib/api/config'
import { live } from './lib/live'

const SttPage = lazy(() => import('./routes/stt'))
const TtsPage = lazy(() => import('./routes/tts'))
const ModelsPage = lazy(() => import('./routes/models'))
const LatencyPage = lazy(() => import('./routes/latency'))
const BenchmarkPage = lazy(() => import('./routes/benchmark'))
const ConnectPage = lazy(() => import('./routes/connect'))
const ConversationPage = lazy(() => import('./routes/conversation'))
const ComponentsPage = lazy(() => import('./routes/components'))
const WordListsPage = lazy(() => import('./routes/wordlists'))
const ProvidersPage = lazy(() => import('./routes/providers'))
const PhonePage = lazy(() => import('./routes/phone'))
const KeysPage = lazy(() => import('./routes/keys'))
const WakePage = lazy(() => import('./routes/wake'))
const DecisionsPage = lazy(() => import('./routes/decisions'))
const ClassifyPage = lazy(() => import('./routes/classify'))
const ComparePage = lazy(() => import('./routes/compare'))
const DiarizePage = lazy(() => import('./routes/diarize'))
const CloneBrowserPage = lazy(() => import('./routes/clone-browser'))
const AnalysePage = lazy(() => import('./routes/analyse'))
const DialoguePage = lazy(() => import('./routes/dialogue'))
const ConvertPage = lazy(() => import('./routes/convert'))
const SpeakersPage = lazy(() => import('./routes/speakers'))
const WatermarkPage = lazy(() => import('./routes/watermark'))
const PipelinesPage = lazy(() => import('./routes/pipelines'))

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
            path="speakers"
            element={<Suspense fallback={<PageFallback />}>{isDemo ? <DiarizePage /> : <SpeakersPage />}</Suspense>}
          />
          <Route
            path="clone"
            element={
              <Suspense fallback={<PageFallback />}>
                <CloneBrowserPage />
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
          <Route
            path="pipelines"
            element={
              <Suspense fallback={<PageFallback />}>
                <PipelinesPage />
              </Suspense>
            }
          />
          <Route
            path="watermark"
            element={
              <Suspense fallback={<PageFallback />}>
                <WatermarkPage />
              </Suspense>
            }
          />
          <Route
            path="convert"
            element={
              <Suspense fallback={<PageFallback />}>
                <ConvertPage />
              </Suspense>
            }
          />
          <Route
            path="dialogue"
            element={
              <Suspense fallback={<PageFallback />}>
                <DialoguePage />
              </Suspense>
            }
          />
          <Route
            path="conversation"
            element={
              <Suspense fallback={<PageFallback />}>
                <ConversationPage />
              </Suspense>
            }
          />
          <Route
            path="components"
            element={
              <Suspense fallback={<PageFallback />}>
                <ComponentsPage />
              </Suspense>
            }
          />
          <Route
            path="wordlists"
            element={
              <Suspense fallback={<PageFallback />}>
                <WordListsPage />
              </Suspense>
            }
          />
          <Route
            path="providers"
            element={
              <Suspense fallback={<PageFallback />}>
                <ProvidersPage />
              </Suspense>
            }
          />
          <Route
            path="keys"
            element={
              <Suspense fallback={<PageFallback />}>
                <KeysPage />
              </Suspense>
            }
          />
          <Route
            path="classify"
            element={
              <Suspense fallback={<PageFallback />}>
                <ClassifyPage />
              </Suspense>
            }
          />
          <Route
            path="decisions"
            element={
              <Suspense fallback={<PageFallback />}>
                <DecisionsPage />
              </Suspense>
            }
          />
          <Route
            path="wake"
            element={
              <Suspense fallback={<PageFallback />}>
                <WakePage />
              </Suspense>
            }
          />
          <Route
            path="phone"
            element={
              <Suspense fallback={<PageFallback />}>
                <PhonePage />
              </Suspense>
            }
          />
          <Route
            path="compare"
            element={
              <Suspense fallback={<PageFallback />}>
                <ComparePage />
              </Suspense>
            }
          />
          <Route
            path="analyse"
            element={
              <Suspense fallback={<PageFallback />}>
                <AnalysePage />
              </Suspense>
            }
          />
          <Route path="*" element={<Navigate to="/stt" replace />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}
