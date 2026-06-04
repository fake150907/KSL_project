import { Navigate, Route, Routes } from 'react-router-dom'
import { useAuthExpiredRedirect } from './hooks/useAuth'
import KioskLoginPage from './pages/KioskLoginPage'
import KioskStandbyPage from './pages/KioskStandbyPage'
import KioskIntakePage from './pages/KioskIntakePage'
import KioskSessionPage from './pages/KioskSessionPage'

export default function App() {
  useAuthExpiredRedirect()

  return (
    <Routes>
      <Route path="/kiosk"          element={<KioskLoginPage />} />
      <Route path="/kiosk/standby"  element={<KioskStandbyPage />} />
      <Route path="/kiosk/intake"   element={<KioskIntakePage />} />
      <Route path="/kiosk/session"  element={<KioskSessionPage />} />
      <Route path="*"               element={<Navigate to="/kiosk" replace />} />
    </Routes>
  )
}
