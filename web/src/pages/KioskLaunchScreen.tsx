import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { PatientData, Step } from '../components/hangul'

// 💡 방금 생성한 분리된 컴포넌트들을 불러옵니다
import {
  StepStart, StepName, StepDob, StepGender,
  StepPhone, StepConfirm, StepWaiting
} from '../components/KioskSteps'

const NOTIFY_CHANNEL = 'patient-session-notify'

export default function KioskLaunchScreen() {
  const navigate = useNavigate()
  const channelRef = useRef<BroadcastChannel | null>(null)

  // 상태 관리 (데이터 및 현재 화면 단계)
  const [step, setStep] = useState<Step>('start')
  const [data, setData] = useState<PatientData>({ name: '', dob: '', gender: '', phone: '010' })
  const dataRef = useRef(data)

  // 최신 데이터 동기화 유지
  useEffect(() => {
    dataRef.current = data
  }, [data])

  // BroadcastChannel 수신 및 진료실 이동 로직
  useEffect(() => {
    const ch = new BroadcastChannel(NOTIFY_CHANNEL)
    channelRef.current = ch
    ch.onmessage = (e) => {
      if (e.data?.type === 'doctor_ready') {
        navigate('/kiosk/session', { state: { patientData: dataRef.current } })
      }
    }
    return () => ch.close()
  }, [navigate])

  const go = (s: Step) => setStep(s)

  const handleFinish = () => {
    setStep('waiting')
    channelRef.current?.postMessage({ type: 'patient_arrived', payload: data })
  }

  // 화면 단계(Step)에 따라 렌더링할 뷰(View)를 분기 처리
  switch (step) {
    case 'start':   return <StepStart go={go} />
    case 'name':    return <StepName data={data} setData={setData} go={go} />
    case 'dob':     return <StepDob data={data} setData={setData} go={go} />
    case 'gender':  return <StepGender data={data} setData={setData} go={go} />
    case 'phone':   return <StepPhone data={data} setData={setData} go={go} />
    case 'confirm': return <StepConfirm data={data} setData={setData} go={go} onFinish={handleFinish} />
    case 'waiting': return <StepWaiting data={data} setData={setData} go={go} />
    default:        return null
  }
}