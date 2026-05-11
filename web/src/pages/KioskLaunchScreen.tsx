import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { PatientData, Step } from '../components/hangul'
import { socket, registerRole } from '../socket' // ✅ 수정됨: registerRole 추가 임포트

import {
  StepStart, StepName, StepDob, StepGender,
  StepPhone, StepConfirm, StepWaiting
} from '../components/KioskSteps'

export default function KioskLaunchScreen() {
  const navigate = useNavigate()

  const [step, setStep] = useState<Step>('start')
  const [data, setData] = useState<PatientData>({ name: '', dob: '', gender: '', phone: '010' })
  const dataRef = useRef(data)

  // 최신 데이터 동기화
  useEffect(() => {
    dataRef.current = data
  }, [data])

  // ✅ 수정됨: 컴포넌트 마운트 시 키오스크 역할 등록
  useEffect(() => {
    registerRole('kiosk')
  }, [])

  // socket.io: doctor_ready 수신 → 진료실 이동
  useEffect(() => {
    const handleDoctorReady = () => {
      navigate('/kiosk/session', { state: { patientData: dataRef.current } })
    }

    socket.on('doctor_ready', handleDoctorReady)
    return () => {
      socket.off('doctor_ready', handleDoctorReady)
    }
  }, [navigate])

  const go = (s: Step) => setStep(s)

  const handleFinish = () => {
    setStep('waiting')
    // socket.io: 환자 도착 알림 → 서버 → 의사 대기실
    socket.emit('patient_arrived', { patientData: dataRef.current })
  }

  const renderStep = () => {
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

  return (
    <div className="fixed inset-0 bg-slate-100 flex flex-col overflow-hidden">
      <div className="flex flex-col flex-1 w-full h-full bg-white text-slate-900 overflow-y-auto">
        {renderStep()}
      </div>
    </div>
  )
}