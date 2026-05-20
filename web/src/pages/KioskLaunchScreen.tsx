import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { CitizenData, Step } from '../components/hangul'
import { socket, registerRole } from '../socket'

import {
  StepStart, StepName, StepDob, StepGender,
  StepPhone, StepConfirm, StepWaiting
} from '../components/KioskSteps'

export default function KioskLaunchScreen() {
  const navigate = useNavigate()

  const [step, setStep] = useState<Step>('start')
  const [data, setData] = useState<CitizenData>({ name: '', dob: '', gender: '', phone: '010' })
  
  const dataRef = useRef(data)
  // 💡 현재 진행 단계를 추적하기 위한 참조 변수 추가
  const stepRef = useRef(step)

  // 최신 데이터 동기화
  useEffect(() => {
    dataRef.current = data
  }, [data])

  // 💡 최신 진행 단계 동기화
  useEffect(() => {
    stepRef.current = step
  }, [step])

  // 컴포넌트 마운트 시 키오스크 역할 등록
  useEffect(() => {
    registerRole('kiosk')
  }, [])

  // socket.io: agent_ready 수신 → 상담실 이동
  useEffect(() => {
    const handleAgentReady = () => {
      // 💡 탑승객이 모두 탑승한 상태(waiting)인지 확인 후 기차 출발(navigate)
      if (stepRef.current === 'waiting') {
        navigate('/kiosk/session', { state: { citizenData: dataRef.current } })
      } else {
        console.warn('상담원이 준비되었으나, 민원인 정보 입력이 완료되지 않았습니다.')
      }
    }

    socket.on('agent_ready', handleAgentReady)
    return () => {
      socket.off('agent_ready', handleAgentReady)
    }
  }, [navigate])

  const go = (s: Step) => setStep(s)

  const handleFinish = () => {
    setStep('waiting')
    // socket.io: 민원인 도착 알림 → 서버 → 상담원 대기실
    socket.emit('citizen_arrived', { citizenData: dataRef.current })
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