import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { CitizenData, Step } from '../components/hangul'
import { socket, registerRole } from '../socket'
import {
  StepStart, StepConsent, StepPrivacyPolicy, StepName, StepDob, StepGender,
  StepPhone, StepConfirm, StepWaiting
} from '../components/KioskSteps'

export default function KioskIntakePage() {
  const navigate = useNavigate()
  const [step, setStep] = useState<Step>('start')
  const [data, setData] = useState<CitizenData>({ name: '', dob: '', gender: '', phone: '010' })

  const dataRef = useRef(data)
  const stepRef = useRef(step)

  useEffect(() => { dataRef.current = data }, [data])
  useEffect(() => { stepRef.current = step }, [step])

  useEffect(() => {
    registerRole('kiosk')
  }, [])

  // 상담원이 준비되면 세션 화면으로 이동
  useEffect(() => {
    const handleAgentReady = () => {
      if (stepRef.current === 'waiting') {
        navigate('/kiosk/session', { state: { citizenData: dataRef.current } })
      }
    }
    socket.on('agent_ready', handleAgentReady)
    return () => { socket.off('agent_ready', handleAgentReady) }
  }, [navigate])

  const go = (s: Step) => setStep(s)

  const handleFinish = () => {
    setStep('waiting')
    sessionStorage.setItem('current_citizen_data', JSON.stringify(dataRef.current))
    socket.emit('citizen_arrived', { citizenData: dataRef.current })
  }

  const renderStep = () => {
    switch (step) {
      case 'start':   return <StepStart go={go} />
      case 'consent': return <StepConsent go={go} />
      case 'policy':  return <StepPrivacyPolicy go={go} />
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
