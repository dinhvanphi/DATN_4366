import './App.css'
import { useEffect, useMemo, useRef, useState } from 'react'
import { TrafficSim } from './sim/engine'
import { drawSim } from './sim/render'

type Mode = {
  id: string
  title: string
  subtitle: string
  detail: string
  choice: string
}

const MODES: Mode[] = [
  {
    id: 'fixed',
    title: '1. Fixed Timing',
    subtitle: 'Den chu ky co dinh',
    detail: 'Che do co dinh, khong dung RL. Phu hop so sanh base.',
    choice: '1'
  },
  {
    id: 'ppo-train',
    title: '2. PPO Training',
    subtitle: 'Huan luyen PPO',
    detail: 'Bat dau train PPO, luu checkpoint dinh ky.',
    choice: '2'
  },
  {
    id: 'ppo-test',
    title: '3. PPO Testing',
    subtitle: 'Chay voi model da train',
    detail: 'Chay model ppo_model.pth va quan sat thong so.',
    choice: '3'
  },
  {
    id: 'ppo-fresh',
    title: '4. PPO Fresh Training',
    subtitle: 'Xoa model cu, train tu dau',
    detail: 'Reset checkpoint, train moi tu dau.',
    choice: '4'
  },
  {
    id: 'compare',
    title: '5. Compare Benchmark',
    subtitle: 'Fixed vs PPO',
    detail: 'So sanh hieu qua, cung spawn schedule.',
    choice: '5'
  },
  {
    id: 'ppo-last',
    title: '6. PPO Testing',
    subtitle: 'ppo_model_last.pth',
    detail: 'Chay model cuoi cung trong session train.',
    choice: '6'
  },
  {
    id: 'compare-last',
    title: '7. Compare Benchmark',
    subtitle: 'Fixed vs ppo_model_last',
    detail: 'So sanh voi model last snapshot.',
    choice: '7'
  },
  {
    id: 'compare-seed',
    title: '8. Compare Benchmark',
    subtitle: 'Nhieu seed',
    detail: 'Danh gia on dinh qua nhieu seed.',
    choice: '8'
  }
]

function App() {
  const [selected, setSelected] = useState<Mode>(MODES[0])
  const [running, setRunning] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [statusText, setStatusText] = useState('Offline')
  const [metrics, setMetrics] = useState({
    avgQueue: 0,
    avgWaiting: 0,
    throughputRate: 0
  })
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const simRef = useRef<TrafficSim | null>(null)
  const rafRef = useRef<number | null>(null)
  const actionRef = useRef(0)
  const actionBusyRef = useRef(false)

  const isWebSimMode = ['fixed', 'ppo-test', 'ppo-last'].includes(selected.id)

  const logLines = useMemo(() => logs.slice(-6), [logs])

  useEffect(() => {
    const source = new EventSource('/api/stream')
    source.onmessage = (event) => {
      try {
        const lines = JSON.parse(event.data) as string[]
        if (lines.length) {
          setLogs((prev) => [...prev, ...lines])
        }
      } catch {
        setLogs((prev) => [...prev, event.data])
      }
    }
    source.onerror = () => {
      setStatusText('API offline')
    }
    return () => source.close()
  }, [])

  const refreshStatus = async () => {
    try {
      const res = await fetch('/api/status')
      const data = (await res.json()) as { running: boolean }
      setRunning(data.running)
      setStatusText(data.running ? 'Running' : 'Idle')
    } catch {
      setStatusText('API offline')
    }
  }

  useEffect(() => {
    refreshStatus()
    const timer = window.setInterval(refreshStatus, 3000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!simRef.current) {
      simRef.current = new TrafficSim()
    }
  }, [])

  const stopLoop = () => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }

  const loop = () => {
    const sim = simRef.current
    const canvas = canvasRef.current
    if (!sim || !canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    if (isWebSimMode) {
      if (sim.canDecide() && !actionBusyRef.current) {
        actionBusyRef.current = true
        fetch('/api/action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ state: sim.getStateVector() })
        })
          .then((res) => res.json())
          .then((data) => {
            actionRef.current = data.action ?? 0
          })
          .catch(() => {
            actionRef.current = 0
          })
          .finally(() => {
            actionBusyRef.current = false
          })
      }

      const action = actionRef.current
      sim.step(action)
      actionRef.current = 0
      drawSim(ctx, sim)
      setMetrics(sim.getMetrics())
    }

    rafRef.current = requestAnimationFrame(loop)
  }

  const handleRun = async () => {
    setStatusText('Starting...')
    if (isWebSimMode) {
      simRef.current?.reset()
      setLogs([])
      setRunning(true)
      stopLoop()
      rafRef.current = requestAnimationFrame(loop)
      setStatusText('Running (web)')
      return
    }
    await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ choice: selected.choice })
    })
    refreshStatus()
  }

  const handleStop = async () => {
    if (isWebSimMode) {
      stopLoop()
      setRunning(false)
      setStatusText('Idle')
      return
    }
    await fetch('/api/stop', { method: 'POST' })
    refreshStatus()
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <p className="eyebrow">Traffic RL Control Panel</p>
          <h1>Mo phong nga 4 giao thong</h1>
          <p className="subtitle">
            Giao dien don gian de chon che do chay cho du an PPO.
          </p>
        </div>
        <div className="status">
          <span className="pill">Engine: {statusText}</span>
          <span className="pill">Model: ppo_model.pth</span>
          <span className="pill">Seed: 42</span>
        </div>
      </header>

      <main className="layout">
        <section className="panel intersection-panel">
          <div className="panel-header">
            <h2>Giao lo</h2>
            <p>Minh hoa giao thong 4 huong</p>
          </div>

            <div className="intersection">
              <canvas ref={canvasRef} width={520} height={360} />
            </div>

          <div className="stats">
            <div>
              <span>Queue TB</span>
              <strong>{metrics.avgQueue.toFixed(2)}</strong>
            </div>
            <div>
              <span>Xe cho TB</span>
              <strong>{metrics.avgWaiting.toFixed(2)}</strong>
            </div>
            <div>
              <span>Throughput</span>
              <strong>{metrics.throughputRate.toFixed(2)} / phut</strong>
            </div>
          </div>
        </section>

        <section className="panel control-panel">
          <div className="panel-header">
            <h2>Che do chay</h2>
            <p>Chon mot che do de thuc thi</p>
          </div>

          <div className="mode-list">
            {MODES.map((mode) => (
              <button
                key={mode.id}
                type="button"
                className={`mode-item ${selected.id === mode.id ? 'active' : ''}`}
                onClick={() => setSelected(mode)}
              >
                <div>
                  <h3>{mode.title}</h3>
                  <p>{mode.subtitle}</p>
                </div>
                <span className="chevron">↗</span>
              </button>
            ))}
          </div>

          <div className="detail">
            <h3>Thong tin che do</h3>
            <p>{selected.detail}</p>
          </div>

          <div className="actions">
            <button className="primary" type="button" onClick={handleRun}>
              Run
            </button>
            <button className="ghost" type="button" onClick={handleStop} disabled={!running}>
              Stop
            </button>
            <button className="ghost" type="button" onClick={() => setLogs([])}>
              Reset
            </button>
          </div>

          <div className="log">
            <div className="log-header">
              <h3>Logs</h3>
              <span>Demo only</span>
            </div>
            <ul>
              {logLines.length === 0 ? (
                <li>Chua co log. Bam Run de bat dau.</li>
              ) : (
                logLines.map((line, index) => <li key={`${line}-${index}`}>{line}</li>)
              )}
            </ul>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
