import express from 'express'
import cors from 'cors'
import { spawn } from 'child_process'
import path from 'path'
import { fileURLToPath } from 'url'
import readline from 'readline'

const app = express()
app.use(cors())
app.use(express.json())

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const repoRoot = path.resolve(__dirname, '..', '..')

let currentProcess = null
let logBuffer = []
const clients = new Set()

let inferProcess = null
let inferReady = false
let inferId = 0
const pending = new Map()

const appendLog = (line) => {
  const text = line.toString().replace(/\r/g, '')
  const parts = text.split('\n').filter(Boolean)
  for (const part of parts) {
    logBuffer.push(part)
  }
  if (logBuffer.length > 500) {
    logBuffer = logBuffer.slice(-500)
  }
  for (const client of clients) {
    client.write(`data: ${JSON.stringify(parts)}\n\n`)
  }
}

const clearProcess = () => {
  currentProcess = null
}

const startInfer = () => {
  if (inferProcess) return
  const inferPath = path.join(__dirname, 'infer.py')
  inferProcess = spawn('python3', [inferPath], {
    cwd: repoRoot,
    env: { ...process.env, PPO_MODEL_PATH: path.join(repoRoot, 'ppo_model.pth') },
    stdio: ['pipe', 'pipe', 'pipe']
  })

  const rl = readline.createInterface({ input: inferProcess.stdout })
  rl.on('line', (line) => {
    try {
      const msg = JSON.parse(line)
      const resolver = pending.get(msg.id)
      if (resolver) {
        resolver(msg.action)
        pending.delete(msg.id)
      }
    } catch {
      // ignore malformed output
    }
  })

  inferProcess.stderr.on('data', (data) => appendLog(data))
  inferProcess.on('close', () => {
    inferProcess = null
    inferReady = false
  })

  inferReady = true
}

const requestAction = (state) =>
  new Promise((resolve, reject) => {
    if (!inferProcess || !inferReady) {
      return reject(new Error('Inference not ready'))
    }
    inferId += 1
    const id = inferId
    pending.set(id, resolve)
    inferProcess.stdin.write(`${JSON.stringify({ id, state })}\n`)
    setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id)
        resolve(0)
      }
    }, 200)
  })

app.get('/api/status', (_req, res) => {
  res.json({ running: Boolean(currentProcess) })
})

app.post('/api/run', (req, res) => {
  if (currentProcess) {
    res.status(409).json({ error: 'Process already running' })
    return
  }
  const choice = String(req.body?.choice ?? '').trim()
  if (!choice) {
    res.status(400).json({ error: 'Missing choice' })
    return
  }

  logBuffer = []
  const child = spawn('python3', ['traffic_sim.py'], {
    cwd: repoRoot,
    env: process.env,
    stdio: ['pipe', 'pipe', 'pipe']
  })
  currentProcess = child

  appendLog(`> python3 traffic_sim.py (choice ${choice})`)
  child.stdin.write(`${choice}\n`)

  child.stdout.on('data', (data) => appendLog(data))
  child.stderr.on('data', (data) => appendLog(data))

  child.on('close', (code) => {
    appendLog(`Process exited with code ${code}`)
    clearProcess()
  })

  res.json({ ok: true })
})

app.post('/api/action', async (req, res) => {
  try {
    if (!inferProcess) startInfer()
    const state = req.body?.state
    if (!Array.isArray(state)) {
      res.status(400).json({ error: 'Missing state' })
      return
    }
    const action = await requestAction(state)
    res.json({ action })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.post('/api/stop', (_req, res) => {
  if (!currentProcess) {
    res.json({ ok: true })
    return
  }
  currentProcess.kill('SIGTERM')
  appendLog('Sent SIGTERM to process')
  clearProcess()
  res.json({ ok: true })
})

app.get('/api/stream', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection', 'keep-alive')
  res.flushHeaders()

  clients.add(res)
  res.write(`data: ${JSON.stringify(logBuffer)}\n\n`)

  req.on('close', () => {
    clients.delete(res)
  })
})

const port = 5174
app.listen(port, () => {
  console.log(`API server running on http://localhost:${port}`)
})
