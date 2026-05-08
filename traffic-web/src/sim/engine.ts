type Lane = 'N' | 'S' | 'E' | 'W'

const LANES: Lane[] = ['N', 'S', 'E', 'W']

type SpawnProb = Record<Lane, number>

type Metrics = {
  totalSpawned: number
  totalPassed: number
  avgQueue: number
  avgWaiting: number
  throughputRate: number
}

export class TrafficSim {
  phase = 0
  timeInPhase = 0
  inYellow = false
  inAllRed = false
  transitionTimer = 0
  frame = 0

  readonly minGreenTime = 90
  readonly decisionInterval = 12
  readonly yellowTime = 16
  readonly allRedTime = 6
  readonly maxGreen = 150

  queues: Record<Lane, number[]> = {
    N: [],
    S: [],
    E: [],
    W: []
  }

  totalSpawned = 0
  totalPassed = 0
  totalQueueHistory: number[] = []
  totalWaitingHistory: number[] = []

  spawnBase: SpawnProb = { N: 0.01, S: 0.01, E: 0.012, W: 0.012 }
  spawnBurst = 0.03
  spawnMaxBurst = 3

  reset() {
    this.phase = 0
    this.timeInPhase = 0
    this.inYellow = false
    this.inAllRed = false
    this.transitionTimer = 0
    this.frame = 0
    this.totalSpawned = 0
    this.totalPassed = 0
    this.totalQueueHistory = []
    this.totalWaitingHistory = []
    this.queues = { N: [], S: [], E: [], W: [] }
  }

  getQueueCount(lane: Lane) {
    return this.queues[lane].length
  }

  private spawnCars() {
    for (const lane of LANES) {
      let count = 0
      if (Math.random() < this.spawnBase[lane]) count += 1
      if (Math.random() < this.spawnBurst) count += 1 + Math.floor(Math.random() * this.spawnMaxBurst)
      count = Math.min(count, this.spawnMaxBurst)
      for (let i = 0; i < count; i += 1) {
        this.queues[lane].push(0)
        this.totalSpawned += 1
      }
    }
  }

  private incrementWait() {
    for (const lane of LANES) {
      this.queues[lane] = this.queues[lane].map((w) => w + 1)
    }
  }

  private serviceLane(lane: Lane) {
    const queue = this.queues[lane]
    if (queue.length === 0) return 0
    const service = Math.min(queue.length, queue.length > 5 ? 2 : 1)
    for (let i = 0; i < service; i += 1) {
      queue.shift()
    }
    this.totalPassed += service
    return service
  }

  private updatePhase(action: number) {
    if (this.inYellow || this.inAllRed) {
      this.transitionTimer -= 1
      if (this.inYellow && this.transitionTimer <= 0) {
        this.inYellow = false
        this.inAllRed = true
        this.transitionTimer = this.allRedTime
      } else if (this.inAllRed && this.transitionTimer <= 0) {
        this.inAllRed = false
        this.phase = 1 - this.phase
        this.timeInPhase = 0
      }
      return
    }

    if (action === 1 && this.timeInPhase >= this.minGreenTime) {
      this.inYellow = true
      this.transitionTimer = this.yellowTime
      return
    }

    this.timeInPhase += 1
  }

  canDecide() {
    return this.timeInPhase >= this.minGreenTime && this.timeInPhase % this.decisionInterval === 0
  }

  step(action: number) {
    this.frame += 1

    this.spawnCars()
    this.incrementWait()

    if (!this.inYellow && !this.inAllRed) {
      if (this.phase === 0) {
        this.serviceLane('N')
        this.serviceLane('S')
      } else {
        this.serviceLane('E')
        this.serviceLane('W')
      }
    }

    this.updatePhase(action)

    const totalQueue = LANES.reduce((sum, lane) => sum + this.queues[lane].length, 0)
    const totalWaiting = LANES.reduce(
      (sum, lane) => sum + this.queues[lane].length,
      0
    )
    this.totalQueueHistory.push(totalQueue)
    this.totalWaitingHistory.push(totalWaiting)
  }

  getStateVector() {
    const queueCounts: Record<Lane, number> = { N: 0, S: 0, E: 0, W: 0 }
    const waitingCounts: Record<Lane, number> = { N: 0, S: 0, E: 0, W: 0 }
    const waitingTime: Record<Lane, number> = { N: 0, S: 0, E: 0, W: 0 }

    for (const lane of LANES) {
      const q = this.queues[lane]
      queueCounts[lane] = q.length
      waitingCounts[lane] = q.length
      waitingTime[lane] = q.reduce((sum, w) => sum + w, 0)
    }

    const nsQueue = queueCounts.N + queueCounts.S
    const ewQueue = queueCounts.E + queueCounts.W
    const nsWait = waitingTime.N + waitingTime.S
    const ewWait = waitingTime.E + waitingTime.W
    const pressure = (nsQueue - ewQueue) / 20
    const waitPressure = (nsWait - ewWait) / 200
    const totalLoad = (nsQueue + ewQueue) / 20
    const phaseRatio = Math.min(this.timeInPhase / Math.max(this.maxGreen, 1), 1)

    return [
      queueCounts.N / 10,
      queueCounts.S / 10,
      queueCounts.E / 10,
      queueCounts.W / 10,
      waitingCounts.N / 10,
      waitingCounts.S / 10,
      waitingCounts.E / 10,
      waitingCounts.W / 10,
      Math.min(waitingTime.N / 200, 5),
      Math.min(waitingTime.S / 200, 5),
      Math.min(waitingTime.E / 200, 5),
      Math.min(waitingTime.W / 200, 5),
      this.phase,
      phaseRatio,
      pressure,
      waitPressure,
      totalLoad
    ]
  }

  getMetrics(): Metrics {
    const frames = Math.max(this.frame, 1)
    const avgQueue = this.totalQueueHistory.reduce((a, b) => a + b, 0) / frames
    const avgWaiting = this.totalWaitingHistory.reduce((a, b) => a + b, 0) / frames
    const throughputRate = (this.totalPassed / frames) * 60

    return {
      totalSpawned: this.totalSpawned,
      totalPassed: this.totalPassed,
      avgQueue,
      avgWaiting,
      throughputRate
    }
  }
}
