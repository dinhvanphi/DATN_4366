import type { TrafficSim } from './engine'

type Lane = 'N' | 'S' | 'E' | 'W'

const LANES: Lane[] = ['N', 'S', 'E', 'W']

export const drawSim = (ctx: CanvasRenderingContext2D, sim: TrafficSim) => {
  const { width, height } = ctx.canvas
  ctx.clearRect(0, 0, width, height)

  ctx.fillStyle = '#0b0f14'
  ctx.fillRect(0, 0, width, height)

  const roadColor = '#0f172a'
  ctx.fillStyle = roadColor
  ctx.fillRect(width * 0.32, 0, width * 0.36, height)
  ctx.fillRect(0, height * 0.32, width, height * 0.36)

  ctx.strokeStyle = 'rgba(255,255,255,0.08)'
  ctx.lineWidth = 2
  ctx.strokeRect(width * 0.32, 0, width * 0.36, height)
  ctx.strokeRect(0, height * 0.32, width, height * 0.36)

  const maxCars = 12
  const dotRadius = 4
  const offset = 14

  const drawLane = (lane: Lane, x: number, y: number, dx: number, dy: number) => {
    const count = Math.min(sim.getQueueCount(lane), maxCars)
    for (let i = 0; i < count; i += 1) {
      const px = x + dx * (i + 1) * offset
      const py = y + dy * (i + 1) * offset
      ctx.beginPath()
      ctx.fillStyle = lane === 'N' || lane === 'S' ? '#2dd4bf' : '#fbbf24'
      ctx.arc(px, py, dotRadius, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  drawLane('N', width * 0.5, height * 0.28, 0, -1)
  drawLane('S', width * 0.5, height * 0.72, 0, 1)
  drawLane('E', width * 0.72, height * 0.5, 1, 0)
  drawLane('W', width * 0.28, height * 0.5, -1, 0)

  const lightColor = (isOn: boolean, color: string) => (isOn ? color : 'rgba(255,255,255,0.15)')
  const lights = [
    { x: width * 0.5, y: height * 0.08 },
    { x: width * 0.5, y: height * 0.92 },
    { x: width * 0.92, y: height * 0.5 },
    { x: width * 0.08, y: height * 0.5 }
  ]

  const nsGreen = sim.phase === 0 && !sim.inYellow && !sim.inAllRed
  const ewGreen = sim.phase === 1 && !sim.inYellow && !sim.inAllRed

  lights.forEach((pos, index) => {
    ctx.fillStyle = '#10161e'
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'
    ctx.beginPath()
    ctx.roundRect(pos.x - 12, pos.y - 18, 24, 36, 6)
    ctx.fill()
    ctx.stroke()

    const isNS = index < 2
    const greenOn = isNS ? nsGreen : ewGreen
    const redOn = !greenOn
    const yellowOn = sim.inYellow

    const colors = [
      lightColor(redOn, '#f87171'),
      lightColor(yellowOn, '#fbbf24'),
      lightColor(greenOn, '#34d399')
    ]

    colors.forEach((color, idx) => {
      ctx.beginPath()
      ctx.fillStyle = color
      ctx.arc(pos.x, pos.y - 10 + idx * 10, 4, 0, Math.PI * 2)
      ctx.fill()
    })
  })

  ctx.fillStyle = 'rgba(255,255,255,0.6)'
  ctx.font = '12px Space Grotesk, sans-serif'
  ctx.fillText('N', width * 0.5 + 10, height * 0.06)
  ctx.fillText('S', width * 0.5 - 12, height * 0.96)
  ctx.fillText('E', width * 0.94, height * 0.5 - 8)
  ctx.fillText('W', width * 0.04, height * 0.5 - 8)

  ctx.fillStyle = 'rgba(255,255,255,0.4)'
  ctx.font = '11px IBM Plex Mono, monospace'
  ctx.fillText(`Frame: ${sim.frame}`, 12, height - 12)
}
