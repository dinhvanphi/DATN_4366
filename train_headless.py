"""
Headless training + benchmark script.
Trains PPO without matplotlib GUI, then runs comparison benchmark.

Key improvements:
- Environment reset giữa các episode → reward không drift
- Nhiều frames hơn (200k) → agent có đủ data để học
- 16 parallel envs → tăng tốc thu thập transitions
"""
import os
import random
from collections import deque

import matplotlib
matplotlib.use("Agg")

from traffic_constants import CAR_COLORS, H, W, SPAWN_BLOCK_DIST, CAR_H
from traffic_entities import Car, DemandController
from traffic_rl import TrafficLightRL, GREEN_DURATIONS, HeadlessEnv, _make_tl
from traffic_ppo import PPOAgent


def headless_train(total_frames=200000, n_envs=16, base_seed=42):
    """Headless parallel training — no GUI, with env reset."""
    print(f"Starting headless training: {total_frames} frames, {n_envs} envs")
    print(f"Action space: {GREEN_DURATIONS}")

    for f in ("ppo_model.pth", "ppo_model_last.pth", "ppo_checkpoint.pth"):
        if os.path.exists(f):
            os.remove(f)

    agent = PPOAgent(state_size=TrafficLightRL.STATE_SIZE, action_size=TrafficLightRL.ACTION_SIZE)
    envs = [HeadlessEnv(seed=base_seed + i * 1000) for i in range(n_envs)]
    tls = [_make_tl(agent) for _ in range(n_envs)]

    best_score = -float("inf")
    rq = deque(maxlen=500)
    rw = deque(maxlen=500)

    episode_resets = 0

    for f in range(total_frames):
        frame = f + 1
        for i in range(n_envs):
            tp = envs[i].step(tls[i])
            tls[i].step(envs[i].cars, throughput=tp, training=True)

            # Reset env khi episode kết thúc
            if tls[i].episode_just_ended:
                envs[i].reset()
                tls[i].reset_state()
                tls[i].episode_just_ended = False
                episode_resets += 1

        # Track env 0
        wn = sum(1 for c in envs[0].cars if c.is_waiting and not c._past_stop())
        qn = len(envs[0].cars)
        rw.append(wn)
        rq.append(qn)

        if frame % 10000 == 0:
            aw = sum(rw) / len(rw)
            aq = sum(rq) / len(rq)
            sc = -1.8 * aw - 1.0 * aq
            if sc > best_score:
                best_score = sc
                agent.save("ppo_model.pth")
                print(f"  [BEST] f={frame} wait={aw:.1f} q={aq:.1f} sc={sc:.1f} steps={agent.total_steps} resets={episode_resets}")
            else:
                print(f"  f={frame} wait={aw:.1f} q={aq:.1f} sc={sc:.1f} best={best_score:.1f} steps={agent.total_steps} resets={episode_resets}")

    agent.save("ppo_model_last.pth")
    print(f"\nTraining done. Total agent steps: {agent.total_steps}, episode resets: {episode_resets}")
    return agent


def headless_benchmark(model_path="ppo_model.pth", seed=42, max_frames=2000):
    """Run benchmark without GUI."""
    from traffic_simulation import SimulationBenchmark

    rng = random.Random(seed)
    demand = DemandController(rng=rng)
    schedule = []
    for _ in range(max_frames):
        demand.tick()
        fc = {}
        for d in ("N", "S", "E", "W"):
            p_base = demand.base_prob(d, {"N": 0.010, "S": 0.010, "E": 0.012, "W": 0.012})
            p_burst = demand.burst_prob(d, 0.030)
            count = 0
            if rng.random() < p_base: count += 1
            if rng.random() < p_burst: count += rng.randint(1, 3)
            if count: fc[d] = min(count, 3)
        schedule.append(fc)

    rng_f = random.Random(seed)
    sim_f = SimulationBenchmark(use_rl=False, max_frames=max_frames, rng=rng_f, spawn_schedule=schedule)
    sim_f.run()
    rf = sim_f.get_results()

    rng_p = random.Random(seed)
    sim_p = SimulationBenchmark(use_rl=True, model_path=model_path, max_frames=max_frames, rng=rng_p, spawn_schedule=schedule)
    sim_p.run()
    rp = sim_p.get_results()

    return rf, rp


if __name__ == "__main__":
    # 1. Train
    headless_train(total_frames=200000, n_envs=16)

    # 2. Benchmark
    print("\n" + "=" * 62)
    print("BENCHMARK RESULTS")
    print("=" * 62)

    all_fixed_wait = []
    all_ppo_wait = []

    for model_path, label in [("ppo_model.pth", "PPO(best)"), ("ppo_model_last.pth", "PPO(last)")]:
        if not os.path.exists(model_path):
            print(f"Skipping {model_path} (not found)")
            continue

        print(f"\n{'='*62}")
        print(f"Model: {model_path}")
        print(f"{'='*62}")

        for seed in [42, 43, 44, 45, 46]:
            rf, rp = headless_benchmark(model_path=model_path, seed=seed)
            wait_diff = rf['avg_wait_per_car'] - rp['avg_wait_per_car']
            wait_pct = (wait_diff / max(rf['avg_wait_per_car'], 0.001)) * 100
            symbol = "✓" if wait_diff > 0 else "✗"
            print(f"  Seed {seed}: Fixed={rf['avg_wait_per_car']:.1f} {label}={rp['avg_wait_per_car']:.1f}  "
                  f"{'giảm' if wait_diff > 0 else 'tăng'} {abs(wait_pct):.1f}% {symbol}  "
                  f"switches: F={rf['phase_switches']} P={rp['phase_switches']}")

            if label == "PPO(best)":
                all_fixed_wait.append(rf['avg_wait_per_car'])
                all_ppo_wait.append(rp['avg_wait_per_car'])

    if all_fixed_wait:
        avg_f = sum(all_fixed_wait) / len(all_fixed_wait)
        avg_p = sum(all_ppo_wait) / len(all_ppo_wait)
        diff = avg_f - avg_p
        pct = (diff / max(avg_f, 0.001)) * 100
        print(f"\n{'='*62}")
        print(f"TRUNG BÌNH 5 SEED:")
        print(f"  Fixed: {avg_f:.1f}  PPO(best): {avg_p:.1f}")
        print(f"  ★ PPO {'giảm' if diff > 0 else 'tăng'} {abs(pct):.1f}% TG chờ TB/xe")
        print(f"{'='*62}")
