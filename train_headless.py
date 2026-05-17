"""
train_headless.py
-----------------
Headless training script for PPO traffic light controller.
No GUI, just trains and saves best model.
"""

import os
import random
import numpy as np
import torch
from traffic_simulation import SimulationBenchmark, SimulationRL
from traffic_entities import Car, DemandController
from traffic_constants import CAR_COLORS, CAR_H, H, W


def _build_spawn_schedule(seed, max_frames):
    rng = random.Random(seed)
    demand = DemandController(rng=rng)
    schedule = []
    for _ in range(max_frames):
        demand.tick()
        frame_counts = {}
        for d in ("N", "S", "E", "W"):
            p_base = demand.base_prob(d, {"N": 0.010, "S": 0.010, "E": 0.012, "W": 0.012})
            p_burst = demand.burst_prob(d, 0.030)
            count = 0
            if rng.random() < p_base:
                count += 1
            if rng.random() < p_burst:
                count += rng.randint(1, 3)
            if count:
                frame_counts[d] = min(count, 3)
        schedule.append(frame_counts)
    return schedule


def evaluate_checkpoint(model_path, seeds=(42, 43, 44, 45, 46), max_frames=2000, algorithm="ppo"):
    fixed_results = []
    ppo_results = []
    for seed in seeds:
        schedule = _build_spawn_schedule(seed, max_frames)
        fixed = SimulationBenchmark(
            use_rl=False,
            max_frames=max_frames,
            rng=random.Random(seed),
            spawn_schedule=schedule,
        )
        fixed.run()
        ppo = SimulationBenchmark(
            use_rl=True,
            model_path=model_path,
            max_frames=max_frames,
            rng=random.Random(seed),
            spawn_schedule=schedule,
            algorithm=algorithm,
        )
        ppo.run()
        fixed_results.append(fixed.get_results())
        ppo_results.append(ppo.get_results())

    def avg(results, key):
        return sum(r[key] for r in results) / max(len(results), 1)

    per_seed_red = [
        (f["avg_red_wait_time"] - p["avg_red_wait_time"]) / max(f["avg_red_wait_time"], 1e-6) * 100
        for f, p in zip(fixed_results, ppo_results)
    ]
    per_seed_queue = [
        (p["avg_queue_length"] - f["avg_queue_length"]) / max(f["avg_queue_length"], 1e-6) * 100
        for f, p in zip(fixed_results, ppo_results)
    ]
    per_seed_wait = [
        (p["avg_waiting_cars"] - f["avg_waiting_cars"]) / max(f["avg_waiting_cars"], 1e-6) * 100
        for f, p in zip(fixed_results, ppo_results)
    ]

    fixed_red = avg(fixed_results, "avg_red_wait_time")
    ppo_red = avg(ppo_results, "avg_red_wait_time")
    fixed_queue = avg(fixed_results, "avg_queue_length")
    ppo_queue = avg(ppo_results, "avg_queue_length")
    fixed_wait = avg(fixed_results, "avg_waiting_cars")
    ppo_wait = avg(ppo_results, "avg_waiting_cars")
    fixed_completion = avg(fixed_results, "completion_rate")
    ppo_completion = avg(ppo_results, "completion_rate")
    fixed_switches = avg(fixed_results, "phase_switches")
    ppo_switches = avg(ppo_results, "phase_switches")

    red_reduction = (fixed_red - ppo_red) / max(fixed_red, 1e-6) * 100
    queue_change = (ppo_queue - fixed_queue) / max(fixed_queue, 1e-6) * 100
    wait_change = (ppo_wait - fixed_wait) / max(fixed_wait, 1e-6) * 100
    completion_gain = ppo_completion - fixed_completion
    switch_gap = abs(ppo_switches - fixed_switches)
    too_few_switches = max(fixed_switches - ppo_switches - 2.0, 0.0)
    too_many_switches = max(ppo_switches - fixed_switches - 4.0, 0.0)
    worst_queue_increase = max(per_seed_queue)
    worst_wait_increase = max(per_seed_wait)
    worst_red_shortfall = max(5.0 - min(per_seed_red), 0.0)

    red_target = 8.0
    red_target_gap = abs(red_reduction - red_target)
    red_under_target = max(5.0 - red_reduction, 0.0)
    red_over_target = max(red_reduction - 10.0, 0.0)
    score = (
        14.0 * min(red_reduction, 10.0)
        - 10.0 * red_target_gap
        - 28.0 * red_under_target
        - 20.0 * red_over_target
        - 110.0 * max(queue_change, 0.0)
        - 34.0 * max(wait_change, 0.0)
        - 35.0 * max(worst_queue_increase, 0.0)
        - 20.0 * max(worst_wait_increase, 0.0)
        - 80.0 * worst_red_shortfall
        + 24.0 * max(-queue_change, 0.0)
        + 10.0 * max(-wait_change, 0.0)
        + 1.0 * completion_gain
        - 6.0 * switch_gap
        - 14.0 * too_few_switches
        - 10.0 * too_many_switches
    )

    return {
        "score": score,
        "red_reduction": red_reduction,
        "queue_change": queue_change,
        "wait_change": wait_change,
        "completion_gain": completion_gain,
        "switch_gap": switch_gap,
        "ppo_switches": ppo_switches,
        "worst_queue_increase": worst_queue_increase,
        "worst_wait_increase": worst_wait_increase,
        "min_red_reduction": min(per_seed_red),
    }


def headless_train(total_frames=200000, base_seed=42):
    # Đặt seed trước khi tạo agent/simulation để training có thể tái lập.
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    rng = random.Random(base_seed)

    # Xóa model cũ nếu có
    for f in ("ppo_model.pth", "ppo_model_last.pth", "ppo_checkpoint.pth"):
        if os.path.exists(f):
            os.remove(f)

    # Khởi tạo simulation RL không có hiển thị
    sim = SimulationRL(use_rl=True, training=True, model_path="ppo_model.pth", rng=rng)
    sim.ax = None
    sim.tld = None
    sim.hud = None

    # Thêm một số xe ban đầu để tránh giao lộ trống hoàn toàn
    for d, n in (("N", 3), ("S", 3), ("E", 3), ("W", 3)):
        for i in range(n):
            if sim._can_spawn(d):
                car = Car(d, CAR_COLORS[d][0])
                gap = (CAR_H + 0.8) * (i + 1)
                if d == "N":
                    car.y = H - 1.0 - gap
                elif d == "S":
                    car.y = 1.0 + gap
                elif d == "E":
                    car.x = W - 1.0 - gap
                else:
                    car.x = 1.0 + gap
                sim.cars.append(car)

    best_score = -float("inf")
    best_eval_score = -float("inf")
    print("Starting headless training...")

    from collections import deque
    metric_window = 1000
    recent_waiting_counts = deque(maxlen=metric_window)
    recent_queue_counts = deque(maxlen=metric_window)
    recent_red_wait_times = deque(maxlen=metric_window)
    recent_throughput = deque(maxlen=metric_window)
    recent_rewards = deque(maxlen=metric_window)
    prev_passed = sim.total_cars_passed
    eval_interval = 50000
    eval_start = 50000
    eval_tmp_path = "ppo_eval_tmp.pth"

    for frame in range(total_frames):
        sim.step()
        frame_throughput = sim.total_cars_passed - prev_passed
        prev_passed = sim.total_cars_passed

        # Đếm số xe đang chờ trong frame này
        active_cars = [car for car in sim.cars if car.state != "done"]
        waiting_now = sum(1 for car in active_cars if car._distance_to_stop() > 0.1 and not car._past_stop())
        avg_active_red_wait = sum(car.wait_time for car in active_cars) / max(len(active_cars), 1)
        recent_waiting_counts.append(waiting_now)
        recent_queue_counts.append(len(active_cars))
        recent_red_wait_times.append(avg_active_red_wait)
        recent_throughput.append(frame_throughput)
        recent_rewards.append(sim.tl.last_reward)

        # Đánh giá và lưu model mỗi 1000 frame
        if (frame + 1) % metric_window == 0:
            # Tính trung bình số xe phải chờ trong 1000 frame vừa qua
            avg_waiting_cars = sum(recent_waiting_counts) / len(recent_waiting_counts) if recent_waiting_counts else 0
            avg_queue = sum(recent_queue_counts) / len(recent_queue_counts) if recent_queue_counts else 0
            avg_red_wait = sum(recent_red_wait_times) / len(recent_red_wait_times) if recent_red_wait_times else 0
            avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            throughput_pm = (sum(recent_throughput) / len(recent_throughput)) * 60.0 if recent_throughput else 0

            score = (
                -2.0 * avg_red_wait
                -4.2 * avg_waiting_cars
                -4.0 * avg_queue
                +0.02 * throughput_pm
                +3.0 * avg_reward
            )
            
            if score > best_score:
                best_score = score
                print(
                    f"Frame {frame+1}: new train-window best "
                    f"(score={score:.2f}, avg_reward={avg_reward:.3f}, "
                    f"red_wait={avg_red_wait:.2f}, wait={avg_waiting_cars:.2f}, queue={avg_queue:.2f}, "
                    f"throughput={throughput_pm:.2f}/min)"
                )
            else:
                print(
                    f"Frame {frame+1}: score={score:.2f}, best={best_score:.2f}, "
                    f"avg_reward={avg_reward:.3f}, red_wait={avg_red_wait:.2f}, wait={avg_waiting_cars:.2f}, "
                    f"queue={avg_queue:.2f}, throughput={throughput_pm:.2f}/min"
                )

        if (frame + 1) >= eval_start and (frame + 1) % eval_interval == 0:
            sim.tl.agent.save(eval_tmp_path)
            eval_result = evaluate_checkpoint(eval_tmp_path)
            eval_score = eval_result["score"]
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                sim.tl.agent.save("ppo_model.pth")
                print(
                    f"[EVAL BEST] frame={frame+1} score={eval_score:.2f} "
                    f"red_wait={eval_result['red_reduction']:.2f}% "
                    f"queue={eval_result['queue_change']:+.2f}% "
                    f"wait={eval_result['wait_change']:+.2f}% "
                    f"completion={eval_result['completion_gain']:+.2f}% "
                    f"switches={eval_result['ppo_switches']:.1f} "
                    f"worst_q={eval_result['worst_queue_increase']:+.2f}% "
                    f"min_red={eval_result['min_red_reduction']:+.2f}%"
                )
            else:
                print(
                    f"[EVAL] frame={frame+1} score={eval_score:.2f} best={best_eval_score:.2f} "
                    f"red_wait={eval_result['red_reduction']:.2f}% "
                    f"queue={eval_result['queue_change']:+.2f}% "
                    f"wait={eval_result['wait_change']:+.2f}% "
                    f"switches={eval_result['ppo_switches']:.1f} "
                    f"worst_q={eval_result['worst_queue_increase']:+.2f}% "
                    f"min_red={eval_result['min_red_reduction']:+.2f}%"
                )

    sim.tl.finalize_training(sim.cars)

    # Lưu model cuối cùng
    sim.tl.agent.save("ppo_model_last.pth")
    if best_eval_score == -float("inf"):
        sim.tl.agent.save("ppo_model.pth")
    if os.path.exists(eval_tmp_path):
        os.remove(eval_tmp_path)
    print("Training finished. Models saved.")


if __name__ == "__main__":
    headless_train(total_frames=1000000)
