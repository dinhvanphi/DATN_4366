"""
train_dqn_headless.py
---------------------
Huấn luyện DQN trong cùng môi trường, state, reward, timing và benchmark seeds
đang dùng cho PPO.
"""

import argparse
import os
import random
from collections import deque

import numpy as np
import torch

from traffic_constants import CAR_COLORS, CAR_H, H, W
from traffic_entities import Car
from traffic_simulation import SimulationRL
from train_headless import evaluate_checkpoint


def headless_train_dqn(total_frames=1_000_000, base_seed=42):
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    rng = random.Random(base_seed)

    for path in ("dqn_model.pth", "dqn_model_last.pth", "dqn_eval_tmp.pth"):
        if os.path.exists(path):
            os.remove(path)

    sim = SimulationRL(
        use_rl=True,
        training=True,
        model_path="dqn_model.pth",
        rng=rng,
        algorithm="dqn",
    )
    sim.ax = None
    sim.tld = None
    sim.hud = None

    # Cùng điều kiện khởi tạo xe ban đầu như PPO headless training.
    for direction, count in (("N", 3), ("S", 3), ("E", 3), ("W", 3)):
        for i in range(count):
            if sim._can_spawn(direction):
                car = Car(direction, CAR_COLORS[direction][0])
                gap = (CAR_H + 0.8) * (i + 1)
                if direction == "N":
                    car.y = H - 1.0 - gap
                elif direction == "S":
                    car.y = 1.0 + gap
                elif direction == "E":
                    car.x = W - 1.0 - gap
                else:
                    car.x = 1.0 + gap
                sim.cars.append(car)

    print("Starting DQN headless training...")
    metric_window = 1000
    recent_waiting_counts = deque(maxlen=metric_window)
    recent_queue_counts = deque(maxlen=metric_window)
    recent_red_wait_times = deque(maxlen=metric_window)
    recent_throughput = deque(maxlen=metric_window)
    recent_rewards = deque(maxlen=metric_window)
    prev_passed = sim.total_cars_passed

    best_score = -float("inf")
    best_eval_score = -float("inf")
    eval_interval = 50000
    eval_start = 50000
    eval_tmp_path = "dqn_eval_tmp.pth"

    for frame in range(total_frames):
        sim.step()
        frame_throughput = sim.total_cars_passed - prev_passed
        prev_passed = sim.total_cars_passed

        active_cars = [car for car in sim.cars if car.state != "done"]
        waiting_now = sum(
            1 for car in active_cars
            if car._distance_to_stop() > 0.1 and not car._past_stop()
        )
        avg_active_red_wait = sum(car.wait_time for car in active_cars) / max(len(active_cars), 1)

        recent_waiting_counts.append(waiting_now)
        recent_queue_counts.append(len(active_cars))
        recent_red_wait_times.append(avg_active_red_wait)
        recent_throughput.append(frame_throughput)
        recent_rewards.append(sim.tl.last_reward)

        if (frame + 1) % metric_window == 0:
            avg_waiting_cars = sum(recent_waiting_counts) / len(recent_waiting_counts)
            avg_queue = sum(recent_queue_counts) / len(recent_queue_counts)
            avg_red_wait = sum(recent_red_wait_times) / len(recent_red_wait_times)
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            throughput_pm = (sum(recent_throughput) / len(recent_throughput)) * 60.0

            score = (
                -2.0 * avg_red_wait
                -4.2 * avg_waiting_cars
                -4.0 * avg_queue
                + 0.02 * throughput_pm
                + 3.0 * avg_reward
            )

            if score > best_score:
                best_score = score
                print(
                    f"Frame {frame+1}: new train-window best "
                    f"(score={score:.2f}, avg_reward={avg_reward:.3f}, "
                    f"red_wait={avg_red_wait:.2f}, wait={avg_waiting_cars:.2f}, "
                    f"queue={avg_queue:.2f}, throughput={throughput_pm:.2f}/min)"
                )
            else:
                print(
                    f"Frame {frame+1}: score={score:.2f}, best={best_score:.2f}, "
                    f"avg_reward={avg_reward:.3f}, red_wait={avg_red_wait:.2f}, "
                    f"wait={avg_waiting_cars:.2f}, queue={avg_queue:.2f}, "
                    f"throughput={throughput_pm:.2f}/min"
                )

        if (frame + 1) >= eval_start and (frame + 1) % eval_interval == 0:
            sim.tl.agent.save(eval_tmp_path)
            eval_result = evaluate_checkpoint(eval_tmp_path, algorithm="dqn")
            eval_score = eval_result["score"]
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                sim.tl.agent.save("dqn_model.pth")
                print(
                    f"[DQN EVAL BEST] frame={frame+1} score={eval_score:.2f} "
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
                    f"[DQN EVAL] frame={frame+1} score={eval_score:.2f} "
                    f"best={best_eval_score:.2f} "
                    f"red_wait={eval_result['red_reduction']:.2f}% "
                    f"queue={eval_result['queue_change']:+.2f}% "
                    f"wait={eval_result['wait_change']:+.2f}% "
                    f"switches={eval_result['ppo_switches']:.1f} "
                    f"worst_q={eval_result['worst_queue_increase']:+.2f}% "
                    f"min_red={eval_result['min_red_reduction']:+.2f}%"
                )

    sim.tl.finalize_training(sim.cars)
    sim.tl.agent.save("dqn_model_last.pth")
    if best_eval_score == -float("inf"):
        sim.tl.agent.save("dqn_model.pth")
    if os.path.exists(eval_tmp_path):
        os.remove(eval_tmp_path)
    print("DQN training finished. Models saved.")


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện DQN headless cùng điều kiện với PPO.")
    parser.add_argument("--frames", type=int, default=1_000_000, help="Số frame train.")
    parser.add_argument("--seed", type=int, default=42, help="Seed training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    headless_train_dqn(total_frames=args.frames, base_seed=args.seed)
