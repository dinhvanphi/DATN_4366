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
from traffic_simulation import SimulationRL
from traffic_entities import Car
from traffic_constants import CAR_COLORS, CAR_H, H, W


def headless_train(total_frames=200000, base_seed=42):
    # Xóa model cũ nếu có
    for f in ("ppo_model.pth", "ppo_model_last.pth", "ppo_checkpoint.pth"):
        if os.path.exists(f):
            os.remove(f)

    # Khởi tạo simulation RL không có hiển thị
    sim = SimulationRL(use_rl=True, training=True, model_path="ppo_model.pth")
    sim.ax = None
    sim.tld = None
    sim.hud = None

    # Đặt seed
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

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
    print("Starting headless training...")

    from collections import deque
    recent_waiting_counts = deque(maxlen=1000)

    for frame in range(total_frames):
        sim.step()

        # Đếm số xe đang chờ trong frame này
        waiting_now = sum(1 for car in sim.cars if car._distance_to_stop() > 0.1 and not car._past_stop())
        recent_waiting_counts.append(waiting_now)

        # Đánh giá và lưu model mỗi 1000 frame
        if (frame + 1) % 1000 == 0:
            # Tính trung bình số xe phải chờ trong 1000 frame vừa qua
            avg_waiting_cars = sum(recent_waiting_counts) / len(recent_waiting_counts) if recent_waiting_counts else 0
            
            # Điểm càng cao càng tốt (âm ít hơn)
            score = -avg_waiting_cars
            
            if score > best_score:
                best_score = score
                sim.tl.agent.save("ppo_model.pth")
                print(f"Frame {frame+1}: new best model saved (avg_waiting_cars={avg_waiting_cars:.2f}, score={score:.2f})")
            else:
                print(f"Frame {frame+1}: avg_waiting_cars={avg_waiting_cars:.2f}, best={best_score:.2f}")

    # Lưu model cuối cùng
    sim.tl.agent.save("ppo_model_last.pth")
    print("Training finished. Models saved.")


if __name__ == "__main__":
    headless_train(total_frames=1000000)