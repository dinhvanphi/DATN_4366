"""
traffic_simulation.py
---------------------
Chứa toàn bộ lớp mô phỏng và các hàm chạy benchmark/training/testing.
Đây là phần điều phối các module khác: thực thể, RL và phần hiển thị.
"""

import os
import random
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from traffic_constants import CAR_COLORS, CAR_H, H, SPAWN_BLOCK_DIST, W
from traffic_entities import Car, DemandController, HUD, TrafficLight, TrafficLightDisplay, draw_background
from traffic_rl import TrafficLightRL


BENCHMARK_MAX_FRAMES = 2000
BENCHMARK_SPAWN_BASE_PROB = {"N": 0.010, "S": 0.010, "E": 0.012, "W": 0.012}
BENCHMARK_SPAWN_BURST_PROB = 0.030
BENCHMARK_SPAWN_MAX_BURST = 3


def build_benchmark_spawn_schedule(seed, max_frames=BENCHMARK_MAX_FRAMES):
    """Tạo lịch yêu cầu spawn cố định để benchmark Fixed và PPO công bằng."""
    rng = random.Random(seed)
    demand = DemandController(rng=rng)
    schedule = []

    for _ in range(max_frames):
        demand.tick()
        frame_counts = {}
        for direction in ("N", "S", "E", "W"):
            p_base = demand.base_prob(direction, BENCHMARK_SPAWN_BASE_PROB)
            p_burst = demand.burst_prob(direction, BENCHMARK_SPAWN_BURST_PROB)
            count = 0
            if rng.random() < p_base:
                count += 1
            if rng.random() < p_burst:
                count += rng.randint(1, BENCHMARK_SPAWN_MAX_BURST)
            if count:
                frame_counts[direction] = min(count, BENCHMARK_SPAWN_MAX_BURST)
        schedule.append(frame_counts)

    return schedule


class Simulation:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.tl = TrafficLight()
        self.cars: list[Car] = []
        self.frame = 0
        self.spawn_base_prob = {"N": 0.010, "S": 0.010, "E": 0.012, "W": 0.012}
        self.spawn_burst_prob = 0.030
        self.spawn_max_burst = 3
        self.demand = DemandController(rng=self.rng)

    def _can_spawn(self, direction):
        for c in self.cars:
            if c.direction != direction or c.state == "done":
                continue
            if direction == "N" and c.y > H - SPAWN_BLOCK_DIST:
                return False
            if direction == "S" and c.y < SPAWN_BLOCK_DIST:
                return False
            if direction == "E" and c.x > W - SPAWN_BLOCK_DIST:
                return False
            if direction == "W" and c.x < SPAWN_BLOCK_DIST:
                return False
        return True

    def _sample_spawn_count(self, direction):
        p_base = self.demand.base_prob(direction, self.spawn_base_prob)
        p_burst = self.demand.burst_prob(direction, self.spawn_burst_prob)
        count = 0
        if self.rng.random() < p_base:
            count += 1
        if self.rng.random() < p_burst:
            count += self.rng.randint(1, self.spawn_max_burst)
        return min(count, self.spawn_max_burst)

    def _spawn_cars(self):
        self.demand.tick()
        for d in ("N", "S", "E", "W"):
            for _ in range(self._sample_spawn_count(d)):
                if not self._can_spawn(d):
                    break
                color = self.rng.choice(CAR_COLORS[d])
                car = Car(d, color)
                car.add_to_ax(self.ax)
                self.cars.append(car)

    def _cars_ahead(self, car: Car):
        ahead = []
        for other in self.cars:
            if other.id == car.id or other.direction != car.direction:
                continue
            if car.direction == "N" and other.y < car.y:
                ahead.append(other)
            elif car.direction == "S" and other.y > car.y:
                ahead.append(other)
            elif car.direction == "E" and other.x < car.x:
                ahead.append(other)
            elif car.direction == "W" and other.x > car.x:
                ahead.append(other)
        return ahead

    def _remove_done(self):
        to_remove = [c for c in self.cars if c.state == "done"]
        for c in to_remove:
            try:
                c.remove()
            except Exception:
                pass
        self.cars = [c for c in self.cars if c.state != "done"]

    def setup(self, fig, ax):
        self.ax = ax
        self.fig = fig
        draw_background(ax)
        self.tld = TrafficLightDisplay(ax)
        self.hud = HUD(ax)

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_facecolor("#111111")

    def step(self):
        self.frame += 1
        self.tl.tick()
        self._spawn_cars()

        for car in self.cars:
            if car.state == "done":
                continue
            is_green = self.tl.is_passable(car.direction)
            ahead = self._cars_ahead(car)
            car.update(is_green, ahead)

        self._remove_done()
        if hasattr(self, 'tld') and self.tld:
            self.tld.update(self.tl)
        if hasattr(self, 'hud') and self.hud:
            self.hud.update(self.tl, self.cars, self.frame)


class SimulationRL:
    """Simulation sử dụng RL để điều khiển đèn giao thông."""

    def __init__(self, use_rl=True, training=True, model_path="ppo_model.pth", rng=None, algorithm="ppo"):
        self.rng = rng or random.Random()
        self.use_rl = use_rl
        self.training = training
        self.model_path = model_path
        self.algorithm = algorithm.lower()

        if use_rl:
            self.tl = TrafficLightRL(use_rl=True, algorithm=self.algorithm)
            if not training:
                loaded = self.tl.agent.load(model_path)
                if not loaded:
                    print("Cảnh báo: Không load được model, chạy với trọng số khởi tạo ngẫu nhiên.")
        else:
            self.tl = TrafficLight()

        self.cars: list[Car] = []
        self.frame = 0
        self.spawn_base_prob = {"N": 0.010, "S": 0.010, "E": 0.012, "W": 0.012}
        self.spawn_burst_prob = 0.030
        self.spawn_max_burst = 3
        self.demand = DemandController(rng=self.rng)

        self.episode_reward = 0
        self.total_cars_passed = 0
        self.total_red_wait_time_passed = 0.0
        self.avg_wait_time = 0

    def _can_spawn(self, direction):
        for c in self.cars:
            if c.direction != direction or c.state == "done":
                continue
            if direction == "N" and c.y > H - SPAWN_BLOCK_DIST:
                return False
            if direction == "S" and c.y < SPAWN_BLOCK_DIST:
                return False
            if direction == "E" and c.x > W - SPAWN_BLOCK_DIST:
                return False
            if direction == "W" and c.x < SPAWN_BLOCK_DIST:
                return False
        return True

    def _sample_spawn_count(self, direction):
        p_base = self.demand.base_prob(direction, self.spawn_base_prob)
        p_burst = self.demand.burst_prob(direction, self.spawn_burst_prob)
        count = 0
        if self.rng.random() < p_base:
            count += 1
        if self.rng.random() < p_burst:
            count += self.rng.randint(1, self.spawn_max_burst)
        return min(count, self.spawn_max_burst)

    def _spawn_cars(self):
        self.demand.tick()
        for d in ("N", "S", "E", "W"):
            for _ in range(self._sample_spawn_count(d)):
                if not self._can_spawn(d):
                    break
                color = self.rng.choice(CAR_COLORS[d])
                car = Car(d, color)
                # Chỉ add_to_ax nếu có trục đồ họa (không headless)
                if hasattr(self, 'ax') and self.ax:
                    car.add_to_ax(self.ax)
                self.cars.append(car)

    def _cars_ahead(self, car: Car):
        ahead = []
        for other in self.cars:
            if other.id == car.id or other.direction != car.direction:
                continue
            if car.direction == "N" and other.y < car.y:
                ahead.append(other)
            elif car.direction == "S" and other.y > car.y:
                ahead.append(other)
            elif car.direction == "E" and other.x < car.x:
                ahead.append(other)
            elif car.direction == "W" and other.x > car.x:
                ahead.append(other)
        return ahead

    def _remove_done(self):
        to_remove = [c for c in self.cars if c.state == "done"]
        self.total_red_wait_time_passed += sum(c.wait_time for c in to_remove)
        for c in to_remove:
            try:
                c.remove()
            except Exception:
                pass
        self.cars = [c for c in self.cars if c.state != "done"]
        self.total_cars_passed += len(to_remove)
        return len(to_remove)

    def setup(self, fig, ax):
        self.ax = ax
        self.fig = fig
        draw_background(ax)
        self.tld = TrafficLightDisplay(ax)
        self.hud = HUD(ax)

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_facecolor("#111111")

    def step(self):
        self.frame += 1
        self._spawn_cars()

        for car in self.cars:
            if car.state == "done":
                continue
            is_green = self.tl.is_passable(car.direction)
            ahead = self._cars_ahead(car)
            car.update(is_green, ahead)

        throughput = self._remove_done()

        if self.use_rl:
            reward, action = self.tl.step(self.cars, throughput=throughput, training=self.training)
            self.episode_reward += reward
        else:
            self.tl.tick()

        # Cập nhật hiển thị nếu có đối tượng đồ họa
        if hasattr(self, 'tld') and self.tld:
            self.tld.update(self.tl)
        if hasattr(self, 'hud') and self.hud:
            self.hud.update(self.tl, self.cars, self.frame)
        return self.episode_reward

    def save_model(self):
        if self.use_rl:
            self.tl.agent.save(self.model_path)


class SimulationBenchmark:
    """Benchmark để so sánh Fixed Timing và RL trong cùng điều kiện spawn."""

    def __init__(self, use_rl=False, model_path="ppo_model.pth", max_frames=2000,
                 rng=None, spawn_schedule=None, algorithm="ppo"):
        self.rng = rng or random.Random()
        self.use_rl = use_rl
        self.algorithm = algorithm.lower()
        self.max_frames = max_frames
        self.spawn_schedule = spawn_schedule
        self.pending_spawns = {"N": 0, "S": 0, "E": 0, "W": 0}

        if use_rl:
            self.tl = TrafficLightRL(use_rl=True, algorithm=self.algorithm)
            loaded = self.tl.agent.load(model_path)
            if not loaded:
                raise RuntimeError(f"Không load được model từ {model_path}")
        else:
            self.tl = TrafficLight()

        self.cars: list[Car] = []
        self.frame = 0
        self.spawn_base_prob = {"N": 0.010, "S": 0.010, "E": 0.012, "W": 0.012}
        self.spawn_burst_prob = 0.030
        self.spawn_max_burst = 3
        self.demand = DemandController(rng=self.rng)

        self.total_cars_spawned = 0
        self.total_spawn_requested = 0
        self.total_cars_passed = 0
        self.total_red_wait_time_passed = 0.0
        self.total_wait_time = {"N": 0, "S": 0, "E": 0, "W": 0}
        self.total_queue_length = {"N": 0, "S": 0, "E": 0, "W": 0}
        self.total_internal_queue_length = {"N": 0, "S": 0, "E": 0, "W": 0}
        self.phase_switches = 0
        self.prev_phase = 0
        self.episode_reward = 0
        self.waiting_cars_history = []
        self.queue_length_history = []

    def _can_spawn(self, direction):
        for c in self.cars:
            if c.direction != direction or c.state == "done":
                continue
            if direction == "N" and c.y > H - SPAWN_BLOCK_DIST:
                return False
            if direction == "S" and c.y < SPAWN_BLOCK_DIST:
                return False
            if direction == "E" and c.x > W - SPAWN_BLOCK_DIST:
                return False
            if direction == "W" and c.x < SPAWN_BLOCK_DIST:
                return False
        return True

    def _sample_spawn_count(self, direction):
        p_base = self.demand.base_prob(direction, self.spawn_base_prob)
        p_burst = self.demand.burst_prob(direction, self.spawn_burst_prob)
        count = 0
        if self.rng.random() < p_base:
            count += 1
        if self.rng.random() < p_burst:
            count += self.rng.randint(1, self.spawn_max_burst)
        return min(count, self.spawn_max_burst)

    def _spawn_cars(self):
        if self.spawn_schedule is not None:
            self._spawn_from_schedule()
            return
        self.demand.tick()
        for d in ("N", "S", "E", "W"):
            for _ in range(self._sample_spawn_count(d)):
                if not self._can_spawn(d):
                    break
                color = self.rng.choice(CAR_COLORS[d])
                car = Car(d, color)
                self.cars.append(car)
                self.total_cars_spawned += 1

    def _spawn_from_schedule(self):
        frame_index = max(self.frame - 1, 0)
        if frame_index < len(self.spawn_schedule):
            frame_counts = self.spawn_schedule[frame_index]
            for d, count in frame_counts.items():
                self.pending_spawns[d] += count
                self.total_spawn_requested += count

        for d in ("N", "S", "E", "W"):
            while self.pending_spawns[d] > 0 and self._can_spawn(d):
                color = self.rng.choice(CAR_COLORS[d])
                car = Car(d, color)
                self.cars.append(car)
                self.total_cars_spawned += 1
                self.pending_spawns[d] -= 1

    def _cars_ahead(self, car: Car):
        ahead = []
        for other in self.cars:
            if other.id == car.id or other.direction != car.direction:
                continue
            if car.direction == "N" and other.y < car.y:
                ahead.append(other)
            elif car.direction == "S" and other.y > car.y:
                ahead.append(other)
            elif car.direction == "E" and other.x < car.x:
                ahead.append(other)
            elif car.direction == "W" and other.x > car.x:
                ahead.append(other)
        return ahead

    def _remove_done(self):
        to_remove = [c for c in self.cars if c.state == "done"]
        self.total_red_wait_time_passed += sum(c.wait_time for c in to_remove)
        for _ in to_remove:
            self.total_cars_passed += 1
        self.cars = [c for c in self.cars if c.state != "done"]
        return len(to_remove)

    def _calculate_metrics(self):
        queue = {"N": 0, "S": 0, "E": 0, "W": 0}
        waiting = {"N": 0, "S": 0, "E": 0, "W": 0}

        for car in self.cars:
            if car.state == "done":
                continue
            d = car.direction
            queue[d] += 1
            dist = car._distance_to_stop()
            if dist > 0.1 and not car._past_stop():
                waiting[d] += 1

        for d in ("N", "S", "E", "W"):
            pending = self.pending_spawns[d]
            self.total_internal_queue_length[d] += queue[d]
            self.total_queue_length[d] += queue[d] + pending
            self.total_wait_time[d] += waiting[d] + pending

        self.queue_length_history.append(sum(queue.values()) + sum(self.pending_spawns.values()))
        self.waiting_cars_history.append(sum(waiting.values()) + sum(self.pending_spawns.values()))

        current_phase = None
        if hasattr(self.tl, "phase"):
            current_phase = self.tl.phase
        elif hasattr(self.tl, "get"):
            # For Fixed Timing, deduce phase from NS green/yellow
            ns_state = self.tl.get("N")
            current_phase = 0 if ns_state in ("green", "yellow") else 1
            
        if current_phase is not None and self.prev_phase != current_phase:
            self.phase_switches += 1
            self.prev_phase = current_phase

    def run(self):
        for d, n in (("N", 3), ("S", 3), ("E", 3), ("W", 3)):
            for i in range(n):
                if self._can_spawn(d):
                    color = self.rng.choice(CAR_COLORS[d])
                    car = Car(d, color)
                    gap = (CAR_H + 0.8) * (i + 1)
                    if d == "N":
                        car.y = H - 1.0 - gap
                    elif d == "S":
                        car.y = 1.0 + gap
                    elif d == "E":
                        car.x = W - 1.0 - gap
                    else:
                        car.x = 1.0 + gap
                    self.cars.append(car)
                    self.total_cars_spawned += 1

        for _ in range(self.max_frames):
            self.frame += 1
            self._spawn_cars()

            for car in self.cars:
                if car.state == "done":
                    continue
                is_green = self.tl.is_passable(car.direction)
                ahead = self._cars_ahead(car)
                car.update(is_green, ahead)

            throughput = self._remove_done()

            if self.use_rl:
                reward, action = self.tl.step(self.cars, throughput=throughput, training=False)
                self.episode_reward += reward
            else:
                self.tl.tick()

            self._calculate_metrics()

    def get_results(self):
        frames = self.frame
        avg_queue = sum(self.total_queue_length.values()) / max(frames, 1)
        avg_internal_queue = sum(self.total_internal_queue_length.values()) / max(frames, 1)
        avg_waiting = sum(self.total_wait_time.values()) / max(frames, 1)
        max_queue = max(self.queue_length_history) if self.queue_length_history else 0
        max_waiting = max(self.waiting_cars_history) if self.waiting_cars_history else 0

        return {
            "algorithm": self.algorithm.upper() if self.use_rl else "Fixed Timing",
            "frames": frames,
            "cars_requested": self.total_spawn_requested,
            "cars_spawned": self.total_cars_spawned,
            "cars_passed": self.total_cars_passed,
            "pending_spawns": sum(self.pending_spawns.values()),
            "remaining_cars": len(self.cars) + sum(self.pending_spawns.values()),
            "completion_rate": self.total_cars_passed / max(self.total_cars_spawned, 1) * 100,
            "avg_red_wait_time": self.total_red_wait_time_passed / max(self.total_cars_passed, 1),
            "throughput_rate": self.total_cars_passed / max(frames, 1) * 60,
            "avg_queue_length": avg_queue,
            "avg_internal_queue_length": avg_internal_queue,
            "max_queue": max_queue,
            "avg_waiting_cars": avg_waiting,
            "max_waiting": max_waiting,
            "phase_switches": self.phase_switches,
            "avg_phase_duration": frames / max(self.phase_switches, 1),
            "total_reward": self.episode_reward if self.use_rl else None,
        }


def main_fixed_timing():
    """Chạy simulation với đèn giao thông có chu kỳ cố định."""
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    sim = Simulation()
    sim.setup(fig, ax)

    for d, n in (("N", 3), ("S", 3), ("E", 3), ("W", 3)):
        for i in range(n):
            if sim._can_spawn(d):
                color = sim.rng.choice(CAR_COLORS[d])
                car = Car(d, color)
                gap = (CAR_H + 0.8) * (i + 1)
                if d == "N":
                    car.y = H - 1.0 - gap
                elif d == "S":
                    car.y = 1.0 + gap
                elif d == "E":
                    car.x = W - 1.0 - gap
                else:
                    car.x = 1.0 + gap
                car._update_patch()
                car.add_to_ax(ax)
                sim.cars.append(car)

    ax.set_title("Mô phỏng giao thông — Fixed Timing", color="white", fontsize=12, fontweight="bold", pad=4)
    fig.patch.set_facecolor("#111111")

    def animate(frame):
        sim.step()
        return []

    ani = FuncAnimation(fig, animate, interval=40, blit=False, cache_frame_data=False)
    plt.show()


def main_dqn_train():
    """Chạy simulation với PPO training."""
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    best_model_path = "ppo_model.pth"
    checkpoint_path = "ppo_checkpoint.pth"
    last_model_path = "ppo_model_last.pth"

    sim = SimulationRL(use_rl=True, training=True, model_path=checkpoint_path)
    sim.setup(fig, ax)

    score_window = 400
    eval_interval = 500
    warmup_frames = 3000
    recent_queue = deque(maxlen=score_window)
    recent_waiting = deque(maxlen=score_window)
    recent_throughput = deque(maxlen=score_window)
    prev_passed = 0
    best_score = -float("inf")
    best_frame = -1
    best_saved = False
    switch_counts = {"AGENT": 0, "SOFT_FORCE": 0, "HARD_FORCE": 0, "FORCE": 0}
    total_switches = 0
    last_switch_frame = 0
    avg_phase_len = 0.0

    for d, n in (("N", 3), ("S", 3), ("E", 3), ("W", 3)):
        for i in range(n):
            if sim._can_spawn(d):
                color = sim.rng.choice(CAR_COLORS[d])
                car = Car(d, color)
                gap = (CAR_H + 0.8) * (i + 1)
                if d == "N":
                    car.y = H - 1.0 - gap
                elif d == "S":
                    car.y = 1.0 + gap
                elif d == "E":
                    car.x = W - 1.0 - gap
                else:
                    car.x = 1.0 + gap
                car._update_patch()
                car.add_to_ax(ax)
                sim.cars.append(car)

    ax.set_title("Mô phỏng giao thông — PPO Training", color="white", fontsize=12, fontweight="bold", pad=4)
    fig.patch.set_facecolor("#111111")

    rl_text = ax.text(
        0.01,
        0.85,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="cyan",
        fontfamily="sans-serif",
        bbox=dict(facecolor="#000000aa", edgecolor="#00ffff22", boxstyle="round,pad=0.5"),
        zorder=20,
    )

    def animate(frame):
        nonlocal prev_passed, best_score, best_frame, best_saved
        nonlocal total_switches, last_switch_frame, avg_phase_len

        sim.step()
        reward = sim.tl.last_reward
        frame_throughput = sim.total_cars_passed - prev_passed
        prev_passed = sim.total_cars_passed

        waiting_now = 0
        queue_now = 0
        for car in sim.cars:
            if car.state == "done":
                continue
            queue_now += 1
            dist = car._distance_to_stop()
            if dist > 0.1 and not car._past_stop():
                waiting_now += 1

        recent_throughput.append(frame_throughput)
        recent_waiting.append(waiting_now)
        recent_queue.append(queue_now)

        action_str = "N/A" if sim.tl.last_action is None else ("KEEP" if sim.tl.last_action == 0 else "SWITCH")
        if sim.tl.last_switch_reason in switch_counts:
            switch_counts[sim.tl.last_switch_reason] += 1
            total_switches += 1
            phase_len = frame - last_switch_frame
            last_switch_frame = frame
            avg_phase_len = ((avg_phase_len * (total_switches - 1)) + phase_len) / total_switches

        current_score = float("nan")
        if len(recent_throughput) >= score_window:
            throughput_pm = (sum(recent_throughput) / len(recent_throughput)) * 60.0
            avg_wait = sum(recent_waiting) / len(recent_waiting)
            avg_queue = sum(recent_queue) / len(recent_queue)
            current_score = throughput_pm - 1.8 * avg_wait - 1.0 * avg_queue

            should_eval = frame >= warmup_frames and frame % eval_interval == 0
            if should_eval and current_score > best_score:
                best_score = current_score
                best_frame = frame
                sim.tl.agent.save(best_model_path)
                best_saved = True
                print(
                    f"[BEST] frame={frame} score={best_score:.2f} "
                    f"throughput={throughput_pm:.2f}/min wait={avg_wait:.2f} queue={avg_queue:.2f}"
                )

        score_text = f"{current_score:.2f}" if not np.isnan(current_score) else "warming"
        best_text = f"{best_score:.2f}@{best_frame}" if best_frame >= 0 else "N/A"
        agent_pct = (switch_counts["AGENT"] / total_switches * 100.0) if total_switches > 0 else 0.0
        rl_text.set_text(
            f"PPO Training\n"
            f"Reward: {reward:.1f}\n"
            f"Passed: {sim.total_cars_passed}\n"
            f"Phase: {'NS' if sim.tl.phase == 0 else 'EW'}\n"
            f"Time: {sim.tl.time_in_phase}/{sim.tl.MAX_GREEN_TIME} (hard {sim.tl.HARD_MAX_GREEN_TIME})\n"
            f"Action: {action_str}\n"
            f"Switch: {sim.tl.last_switch_reason}\n"
            f"Switches: {total_switches} (agent {agent_pct:.1f}%)\n"
            f"Avg phase: {avg_phase_len:.1f} fr\n"
            f"Score: {score_text}\n"
            f"Best : {best_text}"
        )

        if frame > 0 and frame % 1000 == 0:
            sim.save_model()
            print(f"Saved checkpoint at frame {frame}")

        return []

    ani = FuncAnimation(fig, animate, interval=40, blit=False, cache_frame_data=False)
    plt.show()

    sim.tl.agent.save(last_model_path)
    if not best_saved:
        sim.tl.agent.save(best_model_path)
        print("Training completed. No best checkpoint during eval; saved final as best model.")
    else:
        print(f"Training completed. Best model kept at {best_model_path} (frame={best_frame}, score={best_score:.2f}).")
    print(f"Final snapshot saved to {last_model_path}")


def main_dqn_test(model_path="ppo_model.pth", algorithm="ppo"):
    """Chạy simulation với model RL đã train."""
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    sim = SimulationRL(use_rl=True, training=False, model_path=model_path, algorithm=algorithm)
    sim.setup(fig, ax)

    for d, n in (("N", 3), ("S", 3), ("E", 3), ("W", 3)):
        for i in range(n):
            if sim._can_spawn(d):
                color = sim.rng.choice(CAR_COLORS[d])
                car = Car(d, color)
                gap = (CAR_H + 0.8) * (i + 1)
                if d == "N":
                    car.y = H - 1.0 - gap
                elif d == "S":
                    car.y = 1.0 + gap
                elif d == "E":
                    car.x = W - 1.0 - gap
                else:
                    car.x = 1.0 + gap
                car._update_patch()
                car.add_to_ax(ax)
                sim.cars.append(car)

    model_name = os.path.basename(model_path)
    ax.set_title(f"Mô phỏng giao thông — PPO ({model_name})", color="white", fontsize=12, fontweight="bold", pad=4)
    fig.patch.set_facecolor("#111111")

    rl_text = ax.text(
        0.01,
        0.85,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="cyan",
        fontfamily="sans-serif",
        bbox=dict(facecolor="#000000aa", edgecolor="#00ffff22", boxstyle="round,pad=0.5"),
        zorder=20,
    )
    switch_counts = {"AGENT": 0, "SOFT_FORCE": 0, "HARD_FORCE": 0, "FORCE": 0}
    total_switches = 0
    last_switch_frame = 0
    avg_phase_len = 0.0

    def animate(frame):
        nonlocal total_switches, last_switch_frame, avg_phase_len
        sim.step()
        reward = sim.tl.last_reward
        action_str = "N/A" if sim.tl.last_action is None else ("KEEP" if sim.tl.last_action == 0 else "SWITCH")
        if sim.tl.last_switch_reason in switch_counts:
            switch_counts[sim.tl.last_switch_reason] += 1
            total_switches += 1
            phase_len = frame - last_switch_frame
            last_switch_frame = frame
            avg_phase_len = ((avg_phase_len * (total_switches - 1)) + phase_len) / total_switches
        agent_pct = (switch_counts["AGENT"] / total_switches * 100.0) if total_switches > 0 else 0.0
        rl_text.set_text(
            f"PPO Trained\n"
            f"Reward: {reward:.1f}\n"
            f"Passed: {sim.total_cars_passed}\n"
            f"Phase: {'NS' if sim.tl.phase == 0 else 'EW'}\n"
            f"Time: {sim.tl.time_in_phase}/{sim.tl.MAX_GREEN_TIME} (hard {sim.tl.HARD_MAX_GREEN_TIME})\n"
            f"Action: {action_str}\n"
            f"Switch: {sim.tl.last_switch_reason}\n"
            f"Switches: {total_switches} (agent {agent_pct:.1f}%)\n"
            f"Avg phase: {avg_phase_len:.1f} fr"
        )
        return []

    ani = FuncAnimation(fig, animate, interval=40, blit=False, cache_frame_data=False)
    plt.show()


def run_comparison(model_path="ppo_model.pth", dqn_label="PPO", seed=42, algorithm="ppo"):
    """Chạy so sánh Fixed Timing và agent RL."""
    print("\n" + "=" * 70)
    print(f"SO SÁNH HIỆU QUẢ: FIXED TIMING vs {dqn_label}")
    print("=" * 70)

    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file {model_path}")
        print("Vui lòng chạy training trước (option 2 hoặc 4)")
        return None

    max_frames = BENCHMARK_MAX_FRAMES

    print(f"\nChạy benchmark với {max_frames} frames...")
    print("-" * 50)

    spawn_schedule = build_benchmark_spawn_schedule(seed, max_frames)

    print("1. Đang chạy Fixed Timing...")
    rng_fixed = random.Random(seed)
    sim_fixed = SimulationBenchmark(
        use_rl=False,
        max_frames=max_frames,
        rng=rng_fixed,
        spawn_schedule=spawn_schedule,
    )
    sim_fixed.run()
    results_fixed = sim_fixed.get_results()
    print("   ✓ Hoàn thành")

    print(f"2. Đang chạy {dqn_label} ({model_path})...")
    rng_dqn = random.Random(seed)
    sim_dqn = SimulationBenchmark(
        use_rl=True,
        model_path=model_path,
        max_frames=max_frames,
        rng=rng_dqn,
        spawn_schedule=spawn_schedule,
        algorithm=algorithm,
    )
    sim_dqn.run()
    results_dqn = sim_dqn.get_results()
    print("   ✓ Hoàn thành")

    print("\n" + "=" * 70)
    print("KẾT QUẢ SO SÁNH")
    print("=" * 70)
    print(f"{'Metric':<30} {'Fixed Timing':>18} {dqn_label:>18}")
    print("-" * 70)
    print(f"{'Tổng yêu cầu spawn':<30} {results_fixed['cars_requested']:>18} {results_dqn['cars_requested']:>18}")
    print(f"{'Tổng xe spawn':<30} {results_fixed['cars_spawned']:>18} {results_dqn['cars_spawned']:>18}")
    print(f"{'Tổng xe đi qua':<30} {results_fixed['cars_passed']:>18} {results_dqn['cars_passed']:>18}")
    print(f"{'Xe còn chờ spawn':<30} {results_fixed['pending_spawns']:>18} {results_dqn['pending_spawns']:>18}")
    print(f"{'Xe còn trong hệ thống':<30} {results_fixed['remaining_cars']:>18} {results_dqn['remaining_cars']:>18}")
    print(f"{'Tỷ lệ hoàn tất (%)':<30} {results_fixed['completion_rate']:>18.2f} {results_dqn['completion_rate']:>18.2f}")
    print(f"{'Thời gian chờ đỏ TB':<30} {results_fixed['avg_red_wait_time']:>18.2f} {results_dqn['avg_red_wait_time']:>18.2f}")
    print(f"{'Throughput (xe/phút)':<30} {results_fixed['throughput_rate']:>18.2f} {results_dqn['throughput_rate']:>18.2f}")
    print(f"{'Độ dài hàng đợi TB':<30} {results_fixed['avg_queue_length']:>18.2f} {results_dqn['avg_queue_length']:>18.2f}")
    print(f"{'Hàng đợi nội bộ TB':<30} {results_fixed['avg_internal_queue_length']:>18.2f} {results_dqn['avg_internal_queue_length']:>18.2f}")
    print(f"{'Độ dài hàng đợi max':<30} {results_fixed['max_queue']:>18} {results_dqn['max_queue']:>18}")
    print(f"{'Xe đang chờ TB':<30} {results_fixed['avg_waiting_cars']:>18.2f} {results_dqn['avg_waiting_cars']:>18.2f}")
    print(f"{'Số lần chuyển phase':<30} {results_fixed['phase_switches']:>18} {results_dqn['phase_switches']:>18}")
    print(f"{'Thời gian phase TB':<30} {results_fixed['avg_phase_duration']:>18.1f} {results_dqn['avg_phase_duration']:>18.1f}")
    if results_dqn['total_reward'] is not None:
        print(f"{'Tổng reward':<30} {'N/A':>18} {results_dqn['total_reward']:>18.1f}")
    print("=" * 70)

    print("\nPHÂN TÍCH:")
    print("-" * 50)

    throughput_diff = results_dqn['throughput_rate'] - results_fixed['throughput_rate']
    throughput_pct = (throughput_diff / max(results_fixed['throughput_rate'], 0.001)) * 100
    print(f"• Throughput: {dqn_label} {'+' if throughput_diff > 0 else ''}{throughput_pct:.1f}% so với Fixed Timing")

    queue_diff = results_fixed['avg_queue_length'] - results_dqn['avg_queue_length']
    queue_pct = (queue_diff / max(results_fixed['avg_queue_length'], 0.001)) * 100
    print(f"• Hàng đợi TB: {dqn_label} {'giảm' if queue_diff > 0 else 'tăng'} {abs(queue_pct):.1f}%")

    wait_diff = results_fixed['avg_waiting_cars'] - results_dqn['avg_waiting_cars']
    wait_pct = (wait_diff / max(results_fixed['avg_waiting_cars'], 0.001)) * 100
    print(f"• Xe chờ TB: {dqn_label} {'giảm' if wait_diff > 0 else 'tăng'} {abs(wait_pct):.1f}%")

    red_wait_diff = results_fixed['avg_red_wait_time'] - results_dqn['avg_red_wait_time']
    red_wait_pct = (red_wait_diff / max(results_fixed['avg_red_wait_time'], 0.001)) * 100
    print(f"• Thời gian chờ đỏ TB: {dqn_label} {'giảm' if red_wait_diff > 0 else 'tăng'} {abs(red_wait_pct):.1f}%")

    print(f"• Số lần chuyển phase: Fixed={results_fixed['phase_switches']}, {dqn_label}={results_dqn['phase_switches']}")
    if results_dqn['phase_switches'] > results_fixed['phase_switches']:
        print(f"  → {dqn_label} linh hoạt hơn, chuyển phase nhiều hơn")
    else:
        print(f"  → {dqn_label} ổn định hơn, ít chuyển phase hơn")

    print("\n" + "=" * 70)
    return results_fixed, results_dqn


def run_comparison_multi_seed(model_path="ppo_model.pth", dqn_label="PPO", seeds=None, algorithm="ppo"):
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    fixed_results = []
    dqn_results = []

    print("\n" + "=" * 70)
    print("SO SÁNH HIỆU QUẢ (NHIỀU SEED)")
    print("=" * 70)
    print(f"Seeds: {', '.join(str(s) for s in seeds)}")

    for seed in seeds:
        print("\n" + "-" * 70)
        print(f"Seed {seed}")
        print("-" * 70)
        result_pair = run_comparison(
            model_path=model_path,
            dqn_label=dqn_label,
            seed=seed,
            algorithm=algorithm,
        )
        if result_pair is None:
            return

        results_fixed, results_dqn = result_pair
        fixed_results.append(results_fixed)
        dqn_results.append(results_dqn)

    def mean(values):
        return sum(values) / max(len(values), 1)

    def summarize(results):
        return {
            "throughput_rate": mean([r["throughput_rate"] for r in results]),
            "avg_queue_length": mean([r["avg_queue_length"] for r in results]),
            "avg_internal_queue_length": mean([r["avg_internal_queue_length"] for r in results]),
            "avg_waiting_cars": mean([r["avg_waiting_cars"] for r in results]),
            "cars_requested": mean([r["cars_requested"] for r in results]),
            "cars_spawned": mean([r["cars_spawned"] for r in results]),
            "cars_passed": mean([r["cars_passed"] for r in results]),
            "pending_spawns": mean([r["pending_spawns"] for r in results]),
            "remaining_cars": mean([r["remaining_cars"] for r in results]),
            "completion_rate": mean([r["completion_rate"] for r in results]),
            "avg_red_wait_time": mean([r["avg_red_wait_time"] for r in results]),
        }

    fixed_mean = summarize(fixed_results)
    dqn_mean = summarize(dqn_results)

    print("\n" + "=" * 70)
    print("TRUNG BÌNH NHIỀU SEED")
    print("=" * 70)
    print(f"{'Metric':<30} {'Fixed Timing':>18} {dqn_label:>18}")
    print("-" * 70)
    print(f"{'Tổng yêu cầu spawn':<30} {fixed_mean['cars_requested']:>18.2f} {dqn_mean['cars_requested']:>18.2f}")
    print(f"{'Tổng xe spawn':<30} {fixed_mean['cars_spawned']:>18.2f} {dqn_mean['cars_spawned']:>18.2f}")
    print(f"{'Tổng xe đi qua':<30} {fixed_mean['cars_passed']:>18.2f} {dqn_mean['cars_passed']:>18.2f}")
    print(f"{'Xe còn chờ spawn':<30} {fixed_mean['pending_spawns']:>18.2f} {dqn_mean['pending_spawns']:>18.2f}")
    print(f"{'Xe còn trong hệ thống':<30} {fixed_mean['remaining_cars']:>18.2f} {dqn_mean['remaining_cars']:>18.2f}")
    print(f"{'Tỷ lệ hoàn tất (%)':<30} {fixed_mean['completion_rate']:>18.2f} {dqn_mean['completion_rate']:>18.2f}")
    print(f"{'Thời gian chờ đỏ TB':<30} {fixed_mean['avg_red_wait_time']:>18.2f} {dqn_mean['avg_red_wait_time']:>18.2f}")
    print(f"{'Throughput (xe/phút)':<30} {fixed_mean['throughput_rate']:>18.2f} {dqn_mean['throughput_rate']:>18.2f}")
    print(f"{'Độ dài hàng đợi TB':<30} {fixed_mean['avg_queue_length']:>18.2f} {dqn_mean['avg_queue_length']:>18.2f}")
    print(f"{'Hàng đợi nội bộ TB':<30} {fixed_mean['avg_internal_queue_length']:>18.2f} {dqn_mean['avg_internal_queue_length']:>18.2f}")
    print(f"{'Xe đang chờ TB':<30} {fixed_mean['avg_waiting_cars']:>18.2f} {dqn_mean['avg_waiting_cars']:>18.2f}")
    print("=" * 70)


def main():
    """Main function - chọn chế độ chạy."""
    print("=" * 50)
    print("Traffic Simulation with Proximal Policy Optimization (PPO)")
    print("=" * 50)
    print("1. Fixed Timing (đèn chu kỳ cố định)")
    print("2. PPO Training (huấn luyện PPO)")
    print("3. PPO Testing (chạy với model đã train)")
    print("4. PPO Fresh Training (xóa model cũ, train từ đầu)")
    print("5. Compare Benchmark (so sánh Fixed vs PPO)")
    print("6. PPO Testing (ppo_model_last.pth)")
    print("7. Compare Benchmark (Fixed vs ppo_model_last)")
    print("8. Compare Benchmark (nhiều seed)")
    print("9. DQN Testing (dqn_model.pth)")
    print("10. Compare Benchmark (Fixed vs DQN)")
    print("11. Compare Benchmark (DQN nhiều seed)")
    print("=" * 50)

    choice = input("Chọn chế độ (1-11): ").strip()

    if choice == "1":
        main_fixed_timing()
    elif choice == "2":
        main_dqn_train()
    elif choice == "3":
        main_dqn_test(model_path="ppo_model.pth")
    elif choice == "4":
        removed_any = False
        for stale_file in ("ppo_model.pth", "ppo_model_last.pth", "ppo_checkpoint.pth"):
            if os.path.exists(stale_file):
                os.remove(stale_file)
                removed_any = True
        if removed_any:
            print("Đã xóa model/checkpoint cũ. Bắt đầu training từ đầu...")
        main_dqn_train()
    elif choice == "5":
        run_comparison(model_path="ppo_model.pth", dqn_label="PPO(best)")
    elif choice == "6":
        main_dqn_test(model_path="ppo_model_last.pth")
    elif choice == "7":
        run_comparison(model_path="ppo_model_last.pth", dqn_label="PPO(last)")
    elif choice == "8":
        run_comparison_multi_seed(model_path="ppo_model.pth", dqn_label="PPO(best)")
    elif choice == "9":
        main_dqn_test(model_path="dqn_model.pth", algorithm="dqn")
    elif choice == "10":
        run_comparison(model_path="dqn_model.pth", dqn_label="DQN", algorithm="dqn")
    elif choice == "11":
        run_comparison_multi_seed(model_path="dqn_model.pth", dqn_label="DQN", algorithm="dqn")
    else:
        print("Lựa chọn không hợp lệ. Chạy chế độ PPO Training mặc định.")
        main_dqn_train()
