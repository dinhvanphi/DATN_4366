"""
traffic_rl.py
-------------
Điều khiển đèn giao thông bằng PPO — quyết định pha tiếp theo + thời lượng xanh.

Key fixes:
- Environment reset giữa các episode → reward stationary
- Reward phạt trực tiếp xe chờ ở pha đỏ + thưởng throughput
- Average reward per frame (đúng cho semi-MDP)
- Hệ số đủ mạnh để agent phân biệt được action tốt/xấu
"""

import random
import numpy as np

from traffic_ppo import PPOAgent

GREEN_DURATIONS = [60, 80, 100, 120, 140, 160, 180]
PHASE_CHOICES = 2


class TrafficLightRL:
    """Đèn giao thông PPO — chọn pha tiếp theo và thời lượng xanh."""

    STATE_SIZE = 19
    ACTION_SIZE = PHASE_CHOICES * len(GREEN_DURATIONS)

    PHASE_NS_GREEN = 0
    PHASE_EW_GREEN = 1

    YELLOW_TIME = 20
    ALL_RED_TIME = 12
    EPISODE_LENGTH = 2000

    MAX_GREEN_TIME = max(GREEN_DURATIONS)
    HARD_MAX_GREEN_TIME = max(GREEN_DURATIONS)

    def __init__(self, use_rl=True):
        self.use_rl = use_rl
        self.phase = self.PHASE_NS_GREEN
        self.time_in_phase = 0
        self.frame = 0
        self.pending_phase = None
        self.in_yellow = False
        self.in_all_red = False
        self.transition_timer = 0
        self.last_switch_reason = "INIT"
        self.last_switch_frame = -9999
        self.last_action = None

        self.green_remaining = 0
        self.chosen_green = 120
        self.needs_decision = True

        if use_rl:
            self.agent = PPOAgent(
                state_size=self.STATE_SIZE,
                action_size=self.ACTION_SIZE,
            )
            self.last_state = None
            self.last_log_prob = 0.0
            self.last_value = 0.0
            self.accumulated_reward = 0.0
            self.frames_since_last_decision = 0
            self.episode_frame = 0
            self.episode_just_ended = False

    def get_state(self, cars):
        queue = {"N": 0, "S": 0, "E": 0, "W": 0}
        waiting_time = {"N": 0, "S": 0, "E": 0, "W": 0}
        waiting_counts = {"N": 0, "S": 0, "E": 0, "W": 0}

        for car in cars:
            if car.state == "done":
                continue
            d = car.direction
            queue[d] += 1
            if not car._past_stop():
                waiting_time[d] += car.wait_time
            if car.is_waiting and not car._past_stop():
                waiting_counts[d] += 1

        ns_queue = queue["N"] + queue["S"]
        ew_queue = queue["E"] + queue["W"]
        ns_wait_time = waiting_time["N"] + waiting_time["S"]
        ew_wait_time = waiting_time["E"] + waiting_time["W"]

        pressure = (ns_queue - ew_queue) / 20.0
        wait_pressure = (ns_wait_time - ew_wait_time) / 500.0
        total_load = sum(queue.values()) / 20.0
        green_ratio = self.chosen_green / max(self.MAX_GREEN_TIME, 1)

        phase_ns = 1.0 if self.phase == self.PHASE_NS_GREEN else 0.0
        phase_ew = 1.0 if self.phase == self.PHASE_EW_GREEN else 0.0

        state = np.array(
            [
                queue["N"] / 10.0,
                queue["S"] / 10.0,
                queue["E"] / 10.0,
                queue["W"] / 10.0,
                waiting_counts["N"] / 10.0,
                waiting_counts["S"] / 10.0,
                waiting_counts["E"] / 10.0,
                waiting_counts["W"] / 10.0,
                min(waiting_time["N"] / 200.0, 5.0),
                min(waiting_time["S"] / 200.0, 5.0),
                min(waiting_time["E"] / 200.0, 5.0),
                min(waiting_time["W"] / 200.0, 5.0),
                phase_ns,
                phase_ew,
                green_ratio,
                pressure,
                wait_pressure,
                total_load,
                self.episode_frame / self.EPISODE_LENGTH if self.use_rl else 0.0,
            ],
            dtype=np.float32,
        )
        return state, queue, waiting_time, waiting_counts

    # Thời lượng tham chiếu để normalize reward
    REF_DURATION = 120.0 + YELLOW_TIME + ALL_RED_TIME  # ~152 frames

    def _decode_action(self, action):
        phase_idx = action // len(GREEN_DURATIONS)
        duration_idx = action % len(GREEN_DURATIONS)
        next_phase = self.PHASE_NS_GREEN if phase_idx == 0 else self.PHASE_EW_GREEN
        green_time = GREEN_DURATIONS[duration_idx]
        return next_phase, green_time

    def _phase_lanes(self, phase):
        if phase == self.PHASE_NS_GREEN:
            return ("N", "S"), ("E", "W")
        return ("E", "W"), ("N", "S")

    def _calculate_step_reward(self, waiting_counts, waiting_time, throughput,
                               in_transition=False):
        """
        Reward stationary, bounded:
        - Phạt xe đang chờ, đặc biệt là xe ở pha đỏ
        - Thưởng throughput
        - Phạt thêm khi đang transition (yellow/all-red)
        """
        total_waiting = sum(waiting_counts.values())
        _, red_lanes = self._phase_lanes(self.phase)
        red_waiting = sum(waiting_counts[d] for d in red_lanes)
        red_wait_time = sum(waiting_time[d] for d in red_lanes)

        reward = 0.0

        # Phạt chính: số xe đang chờ ở pha đỏ
        reward -= 0.20 * red_waiting
        reward -= 0.002 * red_wait_time

        # Phạt nhẹ cho toàn bộ queue để tránh bỏ đói làn xanh
        reward -= 0.05 * total_waiting

        # Thưởng throughput
        reward += 2.0 * throughput

        # Phạt transition: khuyến khích chuyển phase có chọn lọc
        if in_transition:
            reward -= 0.15 * total_waiting

        return reward

    def step(self, cars, throughput=0, training=True):
        self.frame += 1
        state, queue, waiting_time, waiting_counts = self.get_state(cars)
        in_transition = self.in_yellow or self.in_all_red
        step_reward = self._calculate_step_reward(
            waiting_counts, waiting_time, throughput,
            in_transition=in_transition,
        )

        if self.use_rl:
            self.accumulated_reward += step_reward
            self.frames_since_last_decision += 1
            self.episode_frame += 1
            self.episode_just_ended = False

        # Yellow / All-red
        if self.in_yellow or self.in_all_red:
            self.transition_timer -= 1
            if self.in_yellow and self.transition_timer <= 0:
                self.in_yellow = False
                self.in_all_red = True
                self.transition_timer = self.ALL_RED_TIME
            elif self.in_all_red and self.transition_timer <= 0:
                self.in_all_red = False
                if self.pending_phase is not None:
                    self.phase = self.pending_phase
                self.pending_phase = None
                self.time_in_phase = 0
                self.needs_decision = True
            return step_reward, 0

        # GREEN PHASE
        self.time_in_phase += 1

        if self.needs_decision:
            self.needs_decision = False

            if self.use_rl:
                action, log_prob, value = self.select_action(state, training)
                is_done = self.episode_frame >= self.EPISODE_LENGTH
                next_phase, chosen_green = self._decode_action(action)

                if training and self.last_state is not None:
                    # REF_DURATION normalization:
                    # Normalize tổng reward theo thời lượng tham chiếu
                    # → Phase ngắn: reward scale up (công bằng so sánh)
                    # → Phase dài: reward scale down
                    # → Agent chọn cả phase tiếp theo và duration tối ưu theo traffic
                    tau = max(1, self.frames_since_last_decision)
                    phase_reward = self.accumulated_reward * (self.REF_DURATION / tau)

                    self.agent.store_transition(
                        self.last_state, self.last_action,
                        self.last_log_prob, self.last_value,
                        phase_reward, done=is_done,
                    )
                    if self.agent.ready():
                        last_value = 0.0 if is_done else self.agent.get_value(state)
                        self.agent.update(last_value=last_value)

                self.accumulated_reward = 0.0
                self.frames_since_last_decision = 0
                if is_done:
                    self.episode_frame = 0
                    self.episode_just_ended = True
                    # Reset internal state cho episode mới
                    self.last_state = None
                    self.last_action = None

                if not is_done:
                    self.pending_phase = next_phase
                    self.green_remaining = chosen_green
                    self.chosen_green = chosen_green
                    self.last_state = state.copy()
                    self.last_action = action
                    self.last_log_prob = log_prob
                    self.last_value = value
                    phase_name = "NS" if next_phase == self.PHASE_NS_GREEN else "EW"
                    self.last_switch_reason = f"P={phase_name},G={self.chosen_green}"
                else:
                    # Episode vừa kết thúc, dùng default green cho transition
                    self.green_remaining = 120
                    self.chosen_green = 120
                    self.last_switch_reason = "EPISODE_END"
            else:
                self.green_remaining = 120
                self.chosen_green = 120

        self.green_remaining -= 1
        if self.green_remaining <= 0:
            self._do_switch()
            self.last_switch_reason = "AUTO"
            return step_reward, 1

        return step_reward, 0

    def reset_state(self):
        """Reset traffic light state cho episode mới."""
        self.phase = self.PHASE_NS_GREEN
        self.time_in_phase = 0
        self.pending_phase = None
        self.in_yellow = False
        self.in_all_red = False
        self.transition_timer = 0
        self.green_remaining = 0
        self.chosen_green = 120
        self.needs_decision = True
        self.last_switch_reason = "INIT"
        self.last_switch_frame = -9999
        if self.use_rl:
            self.accumulated_reward = 0.0
            self.frames_since_last_decision = 0

    def _do_switch(self, force=False):
        self.in_yellow = True
        self.in_all_red = False
        self.transition_timer = self.YELLOW_TIME
        if self.pending_phase is None:
            self.pending_phase = 1 - self.phase
        self.last_switch_frame = self.frame

    def select_action(self, state, training=True):
        if not self.use_rl:
            return 3, 0.0, 0.0  # Index 3 = 120 = fixed timing
        return self.agent.select_action(state, training)

    def get(self, direction):
        if self.in_all_red:
            return "red"
        if self.in_yellow:
            if self.phase == self.PHASE_NS_GREEN:
                return "yellow" if direction in ("N", "S") else "red"
            return "yellow" if direction in ("E", "W") else "red"
        if direction in ("N", "S"):
            return "green" if self.phase == self.PHASE_NS_GREEN else "red"
        return "green" if self.phase == self.PHASE_EW_GREEN else "red"

    def is_green(self, direction):
        return self.get(direction) == "green"

    def is_passable(self, direction):
        return self.get(direction) == "green"


# =====================================================================
# PARALLEL TRAINING
# =====================================================================

class HeadlessEnv:
    """Một môi trường headless để training."""

    def __init__(self, seed=0):
        from traffic_entities import Car, DemandController
        from traffic_constants import CAR_COLORS, H, W, SPAWN_BLOCK_DIST

        self.seed = seed
        self.rng = random.Random(seed)
        self.cars = []
        self.frame = 0
        self.demand = DemandController(rng=self.rng)
        self.spawn_bp = {"N": 0.010, "S": 0.010, "E": 0.012, "W": 0.012}
        self._episode_count = 0

    def reset(self):
        """Reset environment cho episode mới — tránh reward drift."""
        from traffic_entities import DemandController
        self._episode_count += 1
        self.rng = random.Random(self.seed + self._episode_count * 7919)
        self.cars = []
        self.frame = 0
        self.demand = DemandController(rng=self.rng)

    def _can_spawn(self, d):
        from traffic_constants import H, W, SPAWN_BLOCK_DIST
        for c in self.cars:
            if c.direction != d or c.state == "done": continue
            if d == "N" and c.y > H - SPAWN_BLOCK_DIST: return False
            if d == "S" and c.y < SPAWN_BLOCK_DIST: return False
            if d == "E" and c.x > W - SPAWN_BLOCK_DIST: return False
            if d == "W" and c.x < SPAWN_BLOCK_DIST: return False
        return True

    def _cars_ahead(self, car):
        ahead = []
        for other in self.cars:
            if other.id == car.id or other.direction != car.direction: continue
            if car.direction == "N" and other.y < car.y: ahead.append(other)
            elif car.direction == "S" and other.y > car.y: ahead.append(other)
            elif car.direction == "E" and other.x < car.x: ahead.append(other)
            elif car.direction == "W" and other.x > car.x: ahead.append(other)
        return ahead

    def step(self, tl):
        from traffic_entities import Car
        from traffic_constants import CAR_COLORS
        self.frame += 1
        self.demand.tick()
        for d in ("N", "S", "E", "W"):
            p_base = self.demand.base_prob(d, self.spawn_bp)
            p_burst = self.demand.burst_prob(d, 0.030)
            count = 0
            if self.rng.random() < p_base: count += 1
            if self.rng.random() < p_burst: count += self.rng.randint(1, 3)
            for _ in range(min(count, 3)):
                if not self._can_spawn(d): break
                self.cars.append(Car(d, self.rng.choice(CAR_COLORS[d])))
        for car in self.cars:
            if car.state != "done":
                car.update(tl.is_passable(car.direction), self._cars_ahead(car))
        done = [c for c in self.cars if c.state == "done"]
        self.cars = [c for c in self.cars if c.state != "done"]
        return len(done)


def _make_tl(agent):
    """Tạo TrafficLightRL instance chia sẻ agent."""
    tl = TrafficLightRL.__new__(TrafficLightRL)
    tl.use_rl = True
    tl.phase = TrafficLightRL.PHASE_NS_GREEN
    tl.time_in_phase = 0
    tl.frame = 0
    tl.pending_phase = None
    tl.in_yellow = False
    tl.in_all_red = False
    tl.transition_timer = 0
    tl.last_switch_reason = "INIT"
    tl.last_switch_frame = -9999
    tl.last_action = None
    tl.green_remaining = 0
    tl.chosen_green = 120
    tl.needs_decision = True
    tl.agent = agent
    tl.last_state = None
    tl.last_log_prob = 0.0
    tl.last_value = 0.0
    tl.accumulated_reward = 0.0
    tl.frames_since_last_decision = 0
    tl.episode_frame = 0
    tl.episode_just_ended = False
    return tl


def train_parallel(total_frames=80000, n_envs=8, base_seed=42):
    """Training song song: N env, 1 shared agent, với env reset."""
    from collections import deque

    agent = PPOAgent(state_size=19, action_size=TrafficLightRL.ACTION_SIZE)
    envs = [HeadlessEnv(seed=base_seed + i * 1000) for i in range(n_envs)]
    tls = [_make_tl(agent) for _ in range(n_envs)]

    best_score = -float("inf")
    rq, rw, rt = deque(maxlen=300), deque(maxlen=300), deque(maxlen=300)

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

        # Track env 0 metrics
        wn = sum(1 for c in envs[0].cars if c.is_waiting and not c._past_stop())
        qn = len(envs[0].cars)
        rt.append(sum(1 for c in envs[0].cars if c.state == "done"))
        rw.append(wn)
        rq.append(qn)

        if frame % 10000 == 0 and len(rt) >= 300:
            tpm = sum(rt) / len(rt) * 60
            aw = sum(rw) / len(rw)
            aq = sum(rq) / len(rq)
            sc = tpm - 1.8 * aw - 1.0 * aq
            if sc > best_score:
                best_score = sc
                agent.save("ppo_model.pth")
            print(f"  f={frame} tp={tpm:.1f} wait={aw:.1f} q={aq:.1f} sc={sc:.1f} best={best_score:.1f} steps={agent.total_steps}")

    agent.save("ppo_model_last.pth")
    return agent