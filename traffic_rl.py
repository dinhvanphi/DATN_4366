"""
traffic_rl.py - PHIÊN BẢN TỐI ƯU HOÀN CHỈNH
Mục tiêu: Reward ổn định + Giảm 12-25% thời gian chờ sau 50k frames
"""

import numpy as np
from traffic_ppo import PPOAgent


class TrafficLightRL:
    """Đèn giao thông PPO - Tối ưu reward và timing"""

    STATE_SIZE = 17

    PHASE_NS_GREEN = 0
    PHASE_EW_GREEN = 1

    # ================== TIMING ==================
    MIN_GREEN_TIME = 45
    MAX_GREEN_TIME = 300
    HARD_MAX_GREEN_TIME = 400
    DECISION_INTERVAL = 5
    YELLOW_TIME = 12
    ALL_RED_TIME = 4
    MIN_SWITCH_GAP = 35

    SWITCH_QUEUE_THRESHOLD = 2
    LOW_TRAFFIC_SWITCH_THRESHOLD = 1

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

        if use_rl:
            self.agent = PPOAgent(state_size=self.STATE_SIZE, action_size=2)
            self.last_state = None
            self.last_action = None
            self.last_log_prob = 0.0
            self.last_value = 0.0

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
        pressure = (ns_queue - ew_queue) / 20.0
        ns_wait_time = waiting_time["N"] + waiting_time["S"]
        ew_wait_time = waiting_time["E"] + waiting_time["W"]
        wait_pressure = (ns_wait_time - ew_wait_time) / 200.0
        total_load = sum(queue.values()) / 20.0
        phase_ratio = min(self.time_in_phase / max(self.MAX_GREEN_TIME, 1), 1.0)

        state = np.array([
            queue["N"]/10, queue["S"]/10, queue["E"]/10, queue["W"]/10,
            waiting_counts["N"]/10, waiting_counts["S"]/10, waiting_counts["E"]/10, waiting_counts["W"]/10,
            min(waiting_time["N"]/200, 5), min(waiting_time["S"]/200, 5),
            min(waiting_time["E"]/200, 5), min(waiting_time["W"]/200, 5),
            float(self.phase), phase_ratio, pressure, wait_pressure, total_load
        ], dtype=np.float32)

        return state, queue, waiting_time, waiting_counts

    def calculate_reward(self, queue, waiting_time, waiting_counts, throughput,
                        switched=False, force_switch=False, phase_time_before_switch=0):
        """Reward tập trung hoàn toàn vào việc giảm thời gian chờ (bằng cách phạt số lượng xe chờ)"""
        total_waiting_cars = sum(waiting_counts.values())

        # Đếm số xe đang chờ theo trục (Bắc-Nam và Đông-Tây)
        ns_wait = waiting_counts["N"] + waiting_counts["S"]
        ew_wait = waiting_counts["E"] + waiting_counts["W"]

        # 1. Phạt BÌNH PHƯƠNG số xe chờ của mỗi hướng.
        # Lý do: 1 hàng đợi dài (ví dụ 15 xe -> phạt 225) sẽ tồi tệ hơn rất nhiều
        # so với 2 hàng đợi ngắn (ví dụ 8 xe + 7 xe -> phạt 64 + 49 = 113).
        # Điều này ép Agent PHẢI đổi đèn sớm trước khi hàng đợi ở hướng đèn Đỏ trở nên quá dài!
        reward = -0.01 * (ns_wait**2 + ew_wait**2)

        # 2. Thưởng khi xe thoát khỏi ngã tư
        reward += 0.5 * throughput
        
        # 3. Phạt chuyển phase để tránh flickering, nhưng không phạt quá nặng để nó dám chuyển
        if switched:
            reward -= 0.5  # Giảm penalty chuyển đèn xuống
            if phase_time_before_switch < self.MIN_GREEN_TIME + 10:
                reward -= 2.0  # Vẫn cấm chuyển quá sớm (< 55 frames)

        if force_switch:
            reward -= 0.5

        return float(np.clip(reward, -10.0, 10.0))

    def select_action(self, state, training=True):
        if not self.use_rl:
            return 0, 0.0, 0.0
        return self.agent.select_action(state, training)

    def step(self, cars, throughput=0, training=True):
        self.frame += 1
        state, queue, waiting_time, waiting_counts = self.get_state(cars)

        # Yellow / All-red transition
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

            reward = self.calculate_reward(queue, waiting_time, waiting_counts, throughput)
            self._store_transition(state, 0, reward, training)
            return reward, 0

        # Normal green phase
        phase_time_before = self.time_in_phase
        can_decide = (self.time_in_phase >= self.MIN_GREEN_TIME and 
                     self.time_in_phase % self.DECISION_INTERVAL == 0)

        if can_decide:
            action, log_prob, value = self.select_action(state, training)
        else:
            action = 0
            if self.use_rl:
                log_prob, value = self.agent.evaluate_action(state, 0)
            else:
                log_prob = value = 0.0

        # Switch logic
        ns_waiting = waiting_counts["N"] + waiting_counts["S"]
        ew_waiting = waiting_counts["E"] + waiting_counts["W"]
        current = ns_waiting if self.phase == self.PHASE_NS_GREEN else ew_waiting
        opposite = ew_waiting if self.phase == self.PHASE_NS_GREEN else ns_waiting

        if not self.use_rl:
            hard_force = self.time_in_phase >= 140
            soft_force = False
        else:
            hard_force = self.time_in_phase >= self.HARD_MAX_GREEN_TIME
            soft_force = (self.time_in_phase >= self.MAX_GREEN_TIME and opposite >= self.SWITCH_QUEUE_THRESHOLD + 2)

        force_switch = hard_force or soft_force

        switch_allowed = (opposite >= self.SWITCH_QUEUE_THRESHOLD or 
                         opposite >= current + 2 or 
                         (current == 0 and opposite >= self.LOW_TRAFFIC_SWITCH_THRESHOLD) or 
                         (ns_waiting + ew_waiting >= 12))

        if action == 1 and not force_switch and not switch_allowed:
            action = 0
            if self.use_rl:
                log_prob, value = self.agent.evaluate_action(state, 0)

        switched = False
        if (action == 1 or force_switch) and self.time_in_phase >= self.MIN_GREEN_TIME:
            self.in_yellow = True
            self.transition_timer = self.YELLOW_TIME
            self.pending_phase = 1 - self.phase
            switched = True
            self.last_switch_frame = self.frame
            self.last_switch_reason = "AGENT" if action == 1 else ("HARD_FORCE" if hard_force else "SOFT_FORCE")
        else:
            self.time_in_phase += 1
            self.last_switch_reason = "HOLD"

        reward = self.calculate_reward(queue, waiting_time, waiting_counts, throughput,
                                      switched=switched, force_switch=force_switch,
                                      phase_time_before_switch=phase_time_before)

        self._store_transition(state, action, reward, training)
        return reward, action

    def _store_transition(self, state, action, reward, training):
        if training and self.use_rl and self.last_state is not None:
            self.agent.store_transition(
                self.last_state, self.last_action, self.last_log_prob,
                self.last_value, reward, done=False
            )
            if self.agent.ready():
                last_v = self.agent.get_value(state)
                self.agent.update(last_value=last_v)

        self.last_state = state.copy()
        self.last_action = action
        self.last_log_prob = 0.0
        self.last_value = 0.0

    def get(self, direction):
        if self.in_all_red:
            return "red"
        if self.in_yellow:
            is_ns = self.phase == self.PHASE_NS_GREEN
            return "yellow" if (is_ns and direction in ("N","S")) or (not is_ns and direction in ("E","W")) else "red"
        is_ns_green = self.phase == self.PHASE_NS_GREEN
        return "green" if (is_ns_green and direction in ("N","S")) or (not is_ns_green and direction in ("E","W")) else "red"

    def is_green(self, direction):
        return self.get(direction) == "green"

    def is_passable(self, direction):
        return self.get(direction) == "green"