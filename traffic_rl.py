"""
traffic_rl.py - PHIÊN BẢN TỐI ƯU HOÀN CHỈNH
Mục tiêu: Reward ổn định + Giảm 12-25% thời gian chờ sau 50k frames
"""

import numpy as np
from traffic_dqn import DQNAgent
from traffic_ppo import PPOAgent


class TrafficLightRL:
    """Đèn giao thông RL - dùng chung môi trường cho PPO và DQN."""

    STATE_SIZE = 17

    PHASE_NS_GREEN = 0
    PHASE_EW_GREEN = 1

    # ================== TIMING ==================
    MIN_GREEN_TIME = 95
    MAX_GREEN_TIME = 240
    HARD_MAX_GREEN_TIME = 320
    DECISION_INTERVAL = 15
    YELLOW_TIME = 12
    ALL_RED_TIME = 4
    MIN_SWITCH_GAP = 35

    SWITCH_QUEUE_THRESHOLD = 2
    LOW_TRAFFIC_SWITCH_THRESHOLD = 1

    def __init__(self, use_rl=True, algorithm="ppo"):
        self.use_rl = use_rl
        self.algorithm = algorithm.lower()
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
        self.prev_traffic_cost = None
        self.last_reward = 0.0

        if use_rl:
            if self.algorithm == "ppo":
                self.agent = PPOAgent(state_size=self.STATE_SIZE, action_size=2)
            elif self.algorithm == "dqn":
                self.agent = DQNAgent(state_size=self.STATE_SIZE, action_size=2)
            else:
                raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}")
            self.decision_state = None
            self.decision_action = None
            self.decision_log_prob = 0.0
            self.decision_value = 0.0
            self.decision_reward = 0.0

    def get_state(self, cars):
        queue = {"N": 0, "S": 0, "E": 0, "W": 0}
        waiting_time = {"N": 0, "S": 0, "E": 0, "W": 0}
        waiting_counts = {"N": 0, "S": 0, "E": 0, "W": 0}

        for car in cars:
            if car.state == "done":
                continue
            d = car.direction
            queue[d] += 1
            before_stop = car._distance_to_stop() > 0.1 and not car._past_stop()
            if before_stop:
                waiting_time[d] += car.wait_time
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

    def _traffic_cost(self, queue, waiting_time, waiting_counts):
        """Chi phí ùn tắc càng thấp càng tốt."""
        total_queue = sum(queue.values())
        total_waiting = sum(waiting_counts.values())
        total_wait_time = sum(waiting_time.values())
        ns_wait = waiting_counts["N"] + waiting_counts["S"]
        ew_wait = waiting_counts["E"] + waiting_counts["W"]
        ns_queue = queue["N"] + queue["S"]
        ew_queue = queue["E"] + queue["W"]
        imbalance = abs(ns_queue - ew_queue)

        return (
            0.085 * total_queue
            + 0.080 * total_waiting
            + 0.0030 * total_wait_time
            + 0.0035 * (ns_wait**2 + ew_wait**2)
            + 0.010 * imbalance
        )

    def calculate_reward(self, queue, waiting_time, waiting_counts, throughput,
                        switched=False, force_switch=False, phase_time_before_switch=0):
        """Reward kết hợp giảm ùn tắc theo thời gian, throughput và penalty đổi pha."""
        cost = self._traffic_cost(queue, waiting_time, waiting_counts)
        if self.prev_traffic_cost is None:
            cost_delta = 0.0
        else:
            cost_delta = self.prev_traffic_cost - cost
        self.prev_traffic_cost = cost

        total_waiting = sum(waiting_counts.values())
        total_wait_time = sum(waiting_time.values())
        ns_wait = waiting_counts["N"] + waiting_counts["S"]
        ew_wait = waiting_counts["E"] + waiting_counts["W"]
        current_wait = ns_wait if self.phase == self.PHASE_NS_GREEN else ew_wait
        opposite_wait = ew_wait if self.phase == self.PHASE_NS_GREEN else ns_wait

        total_queue = sum(queue.values())

        reward = 0.50 * cost_delta
        reward -= 0.0110 * total_queue
        reward -= 0.0120 * total_waiting
        reward -= 0.00055 * total_wait_time
        reward -= 0.0035 * opposite_wait

        if switched:
            reward -= 0.32
            if opposite_wait > current_wait + 5:
                reward += 0.025 * min(opposite_wait - current_wait, 10)
            if phase_time_before_switch < self.MIN_GREEN_TIME + 20:
                reward -= 0.55

        if force_switch:
            reward -= 0.04

        reward = float(np.clip(reward, -1.0, 1.0))
        self.last_reward = reward
        return reward

    def select_action(self, state, training=True):
        if not self.use_rl:
            return 0, 0.0, 0.0
        if self.algorithm == "ppo":
            return self.agent.select_action(state, training)
        return self.agent.select_action(state, training), 0.0, 0.0

    def _evaluate_action(self, state, action):
        if not self.use_rl or self.algorithm != "ppo":
            return 0.0, 0.0
        return self.agent.evaluate_action(state, action)

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

            action = 0
            log_prob, value = self._evaluate_action(state, action)
            reward = self.calculate_reward(queue, waiting_time, waiting_counts, throughput)
            self._accumulate_decision_reward(reward)
            return reward, 0

        # Normal green phase
        phase_time_before = self.time_in_phase
        can_decide = (self.time_in_phase >= self.MIN_GREEN_TIME and 
                     self.time_in_phase % self.DECISION_INTERVAL == 0)

        if can_decide:
            self._close_decision(state, training)
            action, log_prob, value = self.select_action(state, training)
        else:
            action = 0
            log_prob, value = self._evaluate_action(state, 0)

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

        switch_allowed = (
            opposite >= current + 5 or
            (opposite >= 6 and current <= 1) or
            (self.time_in_phase >= 150 and opposite >= self.SWITCH_QUEUE_THRESHOLD) or
            (ns_waiting + ew_waiting >= 18 and opposite >= current + 2)
        )

        if action == 1 and not force_switch and not switch_allowed:
            action = 0
            log_prob, value = self._evaluate_action(state, 0)
        self.last_action = action

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

        self._accumulate_decision_reward(reward)
        if can_decide:
            self._begin_decision(state, action, log_prob, value)
        return reward, action

    def _begin_decision(self, state, action, log_prob, value):
        if not self.use_rl:
            return
        self.decision_state = state.copy()
        self.decision_action = action
        self.decision_log_prob = log_prob
        self.decision_value = value
        self.decision_reward = 0.0

    def _accumulate_decision_reward(self, reward):
        if self.use_rl and self.decision_state is not None:
            self.decision_reward += reward

    def _close_decision(self, next_state, training):
        if not self.use_rl or self.decision_state is None:
            return
        if training:
            if self.algorithm == "ppo":
                self.agent.store_transition(
                    self.decision_state, self.decision_action, self.decision_log_prob,
                    self.decision_value, self.decision_reward, done=False
                )
                if self.agent.ready():
                    last_v = self.agent.get_value(next_state)
                    self.agent.update(last_value=last_v)
            elif self.algorithm == "dqn":
                self.agent.store_experience(
                    self.decision_state, self.decision_action,
                    self.decision_reward, next_state.copy(), False
                )
                self.agent.update()
        self.decision_state = None
        self.decision_action = None
        self.decision_log_prob = 0.0
        self.decision_value = 0.0
        self.decision_reward = 0.0

    def finalize_training(self, cars):
        if not self.use_rl:
            return
        state, _, _, _ = self.get_state(cars)
        self._close_decision(state, training=True)
        if self.algorithm == "ppo" and self.agent.states:
            self.agent.update(last_value=self.agent.get_value(state))

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
