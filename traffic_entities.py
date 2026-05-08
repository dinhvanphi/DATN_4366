"""
traffic_entities.py
-------------------
Chứa các thực thể nền tảng của mô phỏng: xe, đèn giao thông, điều khiển nhu cầu,
và các lớp vẽ giao diện như nền đường, đèn hiển thị, HUD.
"""

import math
import random
from collections import deque

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from traffic_constants import (
    CAR_COLORS,
    CAR_H,
    CAR_W,
    CENTER,
    CLR_CROSS,
    CLR_GREEN,
    CLR_RED,
    CLR_ROAD,
    CLR_YELLOW,
    H,
    INTER,
    LANE_X,
    LANE_Y,
    ROAD_W,
    SAFE_GAP,
    SPAWN_BLOCK_DIST,
    STOP_GAP,
    V_CRUISE,
    V_SLOW,
    W,
    GREEN_TIME,
    YELLOW_TIME,
    RED_TIME,
    ALL_RED_TIME,
)


class DemandController:
    """Điều khiển nhu cầu giao thông theo từng giai đoạn."""

    def __init__(self, rng=None):
        self.rng = rng or random
        self.mode = self.rng.choice(("NS_HIGH", "EW_HIGH"))
        self.mode_timer = 0
        self.mode_duration = self.rng.randint(350, 900)
        self.high_mult = 2.2
        self.low_mult = 0.45
        self.blocked_lane = None
        self.block_timer = 0
        self.next_block_after = self.rng.randint(220, 520)

    def tick(self):
        self.mode_timer += 1

        if self.blocked_lane is None:
            self.next_block_after -= 1
            if self.next_block_after <= 0:
                self.blocked_lane = self.rng.choice(("N", "S", "E", "W"))
                self.block_timer = self.rng.randint(120, 300)
                self.next_block_after = self.rng.randint(260, 620)
        else:
            self.block_timer -= 1
            if self.block_timer <= 0:
                self.blocked_lane = None

        if self.mode_timer >= self.mode_duration:
            self.mode_timer = 0
            self.mode = "EW_HIGH" if self.mode == "NS_HIGH" else "NS_HIGH"
            self.high_mult = self.rng.uniform(1.8, 2.8)
            self.low_mult = self.rng.uniform(0.28, 0.70)
            self.mode_duration = self.rng.randint(320, 980)

    def _multiplier(self, direction):
        if self.mode == "NS_HIGH":
            return self.high_mult if direction in ("N", "S") else self.low_mult
        return self.high_mult if direction in ("E", "W") else self.low_mult

    def base_prob(self, direction, base_prob):
        if direction == self.blocked_lane:
            return 0.0
        p = base_prob[direction] * self._multiplier(direction)
        return float(np.clip(p, 0.001, 0.18))

    def burst_prob(self, direction, burst_prob):
        if direction == self.blocked_lane:
            return 0.0
        mult = 1.35 if self._multiplier(direction) > 1.0 else 0.60
        p = burst_prob * mult
        return float(np.clip(p, 0.003, 0.20))


class Car:
    _id_counter = 0

    def __init__(self, direction: str, color: str):
        Car._id_counter += 1
        self.id = Car._id_counter
        self.direction = direction
        self.color = color
        self.state = "moving"
        self.passed = False
        self.wait_time = 0.0
        self.is_waiting = False

        if direction == "N":
            self.x = LANE_X["N"]
            self.y = H + 1.0
            self.dx, self.dy = 0, -1
            self.stop_y = CENTER + INTER / 2 + STOP_GAP
            self.w, self.h = CAR_W, CAR_H
        elif direction == "S":
            self.x = LANE_X["S"]
            self.y = -1.0
            self.dx, self.dy = 0, +1
            self.stop_y = CENTER - INTER / 2 - STOP_GAP
            self.w, self.h = CAR_W, CAR_H
        elif direction == "E":
            self.x = W + 1.0
            self.y = LANE_Y["E"]
            self.dx, self.dy = -1, 0
            self.stop_x = CENTER + INTER / 2 + STOP_GAP
            self.w, self.h = CAR_H, CAR_W
        else:
            self.x = -1.0
            self.y = LANE_Y["W"]
            self.dx, self.dy = +1, 0
            self.stop_x = CENTER - INTER / 2 - STOP_GAP
            self.w, self.h = CAR_H, CAR_W

        self.patch = mpatches.FancyBboxPatch(
            (self.x - self.w / 2, self.y - self.h / 2),
            self.w,
            self.h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#00000066",
            linewidth=0.5,
            zorder=5,
        )
        self.last_x = self.x
        self.last_y = self.y

    def _front_pos(self):
        if self.direction == "N":
            return self.y - self.h / 2
        if self.direction == "S":
            return self.y + self.h / 2
        if self.direction == "E":
            return self.x - self.w / 2
        return self.x + self.w / 2

    def _distance_to_stop(self):
        if self.direction == "N":
            return self._front_pos() - self.stop_y
        if self.direction == "S":
            return self.stop_y - self._front_pos()
        if self.direction == "E":
            return self._front_pos() - self.stop_x
        return self.stop_x - self._front_pos()

    def _past_stop(self):
        if self.direction == "N":
            return self.y < CENTER + INTER / 2
        if self.direction == "S":
            return self.y > CENTER - INTER / 2
        if self.direction == "E":
            return self.x < CENTER + INTER / 2
        return self.x > CENTER - INTER / 2

    def _is_out(self):
        margin = 2.0
        if self.direction == "N":
            return self.y < -margin
        if self.direction == "S":
            return self.y > H + margin
        if self.direction == "E":
            return self.x < -margin
        return self.x > W + margin

    def update(self, is_green, cars_ahead):
        if self.state == "done":
            return

        prev_x = self.x
        prev_y = self.y

        dist_to_stop = self._distance_to_stop()
        past_stop = self._past_stop()

        gap_to_leader = float("inf")
        for other in cars_ahead:
            if other.id == self.id or other.state == "done":
                continue
            if self.direction == "N":
                d = self.y - other.h / 2 - (other.y + other.h / 2)
                if other.y < self.y and d < gap_to_leader:
                    gap_to_leader = d
            elif self.direction == "S":
                d = other.y - other.h / 2 - (self.y + self.h / 2)
                if other.y > self.y and d < gap_to_leader:
                    gap_to_leader = d
            elif self.direction == "E":
                d = self.x - other.w / 2 - (other.x + other.w / 2)
                if other.x < self.x and d < gap_to_leader:
                    gap_to_leader = d
            else:
                d = other.x - other.w / 2 - (self.x + self.w / 2)
                if other.x > self.x and d < gap_to_leader:
                    gap_to_leader = d

        if not past_stop:
            if not is_green and dist_to_stop > 0.01:
                if dist_to_stop < 1.5:
                    target_v = max(0, V_CRUISE * (dist_to_stop / 1.5))
                    if dist_to_stop < STOP_GAP:
                        target_v = 0
                else:
                    target_v = V_CRUISE
            else:
                target_v = V_CRUISE
        else:
            target_v = V_CRUISE

        if gap_to_leader < SAFE_GAP + 0.1:
            target_v = min(target_v, max(0, V_SLOW * gap_to_leader / SAFE_GAP))
        if gap_to_leader < STOP_GAP:
            target_v = 0

        self.x += self.dx * target_v
        self.y += self.dy * target_v

        if self._is_out():
            self.state = "done"

        self._update_patch()

        moved = math.hypot(self.x - prev_x, self.y - prev_y)
        if not past_stop and dist_to_stop > 0.1 and moved < 1e-4:
            self.wait_time += 1.0
            self.is_waiting = True
        else:
            self.is_waiting = False
        self.last_x = self.x
        self.last_y = self.y

    def _update_patch(self):
        self.patch.set_x(self.x - self.w / 2)
        self.patch.set_y(self.y - self.h / 2)

    def add_to_ax(self, ax):
        ax.add_patch(self.patch)

    def remove(self):
        self.patch.remove()


class TrafficLight:
    """Chu kỳ đèn cố định cho mô phỏng baseline."""

    def __init__(self):
        self.frame = 0
        # Một chu kỳ = (Xanh + Vàng + Đỏ Toàn Bộ) * 2
        self.cycle_half = GREEN_TIME + YELLOW_TIME + ALL_RED_TIME

    def tick(self):
        self.frame += 1

    def _phase_ns(self):
        t = self.frame % (self.cycle_half * 2)
        if t < GREEN_TIME:
            return "green"
        if t < GREEN_TIME + YELLOW_TIME:
            return "yellow"
        return "red"

    def _phase_ew(self):
        # Lệch một nửa chu kỳ
        t = (self.frame + self.cycle_half) % (self.cycle_half * 2)
        if t < GREEN_TIME:
            return "green"
        if t < GREEN_TIME + YELLOW_TIME:
            return "yellow"
        return "red"

    def get(self, direction):
        return self._phase_ns() if direction in ("N", "S") else self._phase_ew()

    def is_green(self, direction):
        return self.get(direction) == "green"

    def is_passable(self, direction):
        return self.get(direction) == "green"


def draw_background(ax):
    ax.set_facecolor("#1c1c1c")

    c = CENTER
    r = INTER / 2
    rw = ROAD_W * 2

    ax.add_patch(plt.Rectangle((0, 0), W, H, facecolor="#3a3a3a", zorder=0))
    ax.add_patch(plt.Rectangle((c - rw / 2, 0), rw, H, facecolor=CLR_ROAD, zorder=1))
    ax.add_patch(plt.Rectangle((0, c - rw / 2), W, rw, facecolor=CLR_ROAD, zorder=1))

    dash_kw = dict(color="#ffffffaa", linewidth=0.8, linestyle=(0, (8, 6)), zorder=2)
    ax.plot([c, c], [0, c - r], **dash_kw)
    ax.plot([c, c], [c + r, H], **dash_kw)
    ax.plot([0, c - r], [c, c], **dash_kw)
    ax.plot([c + r, W], [c, c], **dash_kw)

    thin_kw = dict(color="#ffffff44", linewidth=0.5, linestyle=(0, (4, 8)), zorder=2)
    ax.plot([c - ROAD_W, c - ROAD_W], [0, c - r], **thin_kw)
    ax.plot([c - ROAD_W, c - ROAD_W], [c + r, H], **thin_kw)
    ax.plot([c + ROAD_W, c + ROAD_W], [0, c - r], **thin_kw)
    ax.plot([c + ROAD_W, c + ROAD_W], [c + r, H], **thin_kw)
    ax.plot([0, c - r], [c - ROAD_W, c - ROAD_W], **thin_kw)
    ax.plot([c + r, W], [c - ROAD_W, c - ROAD_W], **thin_kw)
    ax.plot([0, c - r], [c + ROAD_W, c + ROAD_W], **thin_kw)
    ax.plot([c + r, W], [c + ROAD_W, c + ROAD_W], **thin_kw)

    stop_kw = dict(color="#ffffffff", linewidth=2.5, zorder=3)
    ax.plot([c - rw / 2, c], [c + r, c + r], **stop_kw)
    ax.plot([c, c + rw / 2], [c - r, c - r], **stop_kw)
    ax.plot([c + r, c + r], [c, c + rw / 2], **stop_kw)
    ax.plot([c - r, c - r], [c - rw / 2, c], **stop_kw)

    zw = 0.22
    for i in range(5):
        offset = -0.5 + i * 0.25
        ax.add_patch(plt.Rectangle((c - rw / 2 + offset * 0.8, c + r + 0.08), zw, 0.35, facecolor=CLR_CROSS, zorder=2))
        ax.add_patch(plt.Rectangle((c + offset * 0.8, c - r - 0.43), zw, 0.35, facecolor=CLR_CROSS, zorder=2))
        ax.add_patch(plt.Rectangle((c + r + 0.08, c + offset * 0.8), 0.35, zw, facecolor=CLR_CROSS, zorder=2))
        ax.add_patch(plt.Rectangle((c - r - 0.43, c - rw / 2 + offset * 0.8 + rw / 2), 0.35, zw, facecolor=CLR_CROSS, zorder=2))

    label_kw = dict(
        fontsize=11,
        fontweight="bold",
        color="#ffffffcc",
        ha="center",
        va="center",
        zorder=10,
        bbox=dict(facecolor="#00000066", boxstyle="round,pad=0.2", edgecolor="none"),
    )
    ax.text(c, H - 0.6, "BẮC ↓", **label_kw)
    ax.text(c, 0.6, "NAM ↑", **label_kw)
    ax.text(W - 0.6, c, "ĐÔNG ←", **label_kw)
    ax.text(0.6, c, "TÂY →", **label_kw)


class TrafficLightDisplay:
    """Vẽ 4 cụm đèn giao thông ở các góc ngã tư."""

    def __init__(self, ax):
        self.ax = ax
        c = CENTER
        r = INTER / 2
        positions = [
            (c - r - 0.65, c + r + 0.1, "N"),
            (c + r + 0.1, c + r + 0.1, "E"),
            (c + r + 0.1, c - r - 0.7, "S"),
            (c - r - 0.65, c - r - 0.7, "W"),
        ]
        box_w, box_h = 0.5, 1.0
        bulb_r = 0.13

        self.bulbs = {}
        for bx, by, direction in positions:
            box = mpatches.FancyBboxPatch(
                (bx, by),
                box_w,
                box_h,
                boxstyle="round,pad=0.04",
                facecolor="#222222",
                edgecolor="#555555",
                linewidth=1,
                zorder=6,
            )
            ax.add_patch(box)
            cx = bx + box_w / 2
            red_c = plt.Circle((cx, by + box_h * 0.78), bulb_r, color="#330000", zorder=7)
            yellow_c = plt.Circle((cx, by + box_h * 0.50), bulb_r, color="#332200", zorder=7)
            green_c = plt.Circle((cx, by + box_h * 0.22), bulb_r, color="#003300", zorder=7)
            ax.add_patch(red_c)
            ax.add_patch(yellow_c)
            ax.add_patch(green_c)
            self.bulbs[direction] = (red_c, yellow_c, green_c)

    def update(self, tl: "TrafficLight"):
        dim = {"red": "#330000", "yellow": "#332200", "green": "#003300"}
        bright = {"red": CLR_RED, "yellow": CLR_YELLOW, "green": CLR_GREEN}
        for direction, (r_c, y_c, g_c) in self.bulbs.items():
            phase = tl.get(direction)
            r_c.set_facecolor(bright["red"] if phase == "red" else dim["red"])
            y_c.set_facecolor(bright["yellow"] if phase == "yellow" else dim["yellow"])
            g_c.set_facecolor(bright["green"] if phase == "green" else dim["green"])


class HUD:
    """Hiển thị thống kê trạng thái xe và đèn trên góc màn hình."""

    def __init__(self, ax):
        self.ax = ax
        self.text = ax.text(
            0.01,
            0.99,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            color="white",
            fontfamily="sans-serif",
            bbox=dict(facecolor="#000000aa", edgecolor="#ffffff22", boxstyle="round,pad=0.5"),
            zorder=20,
        )
        self.phase_text = ax.text(
            0.99,
            0.99,
            "",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8.5,
            color="white",
            fontfamily="sans-serif",
            bbox=dict(facecolor="#000000aa", edgecolor="#ffffff22", boxstyle="round,pad=0.5"),
            zorder=20,
        )

    def update(self, tl: "TrafficLight", cars: list, frame: int):
        counts = {"N": 0, "S": 0, "E": 0, "W": 0}
        for car in cars:
            if car.state != "done":
                counts[car.direction] += 1

        self.text.set_text(
            f"Frame : {frame:>5}\n"
            f"Xe Bắc: {counts['N']:>3}\n"
            f"Xe Nam: {counts['S']:>3}\n"
            f"Xe Đông:{counts['E']:>3}\n"
            f"Xe Tây: {counts['W']:>3}\n"
            f"Tổng  : {sum(counts.values()):>3}"
        )

        ns = tl.get("N")
        ew = tl.get("E")
        ns_clr = {"green": CLR_GREEN, "yellow": CLR_YELLOW, "red": CLR_RED}
        self.phase_text.set_text(f"  B-N: {ns.upper():>6}  \n  Đ-T: {ew.upper():>6}  ")
        self.phase_text.set_color(ns_clr[ns])
