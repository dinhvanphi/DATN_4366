"""
traffic_constants.py
--------------------
Chứa toàn bộ hằng số và cấu hình hình học của mô hình giao thông.
"""

W, H = 24, 24
ROAD_W = 1.6
INTER = ROAD_W * 2
CENTER = W / 2

V_CRUISE = 0.13
V_SLOW = 0.04
STOP_GAP = 0.55
SAFE_GAP = 0.7
SPAWN_BLOCK_DIST = 0.8

# Giảm GREEN_TIME để Fixed Timing switch nhiều hơn → so sánh công bằng hơn
GREEN_TIME = 120
YELLOW_TIME = 20   # Giảm từ 40 → giảm overhead mỗi lần PPO switch
ALL_RED_TIME = 4
RED_TIME = GREEN_TIME + YELLOW_TIME + ALL_RED_TIME

CAR_W, CAR_H = 0.55, 0.9

CLR_ROAD = "#2b2b2b"
CLR_SIDEWALK = "#4a4a4a"
CLR_LINE = "#ffffff"
CLR_CROSS = "#888888"
CLR_GREEN = "#00e676"
CLR_YELLOW = "#ffea00"
CLR_RED = "#ff1744"

CAR_COLORS = {
    "N": ["#42a5f5", "#1565c0", "#29b6f6", "#0277bd", "#4fc3f7"],
    "S": ["#ef5350", "#b71c1c", "#e53935", "#c62828", "#ff8a80"],
    "E": ["#66bb6a", "#2e7d32", "#43a047", "#1b5e20", "#a5d6a7"],
    "W": ["#ffa726", "#e65100", "#fb8c00", "#bf360c", "#ffcc80"],
}

STOP_LINE = {
    "N": CENTER + INTER / 2,
    "S": CENTER - INTER / 2,
    "E": CENTER + INTER / 2,
    "W": CENTER - INTER / 2,
}

# Xác suất sinh xe mỗi frame (60 fps)
# KỊCH BẢN GIỜ CAO ĐIỂM (Bất đối xứng)
# Trục Bắc-Nam (Đường chính): Xe rất đông
# Trục Đông-Tây (Đường phụ): Xe rất vắng
SPAWN_PROBS = {
    "N": 0.018,
    "S": 0.018,
    "E": 0.004,
    "W": 0.004
}

LANE_X = {
    "N": CENTER - ROAD_W * 0.5,
    "S": CENTER + ROAD_W * 0.5,
}

LANE_Y = {
    "E": CENTER + ROAD_W * 0.5,
    "W": CENTER - ROAD_W * 0.5,
}