"""
traffic_sim.py
--------------
Lớp bọc tương thích cho toàn bộ mô phỏng giao thông.
File này giữ lại API cũ để các script hiện có vẫn import được như trước,
nhưng phần triển khai đã được tách ra thành các module nhỏ hơn.
"""

from traffic_constants import *
from traffic_dqn import *
from traffic_entities import *
from traffic_ppo import *
from traffic_rl import *
from traffic_simulation import *


if __name__ == "__main__":
    main()