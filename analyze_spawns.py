import random
import traffic_sim

# Match the globals from traffic_sim if needed
# traffic_sim might have imports like CAR_COLORS, etc.
# But we just want to monkeypatch SimulationBenchmark._spawn_cars

spawn_counts = []

original_spawn_cars = traffic_sim.SimulationBenchmark._spawn_cars

def patched_spawn_cars(self):
    # Track counts before call
    # In SimulationBenchmark, directions are N, S, E, W
    # Total cars is self.cars
    
    pre_counts = {"NS": 0, "EW": 0}
    for c in self.cars:
        if c.direction in ["N", "S"]: pre_counts["NS"] += 1
        else: pre_counts["EW"] += 1
        
    original_spawn_cars(self)
    
    post_counts = {"NS": 0, "EW": 0}
    for c in self.cars:
        if c.direction in ["N", "S"]: post_counts["NS"] += 1
        else: post_counts["EW"] += 1
        
    # Careful: cars might have been removed? 
    # In SimulationBenchmark.run(), _spawn_cars() is called before _remove_done()
    # So counts should only increase.
    
    spawned_ns = post_counts["NS"] - pre_counts["NS"]
    spawned_ew = post_counts["EW"] - pre_counts["EW"]
    
    spawn_counts.append((spawned_ns, spawned_ew))

traffic_sim.SimulationBenchmark._spawn_cars = patched_spawn_cars

# No seed specified but usually good for reproducibility
random.seed(42)

sim = traffic_sim.SimulationBenchmark(use_rl=False, max_frames=2200)
sim.run()

window_size = 200
num_windows = 2200 // window_size

ns_wins = 0
ew_wins = 0

for i in range(num_windows):
    start = i * window_size
    end = (i + 1) * window_size
    window_data = spawn_counts[start:end]
    
    total_ns = sum(s[0] for s in window_data)
    total_ew = sum(s[1] for s in window_data)
    
    dominant = "NS > EW" if total_ns > total_ew else "EW > NS" if total_ew > total_ns else "Equal"
    
    if total_ns > total_ew: ns_wins += 1
    elif total_ew > total_ns: ew_wins += 1
    
    if i < 8:
        print(f"Window {i+1} ({start}-{end}): NS={total_ns}, EW={total_ew} -> {dominant}")

print(f"\nSummary: NS-dominant: {ns_wins}, EW-dominant: {ew_wins}")
