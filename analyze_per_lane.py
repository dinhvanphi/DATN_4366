import random
import traffic_sim

spawn_per_frame = [] # list of (N, S, E, W)

original_spawn_cars = traffic_sim.SimulationBenchmark._spawn_cars

def patched_spawn_cars(self):
    # Track counts before call
    counts_before = {"N": 0, "S": 0, "E": 0, "W": 0}
    for c in self.cars:
        counts_before[c.direction] += 1
        
    original_spawn_cars(self)
    
    counts_after = {"N": 0, "S": 0, "E": 0, "W": 0}
    for c in self.cars:
        counts_after[c.direction] += 1
        
    spawned = (
        counts_after["N"] - counts_before["N"],
        counts_after["S"] - counts_before["S"],
        counts_after["E"] - counts_before["E"],
        counts_after["W"] - counts_before["W"]
    )
    spawn_per_frame.append(spawned)

traffic_sim.SimulationBenchmark._spawn_cars = patched_spawn_cars

random.seed(42)
sim = traffic_sim.SimulationBenchmark(use_rl=False, max_frames=2400)
sim.run()

window_size = 200
num_windows = 2400 // window_size

zero_lane_windows = 0

for i in range(num_windows):
    start = i * window_size
    end = (i + 1) * window_size
    window_data = spawn_per_frame[start:end]
    
    n_tot = sum(s[0] for s in window_data)
    s_tot = sum(s[1] for s in window_data)
    e_tot = sum(s[2] for s in window_data)
    w_tot = sum(s[3] for s in window_data)
    
    if i < 10:
        print(f"Window {i+1} ({start:4}-{end:4}): N={n_tot:2}, S={s_tot:2}, E={e_tot:2}, W={w_tot:2}")
        
    if any(count == 0 for count in [n_tot, s_tot, e_tot, w_tot]):
        zero_lane_windows += 1

print(f"\nWindows with at least one lane total = 0: {zero_lane_windows}")
