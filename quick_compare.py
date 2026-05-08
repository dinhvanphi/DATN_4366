import random
import numpy as np
import torch
from traffic_sim import SimulationRL

class MockDisplay:
    def update(self, *args, **kwargs): pass

def run_benchmark(mode, seed, max_frames=1200):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    sim = SimulationRL(use_rl=(mode == 'dqn'), training=False)
    sim.tld = MockDisplay()
    sim.hud = MockDisplay()
    sim.ax = None
    
    # Patch Car class
    from traffic_sim import Car
    original_add_to_ax = Car.add_to_ax
    original_remove = Car.remove
    original_patch = Car._update_patch
    Car.add_to_ax = lambda self, ax: None
    Car.remove = lambda self: None
    Car._update_patch = lambda self: None
    
    # Track positions to calculate speed
    prev_positions = {}
    
    total_waiting = 0
    total_queue = 0
    
    for _ in range(max_frames):
        sim.step()
        
        waiting = 0
        for c in sim.cars:
            pos = (c.x, c.y)
            if c.id in prev_positions:
                prev_pos = prev_positions[c.id]
                dist = np.sqrt((pos[0]-prev_pos[0])**2 + (pos[1]-prev_pos[1])**2)
                if dist < 0.001: # Threshold for waiting
                    waiting += 1
            prev_positions[c.id] = pos
            
        total_waiting += waiting
        total_queue += waiting
    
    # Restore
    Car.add_to_ax = original_add_to_ax
    Car.remove = original_remove
    Car._update_patch = original_patch
    
    return {
        'throughput': sim.total_cars_passed,
        'queue': total_queue / max_frames,
        'waiting': total_waiting / max_frames
    }

seeds = range(5)
results = []

print(f"Comparing Fixed vs DQN (max_frames=1200)...")
for s in seeds:
    fixed_res = run_benchmark('fixed', s)
    dqn_res = run_benchmark('dqn', s)
    
    diff = {
        'throughput': (dqn_res['throughput'] - fixed_res['throughput']) / max(1, fixed_res['throughput']) * 100,
        'queue': (dqn_res['queue'] - fixed_res['queue']) / max(0.1, fixed_res['queue']) * 100,
        'waiting': (dqn_res['waiting'] - fixed_res['waiting']) / max(0.1, fixed_res['waiting']) * 100
    }
    results.append(diff)
    print(f"Seed {s}: Throughput={diff['throughput']:+5.1f}%, Queue={diff['queue']:+5.1f}%, Wait={diff['waiting']:+5.1f}%")

mean_diff = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
print(f"\nMean Improvement:")
print(f"Throughput: {mean_diff['throughput']:+5.1f}%")
print(f"Queue Length: {mean_diff['queue']:+5.1f}%")
print(f"Waiting Cars: {mean_diff['waiting']:+5.1f}%")
