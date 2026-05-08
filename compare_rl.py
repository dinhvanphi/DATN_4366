import numpy as np
import random
import torch
from traffic_sim import SimulationBenchmark

seeds = [0, 1]
max_frames = 2000
results = []
print(f"Seed | Throughput (%) | Avg Queue (%) | Avg Waiting (%)")
for seed in seeds:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    bf = SimulationBenchmark(use_rl=False, max_frames=max_frames)
    bf.run()
    sf = bf.get_results()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    br = SimulationBenchmark(use_rl=True, model_path='ppo_model.pth', max_frames=max_frames)
    br.run()
    sr = br.get_results()
    tpd = (sr['throughput_rate']-sf['throughput_rate'])/sf['throughput_rate']*100 if sf['throughput_rate']!=0 else 0
    qld = (sr['avg_queue_length']-sf['avg_queue_length'])/sf['avg_queue_length']*100 if sf['avg_queue_length']!=0 else 0
    wtd = (sr['avg_waiting_cars']-sf['avg_waiting_cars'])/sf['avg_waiting_cars']*100 if sf['avg_waiting_cars']!=0 else 0
    results.append((tpd, qld, wtd))
    print(f"{seed:<4} | {tpd:>14.2f}% | {qld:>13.2f}% | {wtd:>14.2f}%")
res = np.array(results); m = np.mean(res, axis=0); s = np.std(res, axis=0)
print(f"Mean | {m[0]:>14.2f}% | {m[1]:>13.2f}% | {m[2]:>14.2f}%")
print(f"Std  | {s[0]:>14.2f}% | {s[1]:>13.2f}% | {s[2]:>14.2f}%")
