import unittest.mock as mock
from traffic_sim import SimulationRL

def run_evaluation():
    with mock.patch('traffic_sim.Car.add_to_ax'), \
         mock.patch('traffic_sim.Car.remove'), \
         mock.patch('traffic_sim.Car._update_patch'):
        
        sim = SimulationRL(use_rl=True, training=False, model_path='ppo_model_last.pth')
        sim.ax = mock.MagicMock()
        sim.tld = mock.MagicMock()
        sim.hud = mock.MagicMock()
        
        counts = {'AGENT': 0, 'SOFT_FORCE': 0, 'HARD_FORCE': 0, 'FORCE': 0, 'HOLD': 0}
        
        for _ in range(5000):
            sim.step()
            reason = getattr(sim.tl, 'last_switch_reason', None)
            if reason in counts:
                counts[reason] += 1
                sim.tl.last_switch_reason = None
        
        print(f"Counts: {counts}")

if __name__ == "__main__":
    run_evaluation()
