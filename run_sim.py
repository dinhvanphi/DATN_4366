import unittest.mock as mock
from traffic_sim import SimulationRL

def run_evaluation():
    # Mocking necessary components to avoid drawing
    with mock.patch('traffic_sim.Car.add_to_ax'), \
         mock.patch('traffic_sim.Car.remove'), \
         mock.patch('traffic_sim.Car._update_patch'):
        
        sim = SimulationRL(use_rl=True, training=False, model_path='ppo_model_last.pth')
        sim.ax = mock.MagicMock()
        sim.tld = mock.MagicMock()
        sim.hud = mock.MagicMock()
        
        counts = {'AGENT': 0, 'SOFT_FORCE': 0, 'HARD_FORCE': 0}
        
        for _ in range(5000):
            sim.step()
            reason = getattr(sim.tl, 'last_switch_reason', None)
            if reason in counts:
                counts[reason] += 1
                # Reset reason after counting to avoid double counting if it stays
                sim.tl.last_switch_reason = None
        
        total_switches = sum(counts.values())
        agent_percent = (counts['AGENT'] / total_switches * 100) if total_switches > 0 else 0
        
        print(f"Counts: {counts}")
        print(f"Percentage AGENT: {agent_percent:.2f}%")

if __name__ == "__main__":
    run_evaluation()
