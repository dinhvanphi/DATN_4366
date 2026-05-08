import random
from traffic_sim import SimulationBenchmark

def run_benchmark():
    random.seed(42)
    max_frames = 2000
    
    setups = [
        {"name": "Fixed Timing", "params": {"use_rl": False}},
        {"name": "PPO (ppo_model.pth)", "params": {"use_rl": True, "model_path": "ppo_model.pth"}},
        {"name": "PPO (ppo_model_last.pth)", "params": {"use_rl": True, "model_path": "ppo_model_last.pth"}},
    ]
    
    results = {}
    
    for setup in setups:
        print(f"Running {setup['name']}...")
        benchmark = SimulationBenchmark(max_frames=max_frames, **setup['params'])
        benchmark.run()
        res = benchmark.get_results()
        results[setup['name']] = res
        print(f"Results for {setup['name']}:")
        print(f"  Cars Passed: {res['cars_passed']}")
        print(f"  Throughput Rate: {res['throughput_rate']:.4f}")
        print(f"  Avg Queue Length: {res['avg_queue_length']:.4f}")
        print(f"  Avg Waiting Cars: {res['avg_waiting_cars']:.4f}")
        print(f"  Phase Switches: {res['phase_switches']}")
        print("-" * 30)

    # Comparison
    dqn1 = results["PPO (ppo_model.pth)"]
    dqn2 = results["PPO (ppo_model_last.pth)"]
    
    print("\nComparison of DQN models:")
    best_queue = "ppo_model.pth" if dqn1['avg_queue_length'] < dqn2['avg_queue_length'] else "ppo_model_last.pth"
    best_waiting = "ppo_model.pth" if dqn1['avg_waiting_cars'] < dqn2['avg_waiting_cars'] else "ppo_model_last.pth"
    
    print(f"Better on Avg Queue Length: {best_queue}")
    print(f"Better on Avg Waiting Cars: {best_waiting}")

if __name__ == "__main__":
    run_benchmark()
