import argparse
import os
import random
import sys

import matplotlib.pyplot as plt

from traffic_simulation import SimulationBenchmark, build_benchmark_spawn_schedule


DEFAULT_SINGLE_SEED = [46]
DEFAULT_MULTI_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_SEED = 46
DEFAULT_MODEL_PLOTS = [
    {
        "algorithm": "ppo",
        "label": "PPO(best)",
        "model_path": "ppo_model.pth",
        "output": "ppo_seed46_benchmark.png",
    },
    {
        "algorithm": "dqn",
        "label": "DQN",
        "model_path": "dqn_model.pth",
        "output": "dqn_seed46_benchmark.png",
    },
]


def run_one(seed, max_frames, model_path, algorithm):
    spawn_schedule = build_benchmark_spawn_schedule(seed, max_frames)

    fixed = SimulationBenchmark(
        use_rl=False,
        max_frames=max_frames,
        rng=random.Random(seed),
        spawn_schedule=spawn_schedule,
    )
    fixed.run()

    model = SimulationBenchmark(
        use_rl=True,
        model_path=model_path,
        max_frames=max_frames,
        rng=random.Random(seed),
        spawn_schedule=spawn_schedule,
        algorithm=algorithm,
    )
    model.run()

    return fixed.get_results(), model.get_results()


def mean(results, key):
    return sum(item[key] for item in results) / max(len(results), 1)


def run_benchmark(seeds, max_frames, model_path, algorithm):
    fixed_results = []
    model_results = []

    for seed in seeds:
        fixed, model = run_one(seed, max_frames, model_path, algorithm)
        fixed_results.append(fixed)
        model_results.append(model)

    return {
        "Throughput": {
            "fixed": mean(fixed_results, "throughput_rate"),
            "model": mean(model_results, "throughput_rate"),
            "unit": "xe/phút",
            "higher_is_better": True,
        },
        "Hàng đợi TB": {
            "fixed": mean(fixed_results, "avg_queue_length"),
            "model": mean(model_results, "avg_queue_length"),
            "unit": "xe",
            "higher_is_better": False,
        },
        "Xe chờ TB": {
            "fixed": mean(fixed_results, "avg_waiting_cars"),
            "model": mean(model_results, "avg_waiting_cars"),
            "unit": "xe",
            "higher_is_better": False,
        },
        "Thời gian chờ đỏ TB": {
            "fixed": mean(fixed_results, "avg_red_wait_time"),
            "model": mean(model_results, "avg_red_wait_time"),
            "unit": "frame/xe",
            "higher_is_better": False,
        },
    }


def improvement_percent(fixed, model, higher_is_better):
    if fixed == 0:
        return 0.0
    if higher_is_better:
        return (model - fixed) / fixed * 100
    return (fixed - model) / fixed * 100


def format_change(metric):
    change = improvement_percent(
        metric["fixed"],
        metric["model"],
        metric["higher_is_better"],
    )
    direction = "tốt hơn" if change >= 0 else "xấu hơn"
    return f"{abs(change):.1f}% {direction}"


def plot_benchmark_results(model_path="ppo_model.pth", seeds=None, max_frames=2000,
                           output="benchmark_results.png", algorithm="ppo", model_label=None):
    if seeds is None:
        seeds = DEFAULT_SINGLE_SEED
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    algorithm = algorithm.lower()
    model_label = model_label or algorithm.upper()
    data = run_benchmark(seeds, max_frames, model_path, algorithm)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"So sánh Fixed Timing và {model_label}",
        fontsize=16,
        fontweight="bold",
    )

    labels = ["Fixed Timing", model_label]
    colors = ["#ff6b6b", "#4ecdc4"]

    for ax, (name, metric) in zip(axes.flat, data.items()):
        values = [metric["fixed"], metric["model"]]
        y_max = max(max(values), 1.0)
        bars = ax.bar(labels, values, color=colors, width=0.55)
        ax.set_title(name, fontweight="bold", pad=10)
        ax.set_ylabel(metric["unit"])
        ax.set_ylim(0, y_max * 1.18)
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.03,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.text(
            0.5,
            -0.18,
            f"{model_label}: {format_change(metric)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#333333",
        )

    footer = (
        f"Seed: {seeds[0]} | {max_frames} frames"
        if len(seeds) == 1
        else f"Trung bình trên seeds: {', '.join(str(seed) for seed in seeds)} | {max_frames} frames/seed"
    )
    fig.text(0.5, 0.02, footer, ha="center", fontsize=10, color="#444444")

    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã lưu biểu đồ vào '{output}'")
    for name, metric in data.items():
        print(
            f"{name}: Fixed={metric['fixed']:.2f}, "
            f"{model_label}={metric['model']:.2f}, {format_change(metric)}"
        )


def plot_default_seed46_models(max_frames=2000, seed=DEFAULT_SEED):
    for config in DEFAULT_MODEL_PLOTS:
        if not os.path.exists(config["model_path"]):
            print(f"Bỏ qua {config['label']}: không tìm thấy {config['model_path']}")
            continue
        plot_benchmark_results(
            model_path=config["model_path"],
            seeds=[seed],
            max_frames=max_frames,
            output=config["output"],
            algorithm=config["algorithm"],
            model_label=config["label"],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ benchmark Fixed Timing vs PPO/DQN.")
    parser.add_argument("--model", default=None, help="Đường dẫn model. Nếu bỏ trống sẽ vẽ cả PPO và DQN ở seed 46.")
    parser.add_argument("--algorithm", default=None, choices=["ppo", "dqn"], help="Thuật toán của model.")
    parser.add_argument("--frames", type=int, default=2000, help="Số frame benchmark mỗi seed.")
    parser.add_argument("--output", default=None, help="File ảnh đầu ra khi vẽ một model.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed dùng khi không truyền --seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Danh sách seed khi vẽ một model.")
    parser.add_argument("--multi-seed", action="store_true", help="Dùng seeds 42..46, khớp option 8.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    single_model_requested = any(
        opt in sys.argv[1:]
        for opt in ("--model", "--algorithm", "--output", "--seeds", "--multi-seed")
    )

    if not single_model_requested:
        plot_default_seed46_models(max_frames=args.frames, seed=args.seed)
    else:
        algorithm = args.algorithm or "ppo"
        model_path = args.model or ("dqn_model.pth" if algorithm == "dqn" else "ppo_model.pth")
        output = args.output or (
            "dqn_benchmark_results.png" if algorithm == "dqn" else "benchmark_results.png"
        )
        seeds = args.seeds
        if seeds is None:
            seeds = DEFAULT_MULTI_SEEDS if args.multi_seed else [args.seed]

        plot_benchmark_results(
            model_path=model_path,
            seeds=seeds,
            max_frames=args.frames,
            output=output,
            algorithm=algorithm,
        )
