import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark_results():
    """
    Vẽ biểu đồ so sánh kết quả trung bình 5 seeds giữa Fixed Timing và PPO.
    Dữ liệu được lấy từ kết quả chạy thực tế gần nhất.
    """
    labels = ['Fixed Timing', 'PPO (Best)']
    
    # Dữ liệu từ log trung bình 5 seed
    throughput = [7.19, 7.54]        # Tăng
    waiting_cars = [24.42, 24.01]    # Giảm
    phase_switches = [14.0, 12.6]    # Giảm (12.6 là trung bình của 12,12,13,13,12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('So sánh Hiệu năng Điều khiển Giao thông (Trung bình 5 Seeds)', fontsize=16, fontweight='bold', y=1.05)

    colors_fixed = '#ff6b6b'  # Màu đỏ nhạt cho Fixed
    colors_ppo = '#4ecdc4'    # Màu xanh ngọc cho PPO

    # Biểu đồ 1: Throughput (Lưu lượng xe qua ngã tư)
    axes[0].bar(labels, throughput, color=[colors_fixed, colors_ppo], width=0.5)
    axes[0].set_title('Lưu lượng xe (Throughput)\n[Cao hơn là tốt hơn]', pad=15)
    axes[0].set_ylabel('Số xe / phút')
    axes[0].set_ylim(0, max(throughput) * 1.2)
    for i, v in enumerate(throughput):
        axes[0].text(i, v + 0.1, f"{v:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Biểu đồ 2: Waiting Cars (Số xe chờ trung bình)
    axes[1].bar(labels, waiting_cars, color=[colors_fixed, colors_ppo], width=0.5)
    axes[1].set_title('Số xe chờ Trung bình\n[Thấp hơn là tốt hơn]', pad=15)
    axes[1].set_ylabel('Số xe')
    axes[1].set_ylim(0, max(waiting_cars) * 1.2)
    for i, v in enumerate(waiting_cars):
        axes[1].text(i, v + 0.3, f"{v:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Biểu đồ 3: Phase Switches (Số lần đổi đèn)
    axes[2].bar(labels, phase_switches, color=[colors_fixed, colors_ppo], width=0.5)
    axes[2].set_title('Số lần chuyển pha đèn\n[Tránh lãng phí đèn vàng/đỏ]', pad=15)
    axes[2].set_ylabel('Số lần')
    axes[2].set_ylim(0, max(phase_switches) * 1.2)
    for i, v in enumerate(phase_switches):
        axes[2].text(i, v + 0.3, f"{v:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Định dạng lại lưới và layout
    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("Đã lưu biểu đồ thành công vào file 'benchmark_results.png'")
    plt.show()

if __name__ == '__main__':
    plot_benchmark_results()
