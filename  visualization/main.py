import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('macosx')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting disabled.")
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from classical.local.hill_climbing import HillClimbing
from nature_inspire.physic_based.simulated_annealing.simulated_annealing import SA
from nature_inspire.evolution_based.differential_evolution.differential_evolution import DE
from nature_inspire.biology_based.particle_swarm_optimization.particle_swarm_optimization import PSO
from nature_inspire.biology_based.artificial_bee_colony.artificial_bee_colony import ABC
from nature_inspire.biology_based.firefly_algorithm.firefly_algorithm import FA
from nature_inspire.biology_based.cuckoo_search.cuckoo_search import CS
from nature_inspire.human_based.teaching_learning_based_optimization.teaching_learning_based_optimization import TLBO
from problems.problem import get_problem


class InteractiveVisualizer:
    def __init__(self, max_iter=50):
        self.max_iter = max_iter
        self.dim = 2  # Số chiều không gian tìm kiếm (Input là 2D -> Đồ thị sẽ là 3D)
        self.histories = {}
        self.alg_names = ["HC", "SA", "DE", "PSO", "ABC", "FA", "CS", "TLBO"]

    def setup_problem(self, func_name):
        self.func_name = func_name
        prob = get_problem(self.func_name)
        self.func = prob["func"]
        self.lb = prob["lb"]
        self.ub = prob["ub"]

    def run_algorithm(self, alg_name):
        print(f"\n>>> Đang chạy {alg_name} trên hàm {self.func_name.upper()}...")

        if alg_name == "HC":
            solver = HillClimbing(self.lb, self.ub, self.dim, step_size=0.5)
            solver.set_optimization_function(self.func)
            solver.run(self.max_iter)
            self.histories["HC"] = getattr(solver, "history", [])

        elif alg_name == "SA":
            solver = SA(bounds=[self.lb, self.ub], function=self.func, dim=self.dim,
                        T=100.0, alpha=0.95, stopping_T=0.001, stopping_iter=self.max_iter)
            solver.run(times=1)
            self.histories["SA"] = getattr(solver, "history", [])

        elif alg_name == "DE":
            solver = DE(self.func, bounds=[self.lb, self.ub], dim=self.dim, pop_size=30, max_gen=self.max_iter)
            solver.run(self.max_iter)
            self.histories["DE"] = getattr(solver, "history", [])

        elif alg_name == "PSO":
            solver = PSO(function=self.func, dimension=self.dim, ranges=[self.lb, self.ub], swarm_size=30,
                         max_interation=self.max_iter)
            solver.run()
            self.histories["PSO"] = getattr(solver, "history", [])

        elif alg_name == "ABC":
            solver = ABC(function=self.func, ranges=[self.lb, self.ub], dimension=self.dim, swarm_size=30,
                         max_iteration=self.max_iter)
            solver.run()
            self.histories["ABC"] = getattr(solver, "history", [])

        elif alg_name == "FA":
            solver = FA(func_name=self.func_name, pop_size=30, dim=self.dim, max_iter=self.max_iter)
            solver.run()
            self.histories["FA"] = getattr(solver, "history", [])

        elif alg_name == "CS":
            solver = CS(func_name=self.func_name, pop_size=30, dim=self.dim, max_iter=self.max_iter)
            solver.run()
            self.histories["CS"] = getattr(solver, "history", [])

        elif alg_name == "TLBO":
            solver = TLBO(self.lb, self.ub, self.dim, population_size=30)
            solver.set_optimization_function(self.func)
            solver.run(self.max_iter)
            self.histories["TLBO"] = getattr(solver, "history", [])

    def visualize_single_3d(self, alg_name, interval=20):
        if not MATPLOTLIB_AVAILABLE: return

        hist = self.histories.get(alg_name, [])
        if not hist:
            print(f"Không có dữ liệu history cho {alg_name}.")
            return

        fig = plt.figure(figsize=(15, 7.5))
        ax = fig.add_subplot(111, projection='3d')

        # Tạo lưới mặt phẳng (Độ phân giải 50x50 là đủ đẹp và render siêu nhanh)
        x = np.linspace(self.lb, self.ub, 50)
        y = np.linspace(self.lb, self.ub, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.func([X[i, j], Y[i, j]])

        # Vẽ Surface (mặt nhấp nhô) 1 lần duy nhất để tối ưu GPU/CPU
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolor='none')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Objective Value')

        # Chọn màu và marker
        marker_dict = {"HC": "o", "SA": "o", "DE": "X", "PSO": "o", "ABC": "P", "FA": "*", "CS": "v", "TLBO": "s"}
        color_dict = {"HC": "red", "SA": "orange", "DE": "black", "PSO": "red", "ABC": "red", "FA": "yellow",
                      "CS": "cyan", "TLBO": "lime"}

        # Khởi tạo hạt scatter 3D trống
        scatter = ax.scatter([], [], [],
                             c=color_dict.get(alg_name, 'red'),
                             marker=marker_dict.get(alg_name, 'o'),
                             edgecolors='black', s=50, depthshade=True)

        ax.set_xlim(self.lb, self.ub)
        ax.set_ylim(self.lb, self.ub)
        ax.set_zlim(np.min(Z), np.max(Z))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Objective (Z)')

        def update(frame):
            positions = np.array(hist[frame])
            if positions.ndim == 1:
                positions = positions.reshape(1, -1)

            xs = positions[:, 0]
            ys = positions[:, 1]
            zs = np.array([self.func([px, py]) for px, py in positions])

            # Hack render siêu tốc: Cập nhật trực tiếp _offsets3d
            scatter._offsets3d = (xs, ys, zs)

            # Hiệu ứng xoay Camera Cinematic
            ax.view_init(elev=30, azim=45 + frame * 1.5)

            ax.set_title(
                f"[3D] Algorithm: {alg_name} | Function: {self.func_name.upper()} | Iter: {frame}/{len(hist) - 1}")
            return scatter,

        anim = animation.FuncAnimation(fig, update, frames=len(hist), interval=interval, blit=False, repeat=False)
        plt.show()

    def visualize_all_3d(self, interval=20):
        if not MATPLOTLIB_AVAILABLE: return

        fig = plt.figure(figsize=(15, 7.5))
        fig.suptitle(f"SO SÁNH 8 THUẬT TOÁN TRÊN HÀM {self.func_name.upper()} (3D)", fontsize=16, fontweight='bold')

        # Giảm độ phân giải lưới xuống 35x35 để M2 Pro kéo mượt 8 cửa sổ 3D cùng lúc
        x = np.linspace(self.lb, self.ub, 35)
        y = np.linspace(self.lb, self.ub, 35)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.func([X[i, j], Y[i, j]])

        axes = []
        scatters = []
        colors = ['red', 'orange', 'black', 'blue', 'purple', 'yellow', 'cyan', 'lime']
        markers = ['o', 'o', 'X', 'o', 'P', '*', 'v', 's']

        # Khởi tạo 8 khung hình 3D
        for idx, alg in enumerate(self.alg_names):
            ax = fig.add_subplot(2, 4, idx + 1, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.4, edgecolor='none')
            ax.set_title(alg, fontweight='bold')
            ax.set_xlim(self.lb, self.ub)
            ax.set_ylim(self.lb, self.ub)
            ax.set_zlim(np.min(Z), np.max(Z))

            # Tắt số ở các trục để đỡ rối mắt khi xem 8 đồ thị
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            scatter = ax.scatter([], [], [], c=colors[idx], marker=markers[idx], edgecolors='black', s=30,
                                 depthshade=True)
            axes.append(ax)
            scatters.append(scatter)

        def update(frame):
            for idx, alg in enumerate(self.alg_names):
                hist = self.histories.get(alg, [])
                if not hist: continue

                f = min(frame, len(hist) - 1)

                axes[idx].set_title(f"{alg} - Vòng: {f}/{len(hist) - 1}", fontweight='bold', fontsize=11)
                try:
                    positions = np.array(hist[f], dtype=float)
                except Exception:
                    continue

                if positions.ndim == 0 or positions.size == 0:
                    continue
                if positions.ndim == 1:
                    positions = positions.reshape(1, -1)
                if positions.shape[1] < 2:
                    continue

                xs = positions[:, 0]
                ys = positions[:, 1]
                zs = np.array([self.func([px, py]) for px, py in positions])

                scatters[idx]._offsets3d = (xs, ys, zs)
                axes[idx].view_init(elev=35, azim=30 + frame * 1.0)

            return scatters

        max_frames = max([len(h) for h in self.histories.values() if h] + [0])
        if max_frames == 0: return

        anim = animation.FuncAnimation(fig, update, frames=max_frames, interval=interval, blit=False, repeat=False)
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90, wspace=0.1, hspace=0.15)
        plt.show()


def main():
    visualizer = InteractiveVisualizer(max_iter=50)

    # 1. Menu chọn Function
    functions = ["sphere", "rastrigin", "rosenbrock", "griewank", "ackley"]
    print("\n" + "=" * 45)
    print(" DANH SÁCH HÀM MỤC TIÊU (CONTINUOUS 3D)")
    print("=" * 45)
    for i, f in enumerate(functions):
        print(f" {i + 1}. {f.capitalize()}")

    try:
        f_choice = int(input(f"\nNhập số thứ tự hàm muốn chạy (1-{len(functions)}): ")) - 1
        if f_choice < 0 or f_choice >= len(functions):
            print("Lựa chọn không hợp lệ. Mặc định chọn Sphere.")
            f_choice = 0
    except ValueError:
        print("Đầu vào không hợp lệ. Mặc định chọn Sphere.")
        f_choice = 0

    selected_func = functions[f_choice]
    visualizer.setup_problem(selected_func)

    # 2. Menu chọn Thuật toán
    print("\n" + "=" * 45)
    print(" DANH SÁCH THUẬT TOÁN")
    print("=" * 45)
    for i, alg in enumerate(visualizer.alg_names):
        print(f" {i + 1}. {alg}")
    print(f" {len(visualizer.alg_names) + 1}. Chạy TẤT CẢ (Lưới 2x4 3D - Nặng hơn)")

    try:
        a_choice = int(input(f"\nNhập số thứ tự thuật toán muốn chạy (1-{len(visualizer.alg_names) + 1}): ")) - 1
    except ValueError:
        print("Đầu vào không hợp lệ. Mặc định chạy HC.")
        a_choice = 0

    # 3. Thực thi với Interval = 20 (Đạt ~50FPS siêu mượt)
    if a_choice == len(visualizer.alg_names):
        # Chạy tất cả thuật toán rồi render 3D
        for alg in visualizer.alg_names:
            visualizer.run_algorithm(alg)
        visualizer.visualize_all_3d(interval=5)
    elif 0 <= a_choice < len(visualizer.alg_names):
        # Chạy 1 thuật toán cụ thể rồi render 3D Cinematic
        selected_alg = visualizer.alg_names[a_choice]
        visualizer.run_algorithm(selected_alg)
        visualizer.visualize_single_3d(selected_alg, interval=20)
    else:
        print("Lựa chọn nằm ngoài danh sách. Thoát chương trình.")


if __name__ == "__main__":
    main()