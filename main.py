from nature_inspire.physic_based.simulated_annealing.simulated_annealing_tsp import SimulatedAnnellingTsp
from nature_inspire.physic_based.simulated_annealing.simulated_annealing_cont import SimulatedAnnealingContinuous
import matplotlib.pyplot as plt
import random
from nature_inspire.continuous_functions import sphere_function, rastrigin_function
import numpy as np

def plot_3d_optimization(sa_instance):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # A. Vẽ bề mặt địa hình (Landscape) Rastrigin
    # Tạo lưới điểm từ -5.12 đến 5.12
    x = np.linspace(sa_instance.ranges[0], sa_instance.ranges[1], 100)
    y = np.linspace(sa_instance.ranges[0], sa_instance.ranges[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Tính Z cho bề mặt (Dùng công thức Rastrigin dạng vector cho numpy)
    A = 10
    Z = (A * 2 + (X**2 - A * np.cos(2 * np.pi * X)) + 
                 (Y**2 - A * np.cos(2 * np.pi * Y)))

    # Vẽ bề mặt (mờ đi một chút bằng alpha=0.3 để nhìn thấy đường đi bên trong)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', alpha=0.4)

    # B. Vẽ đường đi của thuật toán (Trajectory)
    history = np.array(sa_instance.coords_history) # Chuyển sang numpy cho dễ lấy cột
    path_x = history[:, 0]
    path_y = history[:, 1]
    path_z = sa_instance.energy_list # Z chính là giá trị hàm (Fitness)

    # Vẽ đường nối các điểm
    ax.plot(path_x, path_y, path_z, color='red', linewidth=1, label='SA Path', zorder=10)
    
    # Đánh dấu điểm bắt đầu (Xanh lá) và Kết thúc (Đỏ đậm)
    ax.scatter(path_x[0], path_y[0], path_z[0], color='green', s=50, label='Start')
    ax.scatter(path_x[-1], path_y[-1], path_z[-1], color='black', s=50, label='End')
    best_x = sa_instance.best_coords[0]
    best_y = sa_instance.best_coords[1]
    best_z = sa_instance.best_result
    ax.scatter(best_x, best_y, best_z, color='blue', s=100, marker='*', label='Best Found', zorder=20)
    ax.set_title('3D')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Function Value (Energy)')
    ax.legend()
    plt.show()

# --- 4. Main ---
if __name__ == "__main__":
    # QUAN TRỌNG: Để vẽ 3D, dim phải bằng 2
    sa = SimulatedAnnealingContinuous(
        ranges=[-5.12, 5.12],
        function=sphere_function,
        dim=2,              # CHỈNH VỀ 2 CHIỀU ĐỂ VẼ 3D`    `
        T=1000,           
        alpha=0.99,       
        step_size=0.3,    
        stopping_iter=5000 
    )
    
    print("Runnings ... ")
    sa.batch_annealing()
    
    print("-" * 30)
    print(f"Final Result: {sa.cur_result:.6f}")
    print(f"Best Result:  {sa.best_result:.6f}")
    print(f"Best Coords:  {[round(x, 3) for x in sa.best_coords]}")
    
    # Gọi hàm vẽ 3D
    print("Drawing 3D...")
    plot_3d_optimization(sa)
