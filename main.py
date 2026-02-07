from nature_inspire.physic_based.simulated_annealing.travelling_sale_man import SimulatedAnnellingTsp
from nature_inspire.physic_based.simulated_annealing.continuos_functions import SimulatedAnnealingContinuous
import matplotlib.pyplot as plt
import random
import nature_inspire.continuous_functions
import numpy as np

def plot_3d_optimization(sa_instance):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- A. TẠO LƯỚI ĐIỂM DỰA TRÊN RANGE CỦA SA ---
    # Lấy range từ cấu hình SA (ví dụ -5.12 đến 5.12)
    x_range = np.linspace(sa_instance.ranges[0], sa_instance.ranges[1], 100)
    y_range = np.linspace(sa_instance.ranges[0], sa_instance.ranges[1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # --- B. TÍNH Z DỰA TRÊN HÀM TRUYỀN VÀO (QUAN TRỌNG) ---
    # Thay vì viết công thức cứng, ta chạy vòng lặp để gọi hàm sa_instance.function
    # Cách này đảm bảo hàm nào trong Main thì vẽ đúng hàm đó.
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Tạo input [x, y] giống như cách thuật toán chạy
            coords = [X[i, j], Y[i, j]] 
            # Gọi hàm được lưu trong instance SA
            Z[i, j] = sa_instance.function(coords)

    # Vẽ bề mặt (Dùng cmap 'coolwarm' hoặc 'jet' để nhìn rõ độ sâu)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm', 
                           edgecolor='none', alpha=0.6)
    
    # Thêm thanh màu (colorbar) để dễ nhìn giá trị
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # --- C. VẼ ĐƯỜNG ĐI (TRAJECTORY) ---
    history = np.array(sa_instance.coords_history) 
    path_x = history[:, 0]
    path_y = history[:, 1]
    
    # Lấy giá trị hàm fitness tương ứng từ history
    path_z = sa_instance.energy_list 

    # Vẽ đường nối
    ax.plot(path_x, path_y, path_z, color='black', linewidth=1.5, label='SA Path', zorder=10)
    
    # Điểm đầu (Xanh lá)
    ax.scatter(path_x[0], path_y[0], path_z[0], color='green', s=60, label='Start')
    # Điểm cuối (Tím)
    ax.scatter(path_x[-1], path_y[-1], path_z[-1], color='magenta', s=60, label='End')
    
    # Điểm tốt nhất tìm thấy (Sao vàng)
    best_x = sa_instance.best_coords[0]
    best_y = sa_instance.best_coords[1]
    best_z = sa_instance.best_result
    ax.scatter(best_x, best_y, best_z, color='yellow', s=150, marker='*', edgecolors='black', label='Best Found', zorder=20)

    # Thiết lập tiêu đề và nhãn
    func_name = getattr(sa_instance.function, '__name__', 'Unknown Function')
    ax.set_title(f'Optimization Landscape: {func_name}')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Energy (Cost)')
    ax.legend()
    plt.show()
# --- 4. Main ---
if __name__ == "__main__":
    # QUAN TRỌNG: Để vẽ 3D, dim phải bằng 2
    sa = SimulatedAnnealingContinuous(
        ranges=[-5.12, 5.12],
        function=nature_inspire.continuous_functions.rosenbrock_function,
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
