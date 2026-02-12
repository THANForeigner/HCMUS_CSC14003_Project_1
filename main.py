import nature_inspire.physic_based.simulated_annealing as sa
import nature_inspire.biology_based.particle_swarm_optimization.continous_functions as pso
import nature_inspire.biology_based.artificial_bee_colony.continuous_functions as abc
import matplotlib.pyplot as plt
import random
import nature_inspire.continuous_functions as cf
import numpy as np

if __name__ == "__main__":
    # 1. CẤU HÌNH THAM SỐ
    DIMENSION = 2           
    BOUNDS = [-5.12, 5.12]  
    SWARM_SIZE = 50         
    MAX_ITER = 100          
    LIMIT = 50  # Tham số riêng của ABC: Số lần thử tối đa trước khi vứt bỏ nguồn thức ăn
    
    # Chọn hàm mục tiêu (Rastrigin hoặc Sphere...)
    function = cf.rastrigin_function
    # function = lambda x: np.sum(np.array(x)**2) # Ví dụ hàm Sphere đơn giản nếu cần test
    
    print(f"--- Starting ABC test with Function ({DIMENSION} dimensions) ---")
    
    # 2. KHỞI TẠO OPTIMIZER (ABC)
    optimizer = abc.ArtificialBeeColony(
        function=function,
        ranges=BOUNDS,
        dimension=DIMENSION,
        swarm_size=SWARM_SIZE,
        limit=LIMIT,           # Thêm tham số limit
        max_iteration=MAX_ITER 
    )
    
    # 3. CHẠY THUẬT TOÁN
    # Hàm này trả về đối tượng con ong tốt nhất (best_bee)
    optimizer.artificial_bee_colony()
    best_bee = optimizer.best_bee
    
    # 4. XỬ LÝ KẾT QUẢ
    # Vì ABC tối ưu hóa Fitness (1/(1+f)), ta cần tính lại giá trị hàm mục tiêu (Cost)
    # để so sánh (Cost càng về 0 càng tốt)
    best_cost = function(best_bee.coords)
    
    print("\n--- RESULTS ---")
    print(f"Best Fitness (ABC internal): {best_bee.fitness:.10f}")
    print(f"Best Function Cost (Real Min): {best_cost:.10f}")
    
    best_pos_rounded = np.round(best_bee.coords, 5)
    print(f"Global Best Position: {best_pos_rounded}")

    # 5. VẼ BIỂU ĐỒ (CHỈ KHI DIMENSION = 2)
    if DIMENSION == 2:
        print("\nDrawing...")
        # Tạo lưới dữ liệu để vẽ Contour
        x = np.linspace(BOUNDS[0], BOUNDS[1], 100)
        y = np.linspace(BOUNDS[0], BOUNDS[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Tính giá trị hàm tại từng điểm lưới
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Lưu ý: function có thể yêu cầu list hoặc numpy array tùy cách bạn define
                Z[i, j] = function(np.array([X[i, j], Y[i, j]]))

        plt.figure(figsize=(10, 8))
        
        # Vẽ đường đồng mức (Contour plot)
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Function Cost Value') # Cost càng thấp (màu tối) càng tốt

        # Điểm cực tiểu toàn cục thực tế (0,0)
        plt.scatter(0, 0, color='white', marker='x', s=100, label='True Global Min (0,0)')

        # Kết quả thuật toán tìm được
        found_x = best_bee.coords[0]
        found_y = best_bee.coords[1]
        plt.scatter(found_x, found_y, color='red', marker='o', s=100, edgecolors='black', label='ABC Result')

        plt.title(f'ABC Result on Function\nBest Cost: {best_cost:.6f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.show()