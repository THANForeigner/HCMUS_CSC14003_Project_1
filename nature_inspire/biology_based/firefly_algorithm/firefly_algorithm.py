import numpy as np
from problems import problem


class FA:
    def __init__(self, func_name="", pop_size=30, dim=10, max_iter=50, alpha=0.2, beta0=1.0, gamma=0.01, function=None,
                 bounds=None):
        # Load bài toán
        if function is not None and bounds is not None:
            self.func = function
            self.lb = bounds[0]
            self.ub = bounds[1]
        else:
            p = problem.get_problem(func_name)
            self.func = p["func"]
            self.lb = p["lb"]
            self.ub = p["ub"]

        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.history = []

        # Các tham số chuẩn của FA
        self.alpha = alpha  # Độ ngẫu nhiên (Randomness)
        self.beta0 = beta0  # Sức hút tại khoảng cách r=0
        self.gamma = gamma  # Hệ số hấp thụ ánh sáng (Light absorption)

    def run(self):
        # 1. Khởi tạo quần thể đom đóm
        fireflies = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(self.func, 1, fireflies)

        # Tìm con tốt nhất ban đầu
        best_idx = np.argmin(fitness)
        global_best = fireflies[best_idx].copy()
        global_best_score = fitness[best_idx]

        self.history.append(fireflies.copy())

        for t in range(self.max_iter):
            alpha_t = self.alpha * (0.97 ** t)
            
            # 1. Tính chênh lệch vị trí giữa tất cả các cặp
            diffs = fireflies[np.newaxis, :, :] - fireflies[:, np.newaxis, :]
            
            # 2. Tính bình phương khoảng cách (r^2) giữa các đom đóm
            r2 = np.sum(diffs**2, axis=-1)
            
            # 3. Tính độ hấp dẫn (beta) dựa trên khoảng cách:
            beta = self.beta0 * np.exp(-self.gamma * r2)
            
            # 4. Tạo mặt nạ (mask) để lọc ra trường hợp đom đóm j sáng hơn đom đóm i
            # Do là bài toán tìm cực tiểu (minimization), "sáng hơn" đồng nghĩa với fitness nhỏ hơn
            mask = fitness[np.newaxis, :] < fitness[:, np.newaxis]
            
            # 5. Tính tổng quãng đường di chuyển do bị thu hút bởi tất cả đom đóm sáng hơn
            attractions = (beta * mask)[:, :, np.newaxis] * diffs
            total_attraction = np.sum(attractions, axis=1)
            
            # 6. Tạo thành phần bước nhảy ngẫu nhiên (theo logic gốc của thuật toán FA)
            # Epsilon có kích thước (pop_size, dim)
            epsilon = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
            
            # Update all positions at once
            fireflies += total_attraction + alpha_t * epsilon
            
            # Ép lại vào biên một thể
            fireflies = np.clip(fireflies, self.lb, self.ub)
            
            # Cực kỳ quan trọng: CHỈ GỌI hàm mục tiêu MỘT LẦN cho cả đàn! Tiết kiệm NFE gấp 30 lần!
            fitness = np.apply_along_axis(self.func, 1, fireflies)

            # Cập nhật kết quả tốt nhất toàn cục
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < global_best_score:
                global_best_score = fitness[min_fitness_idx]
                global_best = fireflies[min_fitness_idx].copy()

            self.history.append(fireflies.copy())

        return global_best, global_best_score