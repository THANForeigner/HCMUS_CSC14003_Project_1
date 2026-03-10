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

        print(f"--- Bắt đầu FA trên hàm {self.dim} chiều ---")

        for t in range(self.max_iter):
            # Với mỗi con đom đóm i
            for i in range(self.pop_size):
                # So sánh với tất cả các con j khác
                for j in range(self.pop_size):

                    # Nếu con j sáng hơn con i (Fitness j < Fitness i)
                    # Thì con i sẽ bay về phía con j
                    if fitness[j] < fitness[i]:
                        # Tính khoảng cách Euclidean
                        r = np.linalg.norm(fireflies[i] - fireflies[j])

                        # Tính độ hấp dẫn (beta) giảm dần theo khoảng cách
                        # Công thức: beta = beta0 * exp(-gamma * r^2)
                        beta = self.beta0 * np.exp(-self.gamma * (r ** 2))

                        # Tạo bước di chuyển ngẫu nhiên
                        # (alpha giảm dần theo thời gian để hội tụ chính xác hơn)
                        alpha_t = self.alpha * (0.97 ** t)
                        epsilon = np.random.uniform(-0.5, 0.5, self.dim)

                        # CẬP NHẬT VỊ TRÍ MỚI
                        # Xi_mới = Xi_cũ + Độ_hút*(Xj - Xi) + Ngẫu_nhiên
                        fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + alpha_t * epsilon

                        # Kiểm tra biên (không cho bay ra khỏi vùng tìm kiếm)
                        fireflies[i] = np.clip(fireflies[i], self.lb, self.ub)

                        # Tính lại điểm cho con i sau khi di chuyển
                        fitness[i] = self.func(fireflies[i])

            # Cập nhật kết quả tốt nhất toàn cục
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < global_best_score:
                global_best_score = fitness[min_fitness_idx]
                global_best = fireflies[min_fitness_idx].copy()

            self.history.append(fireflies.copy())

            if t % 10 == 0:
                print(f"Vòng {t}: Best Fitness = {global_best_score:.5f}")

        return global_best, global_best_score