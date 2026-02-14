import numpy as np
from nature_inspire.biology_based import problem


def levy_flight_butakova(alpha, dim):
    # Step 4:
    beta = alpha / 2.0

    # Step 6, 7, 8:
    chi = np.random.uniform(0, 2 * np.pi, dim)
    delta = np.random.exponential(1.0, dim)
    nu = np.random.normal(0, 1.0, dim)

    # Step 9:
    epsilon = 1e-10
    term1 = np.sin(beta * chi) / (np.sin(chi) + epsilon)
    term2_tu = np.sin((1 - beta) * chi)
    term2_mau = np.sin(beta * chi) + epsilon
    exponent = (1 - beta) / beta
    ratio = (term2_tu / term2_mau).astype(complex)
    a_val = term1 * (ratio ** exponent)
    zeta = (a_val / (delta + epsilon)) ** exponent

    # Step 10 & Eq (3)
    step = np.real(np.sqrt(2 * zeta) * nu)

    return step


class CS:
    def __init__(self, func_name, pop_size=25, dim=10, max_iter=100):
        # Load bài toán
        p = problem.get_problem(func_name)
        self.func = p["func"]
        self.lb = p["lb"]
        self.ub = p["ub"]
        self.dim = dim
        self.n_nests = pop_size
        self.max_iter = max_iter
        self.history = []

        # Tham số CS
        self.pa = 0.25  # Xác suất bị phát hiện
        self.beta = 1.5  # Tham số Levy

    def solve(self):
        # 1. Khởi tạo tổ chim
        nests = np.random.uniform(self.lb, self.ub, (self.n_nests, self.dim))
        fitness = np.apply_along_axis(self.func, 1, nests)

        # Lưu tổ tốt nhất
        best_idx = np.argmin(fitness)
        best_nest = nests[best_idx].copy()
        best_score = fitness[best_idx]

        print(f"--- Bắt đầu CS (Butakova Levy) trên hàm {self.dim} chiều ---")

        for t in range(self.max_iter):
            # --- BƯỚC 1: Tìm tổ mới bằng Lévy Flight ---
            new_nests = nests.copy()

            for i in range(self.n_nests):
                origin = nests[i]
                step_levy = levy_flight_butakova(self.beta, self.dim)
                step_direction = step_levy * (origin - best_nest)
                new_cuckoo = origin + 0.01 * step_direction
                new_cuckoo = np.clip(new_cuckoo, self.lb, self.ub)
                f_new = self.func(new_cuckoo)
                j = np.random.randint(0, self.n_nests)
                if f_new < fitness[j]:
                    nests[j] = new_cuckoo
                    fitness[j] = f_new

            # --- BƯỚC 2: Loại bỏ tổ tồi (Local Random Walk) ---
            # Tạo ma trận mask K (True là bị phát hiện)
            K = np.random.rand(self.n_nests, self.dim) < self.pa

            # Tạo bước nhảy ngẫu nhiên cục bộ
            # Dùng permutation để trộn ngẫu nhiên các tổ
            nest_shuffled1 = nests[np.random.permutation(self.n_nests)]
            nest_shuffled2 = nests[np.random.permutation(self.n_nests)]

            step_rand = np.random.rand(self.n_nests, self.dim) * (nest_shuffled1 - nest_shuffled2)
            nests = nests + K * step_rand
            nests = np.clip(nests, self.lb, self.ub)
            fitness = np.apply_along_axis(self.func, 1, nests)

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_score:
                best_score = fitness[min_idx]
                best_nest = nests[min_idx].copy()

            self.history.append(best_score)

            if t % 10 == 0:
                print(f"Vòng {t}: Best Fitness = {best_score:.5f}")

        return best_nest, best_score