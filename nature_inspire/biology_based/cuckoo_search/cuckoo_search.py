import numpy as np
from problems import problem


def levy_flight_butakova(alpha, dim, size=1):
    # Step 4:
    beta = alpha / 2.0

    # Step 6, 7, 8:
    chi = np.random.uniform(0, 2 * np.pi, (size, dim))
    delta = np.random.exponential(1.0, (size, dim))
    nu = np.random.normal(0, 1.0, (size, dim))

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
    if size == 1:
        return step[0]
    return step


class CS:
    def __init__(self, func_name="", pop_size=30, dim=10, max_iter=50, pa=0.25, beta=1.5, function=None, bounds=None):
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
        self.n_nests = pop_size
        self.max_iter = max_iter
        self.history = []

        # Tham số CS
        self.pa = pa  # Xác suất bị phát hiện
        self.beta = beta  # Tham số Levy

    def run(self):
        # 1. Khởi tạo tổ chim
        nests = np.random.uniform(self.lb, self.ub, (self.n_nests, self.dim))
        fitness = np.apply_along_axis(self.func, 1, nests)

        # Lưu tổ tốt nhất
        best_idx = np.argmin(fitness)
        best_nest = nests[best_idx].copy()
        best_score = fitness[best_idx]

        self.history.append(nests.copy())

        # print(f"--- Bắt đầu CS (Butakova Levy) trên hàm {self.dim} chiều ---")

        for t in range(self.max_iter):
            # --- BƯỚC 1: Tìm tổ mới bằng Lévy Flight (Vận hành dạng Vector) ---
            step_levy = levy_flight_butakova(self.beta, self.dim, size=self.n_nests)
            step_direction = step_levy * (nests - best_nest)
            new_cuckoos = nests + 0.01 * step_direction
            new_cuckoos = np.clip(new_cuckoos, self.lb, self.ub)
            
            # Tính fitness HÀNG LOẠT cho 100% các tổ chim con
            new_fitness = np.apply_along_axis(self.func, 1, new_cuckoos)
            
            # Random xem chim Cuckoo sẽ đẻ trứng vào tổ nào
            replace_idx = np.random.randint(0, self.n_nests, self.n_nests)
            
            for i in range(self.n_nests):
                j = replace_idx[i]
                if new_fitness[i] < fitness[j]:
                    nests[j] = new_cuckoos[i]
                    fitness[j] = new_fitness[i]

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

            self.history.append(nests.copy())

            # if t % 10 == 0:
            #     print(f"Vòng {t}: Best Fitness = {best_score:.5f}")

        return best_nest, best_score