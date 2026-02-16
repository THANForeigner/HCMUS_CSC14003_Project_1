import numpy as np



class DifferentialEvolution:
    def __init__(self, func, bounds, dim, pop_size=50, max_gen=1000, F=0.5, Cr=0.7):
        """
        func: (Sphere, Rastrigin...)
        bounds: [(min, max), ...] or [min, max]
        dim: Dimension
        F: Scaling Factor (0.4 -> 1.0)
        Cr: Crossover Rate (0.0 -> 1.0)
        """
        self.func = func
        self.bounds = np.array([bounds] * dim) # Giả sử bounds giống nhau cho mọi chiều
        self.dim = dim
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.F = F
        self.Cr = Cr
        
        # Khởi tạo quần thể ngẫu nhiên trong khoảng bounds
        self.min_b, self.max_b = self.bounds[:, 0], self.bounds[:, 1]
        self.population = self.min_b + np.random.rand(pop_size, dim) * (self.max_b - self.min_b)
        
        # Tính fitness ban đầu
        self.fitness = np.apply_along_axis(self.func, 1, self.population)
        
        # Lưu kết quả tốt nhất
        self.best_idx = np.argmin(self.fitness)
        self.best_vector = self.population[self.best_idx]
        self.best_score = self.fitness[self.best_idx]

    def run(self, max_gen):
        self.max_gen = max_gen
        history = []
        
        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                # 1. MUTATION: Chọn 3 vector ngẫu nhiên khác i (r1, r2, r3)
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = self.population[np.random.choice(idxs, 3, replace=False)]
                
                # V = X_r1 + F * (X_r2 - X_r3)
                mutant_vector = r1 + self.F * (r2 - r3)
                
                # Kẹp giá trị trong bounds (Clamping)
                mutant_vector = np.clip(mutant_vector, self.min_b, self.max_b)
                
                # 2. CROSSOVER (Binomial): Tạo vector thử nghiệm U
                cross_points = np.random.rand(self.dim) < self.Cr
                # Luôn giữ ít nhất 1 gen từ mutant để đảm bảo có sự thay đổi
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial_vector = np.where(cross_points, mutant_vector, self.population[i])
                
                # 3. SELECTION: Greedy (Chỉ lấy nếu tốt hơn cha mẹ)
                trial_fitness = self.func(trial_vector)
                
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial_vector
                    
                    # Cập nhật Global Best
                    if trial_fitness < self.best_score:
                        self.best_score = trial_fitness
                        self.best_vector = trial_vector

            history.append(self.best_score)

            # Log mỗi 100 generation
            if (gen+1) % 100 == 0:
                print(f"Gen {gen+1}: Best Fitness = {self.best_score:.10f}")
                
                # Điều kiện dừng sớm nếu đã tìm ra 0 (tuyệt đối)
                if self.best_score == 0:
                    break

        return (self.best_score, self.best_vector), history