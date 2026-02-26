import random
from genetic_algorithm import GA



class GA_GraphColoring(GA):
    def __init__(self, adj_matrix, num_colors, max_local_search_steps=50, **kwargs):
        """
        :param adj_matrix: Ma trận kề (NxN). adj_matrix[i][j] = 1 nếu có cạnh nối.
        :param num_colors: Số lượng màu tối đa được phép dùng (k).
        :param max_local_search_steps: Số bước tối đa cho thuật toán Min-Conflicts.
        """
        super().__init__(**kwargs)
        self.adj_matrix = adj_matrix
        self.n_nodes = len(adj_matrix)
        self.num_colors = num_colors
        self.max_local_search_steps = max_local_search_steps



    def create_individual(self):
        """Tạo màu ngẫu nhiên cho các đỉnh"""
        ind = [random.randint(0, self.num_colors - 1) for _ in range(self.n_nodes)]
        # Áp dụng sửa lỗi ngay từ đầu để quần thể khởi tạo tốt hơn
        return self.min_conflicts_tuning(ind, max_steps=10)
    


    def calculate_fitness(self, individual):
        """
        Fitness = -(Số cạnh bị xung đột).
        Mục tiêu: Fitness = 0 (Không còn xung đột).
        """
        conflicts = 0
        for u in range(self.n_nodes):
            for v in range(u + 1, self.n_nodes):
                if self.adj_matrix[u][v] == 1 and individual[u] == individual[v]:
                    conflicts += 1
        return -conflicts # Dấu âm để GA tìm Max (Max là 0)
    


    def crossover(self, p1, p2):
        """Two-point crossover (Giống Knapsack)"""
        if random.random() > 0.9: # Tỉ lệ lai ghép (có thể lấy từ self.crossover_rate nếu truyền vào)
            return p1[:], p2[:]
            
        point1 = random.randint(0, self.n_nodes - 2)
        point2 = random.randint(point1 + 1, self.n_nodes - 1)
        
        c1 = p1[:point1] + p2[point1:point2] + p1[point2:]
        c2 = p2[:point1] + p1[point1:point2] + p2[point2:]
        
        # Không cần Local Search ở đây để tiết kiệm thời gian, 
        # chỉ làm ở Mutation hoặc Create
        return c1, c2
    


    def mutate(self, individual):
        """
        Đột biến: Đổi màu ngẫu nhiên một đỉnh.
        Sau đó chạy Local Search (Min-Conflicts) để sửa lỗi.
        """
        if random.random() < self.mutation_rate:
            # Chọn ngẫu nhiên 1 đỉnh để đổi màu
            node = random.randint(0, self.n_nodes - 1)
            individual[node] = random.randint(0, self.num_colors - 1)
        
        # --- MEMETIC STEP: MIN-CONFLICTS HEURISTIC ---
        # Giống như 2-opt của TSP hay Repair của Knapsack
        return self.min_conflicts_tuning(individual, self.max_local_search_steps)
    
    

    def min_conflicts_tuning(self, individual, max_steps):
        """
        Thuật toán Min-Conflicts:
        Tìm các đỉnh đang bị xung đột và đổi màu chúng sang màu 'lành' nhất.
        """
        for _ in range(max_steps):
            # 1. Tìm tất cả các đỉnh đang có conflict
            conflicted_nodes = []
            for u in range(self.n_nodes):
                is_conflict = False
                for v in range(self.n_nodes):
                    if self.adj_matrix[u][v] == 1 and individual[u] == individual[v]:
                        is_conflict = True
                        break
                if is_conflict:
                    conflicted_nodes.append(u)
            
            # Nếu không còn conflict nào -> Tuyệt vời, dừng ngay
            if not conflicted_nodes:
                return individual
                
            # 2. Chọn ngẫu nhiên 1 đỉnh bị lỗi để sửa
            node_to_fix = random.choice(conflicted_nodes)
            
            # 3. Tìm màu nào gây ra ÍT conflict nhất cho đỉnh này
            min_conflict_count = float('inf')
            best_color = individual[node_to_fix]
            
            # Duyệt thử tất cả các màu
            original_color = individual[node_to_fix]
            possible_colors = list(range(self.num_colors))
            random.shuffle(possible_colors) # Random để tránh thiên vị màu số 0
            
            for color in possible_colors:
                if color == original_color: continue
                
                # Đếm lỗi nếu chọn màu này
                current_conflicts = 0
                for neighbor in range(self.n_nodes):
                    if self.adj_matrix[node_to_fix][neighbor] == 1 and color == individual[neighbor]:
                        current_conflicts += 1
                
                if current_conflicts < min_conflict_count:
                    min_conflict_count = current_conflicts
                    best_color = color
                    # Nếu tìm được màu 0 lỗi thì chọn luôn cho nhanh
                    if min_conflict_count == 0:
                        break
            
            # Cập nhật màu tốt nhất
            individual[node_to_fix] = best_color
            
        return individual    