import numpy as np
import random
import copy

ALPHA = 2  # Pheromone importance
BETA = 4  # Heuristic importance
RHO = 0.04  # Evaporation rate
P_BEST = 0.05  # Probability for calculating tau limits

class ACOColoring:
    def __init__(self, graph_dict, n_nodes, max_iter=100, n_ants=20):
        self.graph = graph_dict
        self.n = n_nodes
        self.max_iter = max_iter
        self.n_ants = n_ants
        self.nodes = list(range(1, n_nodes + 1))
        self.adj_matrix = np.zeros((n_nodes + 1, n_nodes + 1), dtype=bool)
        self.degrees = {}
        for u, neighbors in graph_dict.items():
            u = int(u)
            self.degrees[u] = len(neighbors)
            for v in neighbors:
                v = int(v)
                self.adj_matrix[u][v] = True
                self.adj_matrix[v][u] = True  # Vô hướng

        # Khởi tạo Pheromone
        # Pheromone tau_ij: độ hấp dẫn để node i và node j CÙNG MÀU
        # Chỉ có ý nghĩa với i, j KHÔNG kề nhau.
        self.tau_max = 1.0
        self.tau_min = 0.001
        self.pheromone = np.ones((n_nodes + 1, n_nodes + 1)) * self.tau_max

    def solve(self):
        best_global_solution = None
        best_global_k = float('inf')  # Số màu ít nhất
        best_global_score = -1  # Điểm hàm mục tiêu Eq(13) cao nhất

        for iteration in range(self.max_iter):
            iteration_best_sol = None
            iteration_best_score = -1

            # --- 1. Construct Solutions (Ants) ---
            for k in range(self.n_ants):
                sol, k_colors = self._construct_solution()
                score = self._evaluate(sol)

                # Cập nhật tốt nhất trong vòng lặp
                if score > iteration_best_score:
                    iteration_best_score = score
                    iteration_best_sol = sol

            # --- 2. Local Search (Kempe Chain) ---
            improved_sol = self._kempe_chain_local_search(iteration_best_sol)
            improved_score = self._evaluate(improved_sol)

            # Cập nhật Global Best
            improved_k = len(improved_sol)

            # Ưu tiên số màu ít hơn, nếu bằng nhau thì ưu tiên score cao hơn (Eq 13)
            if improved_k < best_global_k:
                best_global_k = improved_k
                best_global_score = improved_score
                best_global_solution = improved_sol
                print(f"Iter {iteration}: New Best K = {best_global_k}")
            elif improved_k == best_global_k and improved_score > best_global_score:
                best_global_score = improved_score
                best_global_solution = improved_sol

            # 3. Update Pheromone
            self._update_pheromone(improved_sol, improved_score)

        return self._format_result(best_global_solution)

    def _construct_solution(self):
        uncolored = set(self.nodes)
        color_classes = []

        while uncolored:
            current_class = set()

            # Bước 1: Chọn đỉnh đầu tiên cho màu mới (Eq 9 & 10)
            # Heuristic: Chọn đỉnh có bậc cao nhất trong đồ thị con W (uncolored)
            # Prob = deg_W(v) / sum
            first_node = self._select_first_node(uncolored)
            current_class.add(first_node)
            uncolored.remove(first_node)

            # Bước 2: Chọn các đỉnh tiếp theo thêm vào màu này (Eq 8)
            while True:
                # Tìm các đỉnh ứng viên (feasible): Thuộc uncolored VÀ không kề với đỉnh nào trong current_class
                candidates = []
                for v in uncolored:
                    is_conflict = False
                    for u in current_class:
                        if self.adj_matrix[v][u]:  # Nếu kề nhau
                            is_conflict = True
                            break
                    if not is_conflict:
                        candidates.append(v)

                if not candidates:
                    break

                # Chọn đỉnh tiếp theo dựa trên Pheromone và Heuristic
                next_node = self._select_next_node(candidates, current_class, uncolored)
                current_class.add(next_node)
                uncolored.remove(next_node)

            color_classes.append(current_class)

        return color_classes, len(color_classes)

    def _select_first_node(self, uncolored):
        candidates = list(uncolored)
        # Tính bậc động trong tập uncolored
        degrees = []
        for u in candidates:
            d = 0
            for v in uncolored:
                if u != v and self.adj_matrix[u][v]:
                    d += 1
            degrees.append(d)

        # Eq 10: Prob ~ degree^beta (giả sử beta như heuristic)
        # Để tránh degree=0, cộng thêm epsilon
        probs = np.array(degrees, dtype=float) + 0.1
        probs = np.power(probs, BETA)

        sum_probs = np.sum(probs)
        if sum_probs == 0:
            return random.choice(candidates)

        probs /= sum_probs
        return np.random.choice(candidates, p=probs)

    def _select_next_node(self, candidates, current_class, uncolored):
        probs = []
        for i in candidates:
            # 1. Tính Heuristic eta_ik (Eq 9: degree in uncolored W)
            # Bài báo gọi là deg_W(v)
            deg_w = 0
            for w in uncolored:
                if i != w and self.adj_matrix[i][w]:
                    deg_w += 1
            eta = deg_w

            # 2. Tính Pheromone tau_ik (Eq 7)
            # Tổng pheromone giữa i và các đỉnh j đã có trong current_class
            sum_tau = 0.0
            for j in current_class:
                sum_tau += self.pheromone[i][j]

            avg_tau = sum_tau / len(current_class)  # Eq 7

            # 3. Kết hợp (Eq 8)
            score = (avg_tau ** ALPHA) * ((eta + 0.1) ** BETA)
            probs.append(score)

        probs = np.array(probs)
        sum_probs = np.sum(probs)
        if sum_probs == 0:
            return random.choice(candidates)

        probs /= sum_probs
        return np.random.choice(candidates, p=probs)

    def _evaluate(self, solution):
        score = 0
        for c_set in solution:
            score += len(c_set) ** 2
        return score

    def _kempe_chain_local_search(self, solution):
        new_sol = copy.deepcopy(solution)
        if len(new_sol) < 2:
            return new_sol

        # Thử một số lần random swap
        for _ in range(5):
            # Chọn 2 chỉ số màu ngẫu nhiên
            idx1, idx2 = random.sample(range(len(new_sol)), 2)
            set1, set2 = new_sol[idx1], new_sol[idx2]

            # Gộp 2 set lại
            union_nodes = list(set1 | set2)
            if not union_nodes: continue

            # Xây dựng đồ thị con chỉ gồm các node này
            # Tìm các thành phần liên thông (Kempe Chains)
            visited = set()
            components = []

            for node in union_nodes:
                if node not in visited:
                    comp = []
                    queue = [node]
                    visited.add(node)
                    while queue:
                        u = queue.pop(0)
                        comp.append(u)
                        # Tìm hàng xóm trong union_nodes
                        for v in union_nodes:
                            if v not in visited and self.adj_matrix[u][v]:
                                visited.add(v)
                                queue.append(v)
                    components.append(comp)
            current_score = len(set1) ** 2 + len(set2) ** 2

            for comp in components:
                # Đếm số node thuộc set1 và set2 trong component này
                s1_nodes = [n for n in comp if n in set1]
                s2_nodes = [n for n in comp if n in set2]

                # Giả sử ta swap component này
                # New size of set1 = old_len1 - len(s1_nodes) + len(s2_nodes)
                new_len1 = len(set1) - len(s1_nodes) + len(s2_nodes)
                new_len2 = len(set2) - len(s2_nodes) + len(s1_nodes)

                new_score = new_len1 ** 2 + new_len2 ** 2

                if new_score > current_score:
                    # Thực hiện Swap
                    for n in s1_nodes:
                        new_sol[idx1].remove(n)
                        new_sol[idx2].add(n)
                    for n in s2_nodes:
                        new_sol[idx2].remove(n)
                        new_sol[idx1].add(n)

                    # Update reference sets for next loop
                    set1 = new_sol[idx1]
                    set2 = new_sol[idx2]
                    current_score = new_score

            # Loại bỏ set rỗng nếu có
            new_sol = [s for s in new_sol if len(s) > 0]
            if len(new_sol) < 2: break

        return new_sol

    def _update_pheromone(self, best_sol, best_score):
        # 1. Bay hơi (Evaporation)
        self.pheromone *= (1.0 - RHO)

        deposit_val = best_score

        for color_set in best_sol:
            nodes = list(color_set)
            # Update cho mọi cặp (u, v) cùng màu
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    u, v = nodes[i], nodes[j]
                    self.pheromone[u][v] += deposit_val
                    self.pheromone[v][u] += deposit_val

        t_max = best_score / RHO
        t_min = t_max * 0.001

        self.pheromone = np.clip(self.pheromone, t_min, t_max)

    def _format_result(self, solution):
        colors = [0] * (self.n + 1)
        for color_idx, color_set in enumerate(solution):
            for node in color_set:
                colors[node] = color_idx + 1  # Màu từ 1

        # Trả về (Số màu, List màu)
        return len(solution), colors[1:]