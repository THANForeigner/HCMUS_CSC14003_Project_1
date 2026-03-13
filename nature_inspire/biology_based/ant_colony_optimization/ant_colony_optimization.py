import numpy as np
import random

class ACO:
    def __init__(self, graph_dict, n_nodes, alpha=1.0, beta=2.0, rho=0.1, q=100.0):
        self.HARD_LIMIT_NODES = 3000  # Giới hạn chặn đứng
        if n_nodes > self.HARD_LIMIT_NODES:
            raise ValueError(
                f"Graph quá lớn cho ACO (N={n_nodes} > {self.HARD_LIMIT_NODES}). "
                "Vui lòng dùng thuật toán khác hoặc giảm kích thước input."
            )

        """
        Args:
            graph_dict: Dictionary chứa đồ thị {u: {v: w, ...}}
            n_nodes: Tổng số lượng node
            alpha: Tầm quan trọng của Pheromone
            beta: Tầm quan trọng của Heuristic (Khoảng cách)
            rho: Tốc độ bay hơi (0.1 = bay hơi 10% sau mỗi lượt)
            q: Hệ số thưởng Pheromone
        """
        self.graph = graph_dict
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q

        # Mapping tên node (str) <-> index ma trận (int)
        self.nodes_list = list(graph_dict.keys())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes_list)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes_list)}
        self.n = len(self.nodes_list)

        # 1. Khởi tạo Pheromone (Mùi hương) - ban đầu tất cả bằng nhau
        self.tau = np.ones((self.n, self.n)) * 0.1

        # 2. Khởi tạo Heuristic (Tầm nhìn) - nghịch đảo của khoảng cách (1/distance)
        self.eta = np.zeros((self.n, self.n))
        for u, neighbors in graph_dict.items():
            u_idx = self.node_to_idx[u]
            for v, w in neighbors.items():
                if v in self.node_to_idx:
                    v_idx = self.node_to_idx[v]
                    self.eta[u_idx][v_idx] = 1.0 / w if w > 0 else 0.0

    def run(self, start_node, goal_node, n_ants=10, max_iter=50):
        if start_node not in self.node_to_idx or goal_node not in self.node_to_idx:
            print("Error: Start or Goal node not found in graph.")
            return None, float('inf')

        start_idx = self.node_to_idx[start_node]
        goal_idx = self.node_to_idx[goal_node]

        best_path = None
        best_cost = float('inf')

        # print(f"--- Bắt đầu chạy ACO: {n_ants} kiến, {max_iter} vòng lặp ---")

        for iteration in range(max_iter):
            all_paths = []  # Lưu các đường đi tìm được trong vòng lặp này

            # --- GIAI ĐOẠN 1: Kiến tìm đường ---
            for k in range(n_ants):
                path_indices, cost = self._move_ant(start_idx, goal_idx)

                if path_indices:  # Nếu kiến tìm thấy đích
                    all_paths.append((path_indices, cost))
                    if cost < best_cost:
                        best_cost = cost
                        best_path = [self.idx_to_node[i] for i in path_indices]
                        # print(f"New Best found at iter {iteration}: Cost {best_cost}")

            # --- GIAI ĐOẠN 2: Cập nhật Pheromone ---
            self._update_pheromone(all_paths)

            # (Optional) Print progress
            # if (iteration + 1) % 10 == 0:
                # print(f"Iter {iteration + 1}/{max_iter}, Best Cost so far: {best_cost}")

        return best_path, best_cost

    def _move_ant(self, start_idx, goal_idx):
        """Một con kiến di chuyển từ Start -> Goal"""
        current = start_idx
        path = [current]
        cost = 0
        visited = set([current])

        while current != goal_idx:
            # Lấy danh sách hàng xóm
            # Lưu ý: Cần convert từ index về tên node để tra trong graph dict
            u_name = self.idx_to_node[current]
            neighbors_map = self.graph.get(u_name, {})

            valid_next_nodes = []
            probs = []

            for v_name, w in neighbors_map.items():
                if v_name in self.node_to_idx:
                    v_idx = self.node_to_idx[v_name]
                    if v_idx not in visited:
                        valid_next_nodes.append(v_idx)

                        # Công thức ACO: (Pheromone^alpha) * (Heuristic^beta)
                        t = self.tau[current][v_idx] ** self.alpha
                        e = self.eta[current][v_idx] ** self.beta
                        probs.append(t * e)

            # Nếu đi vào ngõ cụt (dead end)
            if not valid_next_nodes:
                return None, float('inf')

            # Chọn node tiếp theo dựa trên xác suất (Roulette Wheel)
            probs = np.array(probs)
            probs_sum = probs.sum()

            if probs_sum == 0:  # Trường hợp hiếm khi pheromone quá nhỏ
                next_idx = random.choice(valid_next_nodes)
            else:
                probs = probs / probs_sum
                next_idx = np.random.choice(valid_next_nodes, p=probs)

            # Di chuyển
            visited.add(next_idx)
            path.append(next_idx)
            cost += neighbors_map[self.idx_to_node[next_idx]]
            current = next_idx

        return path, cost

    def _update_pheromone(self, successful_paths):
        """Bay hơi và bồi đắp pheromone"""
        # 1. Bay hơi (Evaporation) trên toàn bộ ma trận
        self.tau *= (1.0 - self.rho)

        # 2. Bồi đắp (Deposit) chỉ trên đường đi thành công
        for path, cost in successful_paths:
            delta = self.q / cost  # Đường càng ngắn, thưởng càng nhiều
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Cập nhật 2 chiều vì đồ thị vô hướng
                self.tau[u][v] += delta
                self.tau[v][u] += delta  # Comment dòng này nếu là đồ thị có hướng


# -----------------------------------------------------------
# PHẦN ĐỌC FILE VÀ CHẠY
# -----------------------------------------------------------

def load_graph(file_path):
    graph = {}
    start_node, goal_node = None, None

    try:
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

            # Dòng 2: Start Goal (Bỏ qua dòng 1 N M)
            start_node, goal_node = lines[1].split()

            # Các dòng dữ liệu cạnh
            for line in lines[2:]:
                parts = line.split()
                u, v, w = parts[0], parts[1], float(parts[2])

                if u not in graph: graph[u] = {}
                if v not in graph: graph[v] = {}

                # Xử lý đa cạnh (Multigraph): Giữ cạnh nhỏ nhất
                # Ví dụ file có: 1->4 (159) và 1->4 (140) => Lấy 140
                if v in graph[u]:
                    graph[u][v] = min(graph[u][v], w)
                else:
                    graph[u][v] = w

                # Giả sử đồ thị vô hướng (Undirected) -> Thêm chiều ngược lại
                if u in graph[v]:
                    graph[v][u] = min(graph[v][u], w)
                else:
                    graph[v][u] = w

        return graph, start_node, goal_node

    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return None, None, None