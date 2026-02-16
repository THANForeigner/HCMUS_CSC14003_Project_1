
class DFSColoring:
    def __init__(self, graph, n_nodes, n_colors):
        self.graph = graph
        self.n = n_nodes
        self.k = n_colors  # Số màu tối đa được dùng
        self.colors = [0] * (n_nodes + 1)  # Mảng lưu màu kết quả

    def solve(self):
        # Bắt đầu tô từ đỉnh số 1
        if self._backtrack(1):
            return self.colors
        return None

    def _backtrack(self, node_idx):
        if node_idx > self.n:
            return True  # Đã tô xong hết các đỉnh

        # Thử lần lượt các màu từ 1 đến K
        for c in range(1, self.k + 1):
            if self._is_safe(node_idx, c):
                self.colors[node_idx] = c
                if self._backtrack(node_idx + 1):
                    return True
                self.colors[node_idx] = 0

        return False

    def _is_safe(self, u, c):
        # Kiểm tra xem có hàng xóm nào của u đang tô màu c không
        for v in self.graph[u]:
            if self.colors[v] == c:
                return False
        return True