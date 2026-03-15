import math
import random

class SA_GraphColoring(object):
    def __init__ (self, max_colors: int, max_vertices: int, edges: list, T=-1, stopping_T = -1, alpha = -1, stopping_iter=-1):
        self.max_colors = max_colors
        self.max_n = max_vertices
        self.edges = edges
        self.T= math.sqrt(self.max_n) if T == -1 else T
        self.save_T = self.T
        self.alpha = alpha if alpha >= 0 and alpha <=1 else 0.995
        self.stopping_T = 1e-8 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        self.cur_solution = []
        self.cur_energy = None
        self.best_solution = []
        self.best_energy = float('inf')
        self.energy_list = []
        
    # Năng lượng của đồ thị sẽ là số lượng các đỉnh (nút) có xung đột màu sắc với các đỉnh kề nó
    def get_graph_energy(self, colored_graph):
        energy = 0
        for u,v in self.edges:
            if colored_graph[u] == colored_graph[v]:
                energy += 1
        return energy
    
    # Tạo đồ thị có màu ban đầu bằng cách gán màu ngẫu nhiên cho mọi đỉnh (nút)
    def initial_solution(self):
        return [random.randrange(self.max_colors) for _ in range(self.max_n)]
    
    # Lấy trạng thái lân cận bằng cách sử dụng phép dịch chuyển một nút (One-node Move)
    def generate_new_solution(self, current_solution, node): 
        new_sol = current_solution.copy()
        old_color = current_solution[node] 
        if self.max_colors > 1:
            new_color = random.randrange(self.max_colors)
            while new_color == old_color:
                new_color = random.randrange(self.max_colors)
            new_sol[node] = new_color
        return new_sol
    
    def get_next_node(self):
        return random.randint(0, self.max_n-1)        
    
    def simulated_annealing(self):
        
        # Sinh trạng thái đầu
        self.cur_solution = self.initial_solution()
        self.cur_energy = self.get_graph_energy(self.cur_solution)
        self.energy_list.append(self.cur_energy)
        
        if self.cur_energy < self.best_energy:
             self.best_energy = self.cur_energy
             self.best_solution = self.cur_solution.copy()
        while self.T >= self.stopping_T and self.iteration < self.stopping_iter:
            if self.best_energy == 0: # Thoát khi tìm thấy biểu đồ đúng
                break
            
            # Lấy trạng thái lân cận bằng cách sử dụng phép dịch chuyển một nút (One-node Move)
            rand_node = self.get_next_node()
            new_solution = self.generate_new_solution(self.cur_solution, rand_node)
            
            new_energy = self.get_graph_energy(new_solution)
            delta_energy = new_energy - self.cur_energy
            if delta_energy <= 0:
                self.cur_energy = new_energy
                self.cur_solution = new_solution
                if new_energy < self.best_energy:
                    self.best_energy = new_energy
                    self.best_solution = new_solution
            else:
                p_accept = math.exp(-delta_energy/self.T)
                if random.uniform(0,1) < p_accept:
                    self.cur_energy = new_energy
                    self.cur_solution = new_solution
            self.T = self.T * self.alpha
            self.iteration += 1 
            self.energy_list.append(self.cur_energy)
    
    # Chạy batch annealing để tăng độ chính xác      
    def batch_annealing(self, times = 1000):
        for i in range (1, times + 1):
            if self.best_energy == 0:
                break
            self.T = self.save_T
            self.iteration = 1
            self.simulated_annealing()
            
    def run(self, times = 1000):
        self.batch_annealing(times)