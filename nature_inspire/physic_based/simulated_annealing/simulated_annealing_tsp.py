import math
import random

class SA_TSP(object):
        def __init__(self, max_vertices: int, coords, T=-1, stopping_T = -1, alpha = -1, stopping_iter=-1):
                self.coords=coords
                self.max_n=max_vertices
                self.T= math.sqrt(self.max_n) if T == -1 else T
                self.save_T = self.T
                self.alpha = alpha if alpha >= 0 and alpha <=1 else 0.995
                self.stopping_T = 1e-8 if stopping_T == -1 else stopping_T
                self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
                self.iteration = 1
                self.nodes = [i for i in range(self.max_n)]
                self.cur_solution_result = 0
                self.cur_solution_nodes = None
                self.best_solution_result = float("Inf")
                self.best_solution_nodes = None
                self.best_energy = None
                self.energy_list = []
        
        def calculate_distance(self, i, j):
                return math.sqrt(
                        (self.coords[i][0]-self.coords[j][0])**2
                        + (self.coords[i][1]-self.coords[j][1])**2
                        )
        def calculate_total_distance(self, solution_list):
                dist = 0
                for i in range(self.max_n):
                        u = solution_list[i]
                        v = solution_list[(i + 1) % self.max_n]
                        dist += self.calculate_distance(u, v)
                return dist
        
        #Generate the initial state using simple greedy algorithm:
        #- Start from a city and tranverse through the graph
        #- Find the next shortest distance and add that node to the graph
        def initial_solution(self):
                cur_node = random.choice(self.nodes)
                solution = 0
                solution_list = [cur_node]
                free_nodes = set(self.nodes)
                free_nodes.remove(cur_node)
                
                while free_nodes:
                        next_node = min(free_nodes, key=lambda x: self.calculate_distance(cur_node, x))
                        free_nodes.remove(next_node)
                        solution_list.append(next_node)
                        cur_node = next_node
                        
                solution = self.calculate_total_distance(solution_list)   
                     
                if(solution < self.best_solution_result):
                        self.best_solution_result = solution
                        self.best_solution_nodes = solution_list
                        self.best_energy = solution
                self.energy_list.append(solution)
                
                return solution, solution_list
        
        def simulated_annealing(self):
                
                #Generate the intial state
                self.cur_solution_result, self.cur_solution_nodes = self.initial_solution()
                
                while self.T >= self.stopping_T and self.iteration < self.stopping_iter:
                        
                        #Generate next neighbour state using the Reverse (2 - opt)
                        new_solution_nodes = list(self.cur_solution_nodes)
                        l = random.randint(2, self.max_n - 1) #Generate a random length for the route
                        i = random.randint(0, self.max_n - l) #Generate a random starting city for the route
                        new_solution_nodes[i : (i + l)] = reversed(new_solution_nodes[i : (i + l)]) #Reverse all edges on the route
                        
                        new_solution_result = self.calculate_total_distance(new_solution_nodes)
                        delta_energy = new_solution_result - self.cur_solution_result
                        if(delta_energy <= 0):
                                self.cur_solution_nodes = new_solution_nodes
                                self.cur_solution_result = new_solution_result
                                if(new_solution_result < self.best_solution_result):
                                        self.best_solution_result = new_solution_result
                                        self.best_solution_nodes = new_solution_nodes
                                        self.best_energy = new_solution_result
                        else:
                                p_accept = math.exp(-delta_energy/self.T)
                                if random.uniform(0,1) < p_accept:
                                        self.cur_solution_nodes = new_solution_nodes
                                        self.cur_solution_result = new_solution_result   
                        self.T *= self.alpha
                        self.iteration += 1 
                        self.energy_list.append(self.cur_solution_result)
                        
        #Batch annealing to repeat simulated annealing in order to increase the accuracy of the result                  
        def batch_annealing(self, times = 10):
                for i in range (1, times + 1):
                        self.T = self.save_T
                        self.iteration = 1
                        self.simulated_annealing()
                        
        def run(self, times = 100):
                self.batch_annealing(times)