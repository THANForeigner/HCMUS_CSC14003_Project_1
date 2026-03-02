import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
import continuous_functions as cf

func_config = {
    "sphere": {
        "func": cf.sphere,
        "lb": -5.12, "ub": 5.12,
        "optimal": 0.0
    },
    "rastrigin": {
        "func": cf.rastrigin,
        "lb": -5.12, "ub": 5.12,
        "optimal": 0.0
    },
    "rosenbrock": {
        "func": cf.rosenbrock,
        "lb": -5.0, "ub": 10.0,
        "optimal": 0.0
    },
    "ackley": {
        "func": cf.ackley,
        "lb": -32.768,
        "ub": 32.768,
        "optimal": 0.0
    },
    "easom": {
        "func": cf.easom,
        "lb": -100,
        "ub": 100,
        "optimal": -1.0
    },
    "griewank": {
        "func": cf.griewank,
        "lb": -600.0,
        "ub": 600.0,
        "optimal": 0.0
    }
}

def get_problem(name):
    if name not in func_config:
        raise ValueError(f"Không tìm thấy hàm tên: {name}")
    return func_config[name]

algo_config = {
    # PSO: n_particles (số hạt), w (quán tính), c1/c2 (học cá nhân/xã hội)
    "PSO": {"n_particles": 30, "w": 0.7, "c1": 1.5, "c2": 1.5, "max_iter": 50},

    # ACO: n_ants (số kiến), archive_size (lưu trữ tốt), q/xi (hội tụ/phân tán)
    "ACO": {"n_ants": 30, "archive_size": 50, "q": 0.5, "xi": 0.85, "max_iter": 50},

    # ABC: n_bees (số ong), limit (số lần thử tối đa trước khi bỏ)
    "ABC": {"n_bees": 30, "limit": 20, "max_iter": 50},

    # Firefly: n_fireflies (số đom đóm), alpha (bước nhảy ngẫu nhiên), beta0/gamma (sức hút tối đa/giảm sức hút)
    "Firefly": {"n_fireflies": 30, "alpha": 0.2, "beta0": 1.0, "gamma": 0.01, "max_iter": 50},

    # Cuckoo: n_nests (số tổ), pa (xác suất trứng bị loại bỏ), beta (tham số Levy)
    "Cuckoo": {"n_nests": 30, "pa": 0.25, "beta": 1.5, "max_iter": 50},

    # GA: population_size (kích thước), elite_size (giữ lại), crossover_rate/mutation_rate (lai/đột biến)
    "GA": {"population_size": 50, "crossover_rate": 0.8, "mutation_rate": 0.1, "elite_size": 2, "max_iter": 50},

    # SA: initial_temp/final_temp (nhiệt ban đầu/kết thúc), alpha (hệ số làm mát)
    "SA": {"initial_temp": 100.0, "alpha": 0.95, "final_temp": 0.001, "max_iter": 1000},

    # HC: step_size (bán kính tìm kiếm lân cận)
    "HC": {"step_size": 0.5, "max_iter": 100},

    # DE: pop_size (kích thước quần thể)
    "DE": {"pop_size": 30, "max_iter": 50},

    # TLBO: population_size (kích thước lớp học)
    "TLBO": {"population_size": 30, "max_iter": 50},
    
    # Benchmarking parameter for Continuous Optimization functions 
    "Continuous_Optimization": {"runs": 18, "max_iter": 366, "dim": 18}
}


if __name__ == "__main__":
    prob = get_problem("rastrigin")
    perfect_input = np.zeros(10)
    print(f"Test Rastrigin tại 0: {prob['func'](perfect_input)} (Kỳ vọng: 0.0)")
    random_input = np.random.uniform(prob['lb'], prob['ub'], 10)
    print(f"Test Rastrigin ngẫu nhiên: {prob['func'](random_input)}")