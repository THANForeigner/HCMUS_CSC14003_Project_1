import numpy as np

def sphere(x):
    return np.sum(x ** 2)

def rastrigin(x):
    dim = len(x)
    return 10 * dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def ackley(x):
    dim = len(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / dim))
    term2 = -np.exp(np.sum(np.cos(c * x)) / dim)
    return term1 + term2 + a + np.e

def easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))

func_config = {
    "sphere": {
        "func": sphere,
        "lb": -5.12, "ub": 5.12,
        "optimal": 0.0
    },
    "rastrigin": {
        "func": rastrigin,
        "lb": -5.12, "ub": 5.12,
        "optimal": 0.0
    },
    "rosenbrock": {
        "func": rosenbrock,
        "lb": -5.0, "ub": 10.0,
        "optimal": 0.0
    },
    "ackley": {
        "func": ackley,
        "lb": -32.768,
        "ub": 32.768,
        "optimal": 0.0
    },
    "easom": {
        "func": easom,
        "lb": -100,
        "ub": 100,
        "optimal": -1.0
    }
}

def get_problem(name):
    if name not in func_config:
        raise ValueError(f"Không tìm thấy hàm tên: {name}")
    return func_config[name]


if __name__ == "__main__":
    prob = get_problem("rastrigin")
    perfect_input = np.zeros(10)
    print(f"Test Rastrigin tại 0: {prob['func'](perfect_input)} (Kỳ vọng: 0.0)")
    random_input = np.random.uniform(prob['lb'], prob['ub'], 10)
    print(f"Test Rastrigin ngẫu nhiên: {prob['func'](random_input)}")