import math
import numpy as np

def sphere(x):
    x = np.array(x)
    return np.sum(x ** 2)

def rastrigin(x):
    x = np.array(x)
    dim = len(x)
    return 10 * dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def ackley(x):
    x = np.array(x)
    dim = len(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / dim))
    term2 = -np.exp(np.sum(np.cos(c * x)) / dim)
    return term1 + term2 + a + np.e

def easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))

def griewank(x):
    x = np.array(x)
    term_1 = np.sum(x ** 2) / 4000
    term_2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + term_1 - term_2