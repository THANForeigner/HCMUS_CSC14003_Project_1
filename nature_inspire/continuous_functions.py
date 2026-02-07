import math
import numpy as np

#Sphere function formular
def sphere_function(coords):
    return sum(x**2 for x in coords) 

#Rastrigin function formular
def rastrigin_function(coords):
    A = 10
    return A * len(coords) + sum([(x**2 - A * math.cos(2 * math.pi * x)) for x in coords]) 

# Rosenbrock function formula
def rosenbrock_function(coords):
    return sum(100 * (coords[i+1] - coords[i]**2)**2 + (1 - coords[i])**2 for i in range(len(coords) - 1))

# Griewank function formula
def griewank_function(coords):
    term_1 = sum(x**2 for x in coords) / 4000
    term_2 = math.prod(math.cos(x / math.sqrt(i + 1)) for i, x in enumerate(coords))
    return 1 + term_1 - term_2

# Ackley function formula
def ackley_function(coords):
    n = len(coords)
    term_1 = -20 * math.exp(-0.2 * math.sqrt(sum(x**2 for x in coords) / n))
    term_2 = -math.exp(sum(math.cos(2 * math.pi * x) for x in coords) / n)
    return term_1 + term_2 + 20 + math.e