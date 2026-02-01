import math
import numpy as np

def sphere_function(coords):
    return sum(x**2 for x in coords)

def rastrigin_function(coords):
    A = 10
    return A * len(coords) + sum([(x**2 - A * math.cos(2 * math.pi * x)) for x in coords])
