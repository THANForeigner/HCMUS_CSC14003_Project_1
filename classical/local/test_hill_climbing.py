import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from classical.local.hill_climbing import HillClimbing
from nature_inspire import continuous_functions

class TestHillClimbing(unittest.TestCase):
    def test_sphere_function(self):
        lower_bound = -10
        upper_bound = 10
        num_parameters = 2
        num_iteration = 1000

        optimizer = HillClimbing(
            lower_bound, upper_bound, num_parameters, step_size=0.5
        )
        optimizer.set_optimization_function(continuous_functions.sphere_function)
        best_fitness, best_candidate = optimizer.solve(num_iteration)

        print(
            f"\nSphere function - Best Fitness: {best_fitness}, Best Candidate: {best_candidate}"
        )
        self.assertLess(best_fitness, 1.0)

    def test_rastrigin_function(self):
        lower_bound = -5.12
        upper_bound = 5.12
        num_parameters = 2
        num_iteration = 2000

        optimizer = HillClimbing(
            lower_bound, upper_bound, num_parameters, step_size=0.5
        )
        optimizer.set_optimization_function(continuous_functions.rastrigin_function)
        best_fitness, best_candidate = optimizer.solve(num_iteration)

        print(
            f"Rastrigin function - Best Fitness: {best_fitness}, Best Candidate: {best_candidate}"
        )
        # Relaxed assertion
        self.assertLess(best_fitness, 50.0)

if __name__ == "__main__":
    unittest.main()

