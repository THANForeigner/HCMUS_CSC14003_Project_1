import unittest
from nature_inspire.human_based.teaching_learning_based_optimization import (
    continuous_function as tlbo,
)
from nature_inspire import continuous_functions


class Test(unittest.TestCase):
    def test_sphere_function(self):
        lower_bound = -10
        upper_bound = 10
        num_parameters = 2
        population_size = 50
        num_iteration = 100

        optimizer = tlbo.TeachingLearingBasedOptimization(
            lower_bound, upper_bound, num_parameters, population_size
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
        population_size = 50
        num_iteration = 200

        optimizer = tlbo.TeachingLearingBasedOptimization(
            lower_bound, upper_bound, num_parameters, population_size
        )
        optimizer.set_optimization_function(continuous_functions.rastrigin_function)
        best_fitness, best_candidate = optimizer.solve(num_iteration)

        print(
            f"Rastrigin function - Best Fitness: {best_fitness}, Best Candidate: {best_candidate}"
        )
        self.assertLess(best_fitness, 5.0)


if __name__ == "__main__":
    unittest.main()
