import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from classical.local.hill_climbing_tsp import HillClimbingTSP

class TestHillClimbingTSP(unittest.TestCase):
    def test_small_tsp(self):
        # A simple 4-city problem (square)
        # 0 -- 1
        # |    |
        # 3 -- 2
        # Distances: (0,1)=1, (1,2)=1, (2,3)=1, (3,0)=1, (0,2)=sqrt(2), (1,3)=sqrt(2)
        matrix = [
            [0, 1, 1.414, 1],
            [1, 0, 1, 1.414],
            [1.414, 1, 0, 1],
            [1, 1.414, 1, 0]
        ]
        
        solver = HillClimbingTSP(matrix, max_iterations=100, restarts=5)
        cost, path = solver.solve()
        
        print(f"\nSmall TSP - Cost: {cost}, Path: {path}")
        # The optimal cost for this square should be 4.0
        self.assertAlmostEqual(cost, 4.0, places=2)
        self.assertEqual(len(path), 4)
        self.assertEqual(len(set(path)), 4)

    def test_trivial_tsp(self):
        matrix = [[0]]
        solver = HillClimbingTSP(matrix)
        cost, path = solver.solve()
        self.assertEqual(cost, 0.0)
        self.assertEqual(path, [0])

    def test_two_cities_tsp(self):
        matrix = [
            [0, 5],
            [5, 0]
        ]
        solver = HillClimbingTSP(matrix)
        cost, path = solver.solve()
        # Cost is 0->1 + 1->0 = 5 + 5 = 10
        self.assertEqual(cost, 10.0)
        self.assertEqual(len(path), 2)

if __name__ == "__main__":
    unittest.main()
