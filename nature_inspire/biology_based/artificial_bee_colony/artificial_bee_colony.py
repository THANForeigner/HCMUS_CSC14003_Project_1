import random
import numpy as np
import copy


class Bee:
    # Base class representing a generic Bee and a Food Source.
    # In ABC, each Employed Bee is attached to exactly one food source.
    def __init__(self, bounds, function, limit=-1, dimension=-1, coords=None):
        self.bounds = bounds
        self.function = function
        self.dimension = 2 if dimension == -1 else dimension
        self.coords = self.randposition() if coords is None else coords
        self.fitness = self.getfitness(self.coords)
        self.trial = 0
        self.limit = 3 if limit == -1 else limit

    def randposition(self):
        return np.array([
            self.bounds[0] + random.uniform(0, 1) * (self.bounds[1] - self.bounds[0])
            for _ in range(self.dimension)
        ])

    def getfitness(self, coords):
        # Evaluates the objective function and converts it to a fitness value.
        # This specific conversion implies we are solving a MINIMIZATION problem.
        # (Smaller objective function value -> Higher fitness)
        fi = self.function(coords.tolist())
        fitness = None
        if fi >= 0:
            fitness = 1 / (1 + fi)
        else:
            fitness = 1 + abs(fi)
        return fitness

    def _explore(self, partner_coords):
        # The core search mechanism of ABC. Creates a new candidate solution
        # by modifying one randomly chosen dimension using information from a partner bee.
        new_coords = self.coords.copy()
        j = random.randint(0, self.dimension - 1)
        phi = random.uniform(-1, 1)
        new_coords[j] = self.coords[j] + phi * (self.coords[j] - partner_coords[j])
        new_coords[j] = np.clip(new_coords[j], self.bounds[0], self.bounds[1])
        new_fitness = self.getfitness(new_coords)
        if self.fitness < new_fitness:
            self.fitness = new_fitness
            self.coords = new_coords
            self.trial = 0
            return True
        else:
            self.trial += 1
            return False

    def reset(self):
        # Used in the Scout Bee phase to abandon an exhausted food source and find a new random one.
        self.coords = self.randposition()
        self.fitness = self.getfitness(self.coords)
        self.trial = 0


class EmployeeBee(Bee):
    # Employed bees explore the neighborhood of their currently assigned food source.
    def explore(self, partner_bee):
        self._explore(partner_bee.coords)


class OnlookerBee(Bee):
    # Onlooker bees wait in the hive and choose food sources based on probability (fitness).
    def explore(self, selected_bee, partner_bee):
        self.coords = selected_bee.coords.copy()
        self.fitness = selected_bee.fitness
        self.trial = selected_bee.trial
        is_improved = self._explore(partner_bee.coords)
        if is_improved:
            selected_bee.coords = self.coords.copy()
            selected_bee.fitness = self.fitness
        selected_bee.trial = self.trial


class ABC:
    def __init__(self, function, ranges: list, dimension=-1, swarm_size=-1, limit=-1, max_iteration=-1):
        self.function = function
        self.ranges = ranges
        self.dimension = 2 if dimension == -1 else dimension
        self.swarm_size = 100 if swarm_size == -1 else swarm_size
        self.food_size = int(self.swarm_size / 2)
        self.trial_limit = limit
        self.iter = 0
        self.max_iter = max_iteration
        # In ABC, half the swarm is Employed Bees (equal to number of food sources), half is Onlooker Bees
        # Initiate food sources and employed bees at the food sources
        self.employee_bees = [
            EmployeeBee(self.ranges, self.function, self.trial_limit, self.dimension)
            for _ in range(0, self.food_size)
        ]
        # Initiate onlooker bees
        self.onlooker_bees = [
            OnlookerBee(self.ranges, self.function, self.trial_limit, self.dimension)
            for _ in range(self.food_size)
        ]

        # Find the best food source in the initial population
        initial_best = max(self.employee_bees, key=lambda bee: bee.fitness)
        self.best_bee = copy.deepcopy(initial_best)
        self.fitness_history = []
        self.history = []

    def run(self):
        self.history.append([bee.coords.copy() for bee in self.employee_bees])

        while self.iter < self.max_iter:

            # 1. EMPLOYED BEE PHASE
            # Every employed bee explores near its current food source
            for i, bee in enumerate(self.employee_bees):
                candidates = [b for idx, b in enumerate(self.employee_bees) if idx != i]
                partner = random.choice(candidates)
                bee.explore(partner)

            # Calculate probabilities for Roulette Wheel Selection
            overall_fitness = sum(bee.fitness for bee in self.employee_bees)
            probs = [bee.fitness / overall_fitness for bee in self.employee_bees]
            # Select food sources based on their fitness weights
            selected_sources = random.choices(range(self.food_size), weights=probs, k=self.food_size)

            # 2. ONLOOKER BEE PHASE
            # Onlookers are deployed to the selected sources
            for onlooker, idx in zip(self.onlooker_bees, selected_sources):
                selected_bee = self.employee_bees[idx]
                candidates = [b for i, b in enumerate(self.employee_bees) if i != idx]
                partner_bee = random.choice(candidates)
                onlooker.explore(selected_bee, partner_bee)

            # 3. SCOUT BEE PHASE
            # Abandon food sources that have exceeded the trial limit
            for bee in self.employee_bees:
                if bee.trial > self.trial_limit:
                    bee.reset()

            current_best_bee = max(self.employee_bees, key=lambda bee: bee.fitness)
            if current_best_bee.fitness > self.best_bee.fitness:
                self.best_bee = copy.deepcopy(current_best_bee)

            self.iter += 1
            self.history.append([bee.coords.copy() for bee in self.employee_bees])