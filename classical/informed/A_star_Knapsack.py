import heapq

class Item:
    def __init__(self, index, weight, value):
        self.index = index
        self.weight = weight
        self.value = value
        self.ratio = value / weight if weight > 0 else 0

    def __lt__(self, other):
        return self.ratio > other.ratio # Descending sort

class AStarKnapsack:
    def __init__(self, capacity, weights, values):
        self.capacity = capacity
        # Keep track of original index
        self.items = [Item(i, w, v) for i, (w, v) in enumerate(zip(weights, values))]
        self.n = len(self.items)
        # Sort items by value/weight ratio in descending order for linear relaxation
        self.items.sort()

    def run(self):
        # Priority Queue stores tuples: (-upper_bound, -current_value, level, current_weight, taken_indices_list)
        # Using negative for max-heap behavior on UB and Value (since heapq is min-heap)
        
        # Initial call
        # Level -1 means no items considered yet
        initial_ub = self._calculate_upper_bound(-1, 0, self.capacity)
        
        # Priority Queue
        pq = []
        heapq.heappush(pq, (-initial_ub, 0, -1, 0, []))
        
        max_profit = 0
        best_selection_indices = []
        
        while pq:
            neg_ub, neg_val, level, current_weight, selection = heapq.heappop(pq)
            upper_bound = -neg_ub
            current_value = -neg_val
            
            # Pruning: if the best potential of this branch is less than what we already found
            # Note: since we use float for UB, use exact comparison carefully
            if upper_bound <= max_profit:
                continue
            
            # If we reached the last item
            if level == self.n - 1:
                if current_value > max_profit:
                    max_profit = current_value
                    best_selection_indices = selection
                continue
            
            next_level = level + 1
            item = self.items[next_level]
            
            # Branch 1: Take the item (if fits)
            if current_weight + item.weight <= self.capacity:
                new_weight = current_weight + item.weight
                new_value = current_value + item.value
                new_selection = selection + [item.index]
                
                # If this new node is better than current max, update
                if new_value > max_profit:
                    max_profit = new_value
                    best_selection_indices = new_selection
                
                # Calculate UB for taking the item
                take_ub = self._calculate_upper_bound(next_level, new_value, self.capacity - new_weight)
                
                if take_ub > max_profit:
                    heapq.heappush(pq, (-take_ub, -new_value, next_level, new_weight, new_selection))
            
            # Branch 2: Don't take the item
            # UB needs to be recalculated because we skipped an item we "should" have taken by greedy ratio
            skip_ub = self._calculate_upper_bound(next_level, current_value, self.capacity - current_weight)
            
            if skip_ub > max_profit:
                # push -current_value which is same as neg_val
                heapq.heappush(pq, (-skip_ub, -current_value, next_level, current_weight, selection))

        # Construct final 0/1 selection based on original indices
        final_bits = [0] * self.n
        for idx in best_selection_indices:
            final_bits[idx] = 1
            
        return max_profit, final_bits

    def _calculate_upper_bound(self, current_level, current_value, remaining_capacity):
        if remaining_capacity < 0:
            return 0
            
        upper_bound = current_value
        temp_capacity = remaining_capacity
        
        # Consider items from next level onwards
        # Items are already sorted by ratio desc
        for i in range(current_level + 1, self.n):
            item = self.items[i]
            if item.weight <= temp_capacity:
                upper_bound += item.value
                temp_capacity -= item.weight
            else:
                # Fractional part (Linear Relaxation)
                upper_bound += item.value * (temp_capacity / item.weight)
                break
                
        return upper_bound
