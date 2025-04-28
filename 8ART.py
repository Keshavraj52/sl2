import numpy as np

class ART1:
    def __init__(self, input_size, vigilance=0.8):
        self.input_size = input_size
        self.vigilance = vigilance
        self.weights = []
    
    def _match_score(self, weight, input_pattern):
        numerator = np.sum(np.minimum(weight, input_pattern))
        denominator = np.sum(input_pattern)
        return numerator / denominator if denominator != 0 else 0
    
    def train(self, patterns):
        for pattern in patterns:
            matched = False
            for i, weight in enumerate(self.weights):
                score = self._match_score(weight, pattern)
                if score >= self.vigilance:
                    # Update the weight vector (AND operation)
                    self.weights[i] = np.minimum(weight, pattern)
                    matched = True
                    print(f"Pattern {pattern} matched with Cluster {i}")
                    break
            if not matched:
                # Create a new cluster
                self.weights.append(np.copy(pattern))
                print(f"Pattern {pattern} created a new Cluster {len(self.weights) - 1}")
    
    def predict(self, pattern):
        for i, weight in enumerate(self.weights):
            score = self._match_score(weight, pattern)
            if score >= self.vigilance:
                return i
        return None

# Example Usage

# Define input binary patterns
patterns = np.array([
    [1, 0, 0, 1, 1],
    [1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [1, 1, 0, 1, 1]
])

# Initialize ART1 network
art = ART1(input_size=5, vigilance=0.6)

# Train on the patterns
art.train(patterns)

# Test prediction
test_pattern = np.array([1, 0, 0, 1, 1])
cluster = art.predict(test_pattern)
print(f"\nTest pattern {test_pattern} belongs to Cluster {cluster}")
