import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = p.reshape(-1, 1)  # Make it a column vector
            self.weights += np.dot(p, p.T)
        np.fill_diagonal(self.weights, 0)  # No self-connections
        self.weights /= self.size  # Optional: normalize by size for stability

    def recall(self, pattern, steps=5):
        pattern = pattern.copy()
        for _ in range(steps):
            for i in range(self.size):
                raw = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if raw >= 0 else -1
        return pattern

# Define 4 patterns to store
patterns = np.array([
    [1, -1, 1, -1, 1, -1, 1, -1],
    [1, 1, -1, -1, 1, 1, -1, -1],
    [-1, -1, 1, 1, -1, -1, 1, 1],
    [1, 1, 1, 1, -1, -1, -1, -1]
])

# Create Hopfield network
hopfield_net = HopfieldNetwork(size=patterns.shape[1])

# Train the network
hopfield_net.train(patterns)

# Test recall
print("Testing Recall:")
for test_pattern in patterns:
    output = hopfield_net.recall(test_pattern)
    print("Input :", test_pattern)
    print("Output:", output)
    print()
