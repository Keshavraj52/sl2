import numpy as np

# Simple 7x5 ASCII patterns for digits 0-9
digits = {
    0: [
        "01110",
        "10001",
        "10011",
        "10101",
        "11001",
        "10001",
        "01110"
    ],
    1: [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110"
    ],
    2: [
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111"
    ],
    3: [
        "01110",
        "10001",
        "00001",
        "00110",
        "00001",
        "10001",
        "01110"
    ],
    4: [
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010"
    ],
    5: [
        "11111",
        "10000",
        "11110",
        "00001",
        "00001",
        "10001",
        "01110"
    ],
    6: [
        "00110",
        "01000",
        "10000",
        "11110",
        "10001",
        "10001",
        "01110"
    ],
    7: [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "01000",
        "01000"
    ],
    8: [
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110"
    ],
    9: [
        "01110",
        "10001",
        "10001",
        "01111",
        "00001",
        "00010",
        "01100"
    ]
}

# Convert ASCII art to flattened binary vectors
def ascii_to_vector(ascii_digit):
    return np.array([int(pixel) for row in ascii_digit for pixel in row])

# Prepare data
X = np.array([ascii_to_vector(digits[d]) for d in range(10)])  # shape (10, 35)
y = np.array([d % 2 for d in range(10)])  # Even=0, Odd=1

# Initialize perceptron parameters
weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.1
epochs = 20

# Training loop
for epoch in range(epochs):
    for i in range(len(X)):
        # Weighted sum
        linear_output = np.dot(X[i], weights) + bias
        # Activation function (step function)
        prediction = 1 if linear_output >= 0 else 0
        # Update weights and bias
        error = y[i] - prediction
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

print("Training complete!")

# Test model
print("\nTesting on digits 0-9:")
for d in range(10):
    input_vector = ascii_to_vector(digits[d])
    linear_output = np.dot(input_vector, weights) + bias
    prediction = 1 if linear_output >= 0 else 0
    label = "Odd" if prediction == 1 else "Even"
    print(f"Digit {d}: Predicted as {label}")

