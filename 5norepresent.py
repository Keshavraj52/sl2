import numpy as np
import matplotlib.pyplot as plt

# Define the 5x3 matrices for 0, 1, 2, 3, 9
digits = {
    0: [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ],
    1: [
        [0, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1]
    ],
    2: [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ],
    3: [
        [1, 1, 1],
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ],
    9: [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
}

# Flatten the matrices
X = np.array([np.array(digits[d]).flatten() for d in digits])
y = np.array([0, 1, 2, 3, 9])  # Labels corresponding to the digits

# One-hot encode the labels for multiclass
y_encoded = np.zeros((len(y), 5))
for i, label in enumerate(y):
    y_encoded[i, i] = 1

# Initialize weights and bias
input_size = 15
output_size = 5  # because we have 5 classes (0,1,2,3,9)
weights = np.random.randn(input_size, output_size) * 0.01
bias = np.zeros((1, output_size))

# Activation function: Softmax
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

# Training parameters
learning_rate = 0.1
epochs = 500

# Training loop
for epoch in range(epochs):
    # Forward pass
    z = np.dot(X, weights) + bias
    y_pred = softmax(z)

    # Loss (cross-entropy) - optional to print

    # Backward pass (gradient descent)
    error = y_pred - y_encoded
    weights -= learning_rate * np.dot(X.T, error) / X.shape[0]
    bias -= learning_rate * np.sum(error, axis=0, keepdims=True) / X.shape[0]

print("Training Complete!")

# Prediction function
def predict(input_matrix):
    input_vector = np.array(input_matrix).flatten().reshape(1, -1)
    z = np.dot(input_vector, weights) + bias
    y_pred = softmax(z)
    predicted_class = np.argmax(y_pred)
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 9}
    return mapping[predicted_class]

# Test the model
test_samples = {
    "Zero": digits[0],
    "One": digits[1],
    "Two": digits[2],
    "Three": digits[3],
    "Nine": digits[9]
}

print("\nTesting the trained network:")
for name, matrix in test_samples.items():
    prediction = predict(matrix)
    print(f"{name} is recognized as: {prediction}")
